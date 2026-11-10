#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HTTP Client Policy
HTTP客户端策略，用于与远程策略服务器通信
"""

import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
from pathlib import Path
from uuid import uuid4

from policy.policies.base_policy import BasePolicy, Observation, Action, WireConfig

logger = logging.getLogger(__name__)


class HTTPClientPolicy(BasePolicy):
    """
    HTTP客户端策略包装器
    
    用于与远程策略服务器通信，不处理具体的图像预处理逻辑
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 24851,
        # Needs to tolerate cold-start torch.compile on chunk 0 (~75s on
        # first call). Default generously; pass a smaller value at call sites
        # that know the server is warm.
        timeout: int = 600,
        max_retries: int = 3,
        session_id: str | None = None,
        **kwargs,
    ):
        """
        初始化HTTP客户端
        
        Args:
            host: HTTP服务器主机地址
            port: HTTP服务器端口
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            session_id: 唯一客户端会话 ID；默认自动生成
        """
        self._base_url = f"http://{host}:{port}"
        self._timeout = timeout
        self._session_id = session_id or f"http-{uuid4().hex}"
        if not self._session_id:
            raise ValueError("session_id must not be empty")
        
        # 创建带重试的 session
        self._session = requests.Session()
        self._session.headers.update({"X-Session-Id": self._session_id})
        # 禁用代理，直接连接到本地SSH隧道端口
        self._session.proxies = {
            "http": None,
            "https": None
        }
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        
        # 缓存的观察数据
        self._cached_observation = None
        
        logger.info(
            "HTTP client connected to %s session_id=%s",
            self._base_url,
            self._session_id,
        )
    
    def update_obs(self, observation: Observation):
        """
        更新观察（缓存观察数据）
        
        Args:
            observation: Observation对象
        """
        self._cached_observation = observation
    
    def get_action(self) -> Action:
        """
        获取动作预测
        
        Returns:
            Action对象，包含动作数组
        """
        if self._cached_observation is None:
            raise ValueError("No observation cached. Call update_obs first.")
        
        # 序列化观察数据
        data = self._cached_observation.to_bytes()
        
        # 发送请求
        url = f"{self._base_url}/infer"
        try:
            response = self._session.post(
                url,
                data=data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=self._timeout,
                proxies={"http": None, "https": None}  # 明确禁用代理
            )
            response.raise_for_status()
            
            # 解析响应
            action = Action.from_bytes(response.content)
            action._remote_inference_time = response.headers.get('X-Inference-Time')
            return action
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Server request failed: {e}")
    
    def reset(self):
        """重置服务器端策略状态"""
        try:
            response = self._session.post(
                f"{self._base_url}/reset",
                timeout=5,
                proxies={"http": None, "https": None}  # 明确禁用代理
            )
            response.raise_for_status()
            logger.debug("Server policy state reset successfully")
        except Exception as e:
            logger.debug(f"Could not reset server policy state: {e}")

    def get_wire_config(self) -> WireConfig:
        """通过 HTTP GET /config 拉取远端策略的 wire config（handshake）。"""
        response = self._session.get(
            f"{self._base_url}/config",
            timeout=self._timeout,
            proxies={"http": None, "https": None},
        )
        response.raise_for_status()
        return WireConfig.from_dict(response.json())
