#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Policy Interface
定义所有策略必须实现的抽象接口
"""

from abc import ABC, abstractmethod
import sys
from pathlib import Path

# 添加父目录到路径以导入 utils
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from utils.message import Observation, Action, WireConfig


class BasePolicy(ABC):
    """
    策略抽象基类
    
    所有具体的策略实现必须继承此类并实现以下方法：
    - __init__: 初始化策略（加载模型等）
    - reset: 重置策略状态
    - update_obs: 更新观察数据
    - get_action: 获取动作预测
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        初始化策略
        
        Args:
            **kwargs: 策略特定的初始化参数（由具体实现定义）
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        重置策略状态
        
        在开始新任务或重置环境时调用，用于清除内部状态缓存
        """
        pass
    
    @abstractmethod
    def update_obs(self, observation: Observation):
        """
        更新观察数据
        
        Args:
            observation: Observation对象，包含图像、状态、指令等信息
        """
        pass
    
    @abstractmethod
    def get_action(self) -> Action:
        """
        获取动作预测

        Returns:
            action: Action对象，包含动作数组和其他信息
        """
        pass

    def get_wire_config(self) -> WireConfig:
        """
        返回 executor 应如何准备并发送 Observation 的配置。

        Workers 启动时通过 HTTP handshake 一次性拉取，然后传给 executor。
        默认值适合单帧非 AR 策略；具体策略应 override 以反映其训练先验。
        """
        return WireConfig()

    def set_instruction(self, instruction: str) -> None:
        """SEP session 启动时调用一次，缓存自然语言指令。

        旧路径每帧 obs 都带 instruction，新路径改为 session 级一次性下发。
        默认 no-op，需要的策略自行 override。
        """
        return None

    def get_policy_spec(self) -> dict:
        """返回供 SEP 注册时做兼容性自检的轻量描述。

        默认返回空 dict；具体策略可填入诸如 ``policy_id`` /
        ``action_type`` / ``chunk_len`` / ``camera_views`` 之类字段，
        让 SEP 在派发 obs 之前就能拦下不兼容的 slave。
        """
        return {}

    def set_session_id(self, session_id: str) -> None:
        """Bind subsequent inference to one runtime session."""
        return None

    def close(self) -> None:
        """Release policy-owned clients and other runtime resources."""
        return None
