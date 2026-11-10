#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controller Interface
控制器接口，定义所有机器人控制器需要实现的函数

重要约定：
- Controller 对外只暴露统一的高层数据结构：`utils.message.Observation` / `utils.message.Action`
- 底层原始数据（ROS 消息、numpy 数组等）只能在具体 controller / ros_operator 内部使用
"""
from typing import Dict, Any, Optional

import numpy as np

from utils.message import Observation, Action


class Controller:
    """
    抽象的控制器接口
    
    所有具体的机器人控制器必须继承此类并实现以下方法：
    - get_observation: 获取当前观察数据（非破坏性，返回 Observation）
    - publish_action: 发布高层动作（接收 Action）
    - start: 启动控制器（如果需要）
    - stop: 停止控制器（如果需要）
    - reset: 复位机器人到初始位置
    """

    def __init__(self, config: Dict[str, Any], in_collect: bool = False):
        self.config = config
        self.in_collect = in_collect
        assert "action_type" in config, "action_type not set"
        assert config["action_type"] in ("qpos", "eef"), f"bad action_type: {config['action_type']!r}"
        self.action_type = config["action_type"]

    def get_observation(self) -> Optional[Observation]:
        """
        获取当前观察数据（非破坏性）
        
        具体控制器应返回 `utils.message.Observation`，
        这里给出一个用于文档和测试的缺省实现。
        """
        images = {
            "cam_high": np.zeros((736, 640, 3), dtype=np.uint8),
        }
        state = np.zeros(14, dtype=np.float32)
        return Observation(images=images, state=state)

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        """录制专用：返回实时相机帧 {name: HWC uint8}，不读 SDK 状态。

        缺省返回空 dict：录制线程会回退到推理侧 latest_obs.images，
        保持原行为不变。具备后台相机 buffer 的 controller（如 astribot_s1）
        应覆盖此方法，直接返回 buffer 拷贝，使落盘视频以相机原生帧率刷新、
        与推理节奏解耦。
        """
        return {}

    def start_recording_feedback(self) -> None:
        """录制专用：按需启动机器人 feedback 采样。

        缺省为空实现，保持其他 controller 行为不变。需要更高频录制
        feedback 的 controller 可覆盖它，在 session 录制期间维护轻量缓存。
        """

    def stop_recording_feedback(self) -> None:
        """录制专用：停止机器人 feedback 采样。"""

    def get_recording_feedback(self) -> Optional[Dict[str, Any]]:
        """录制专用：返回最新 feedback 缓存。

        返回 dict 可包含：
            joint_state: Optional[np.ndarray]
            eef_state: Optional[np.ndarray]
            wall_ts: Optional[float]
            mono_ns: Optional[int]
            seq: Optional[int]

        缺省返回 None，LocalRecorder 会回退到 agent latest cache。
        """
        return None

    def publish_action(self, action: Action):
        """
        发布动作到机器人
        
        Args:
            action: 高层策略产生的 `Action` 对象
        """
        # 缺省实现仅打印，真实机器人请在子类中覆盖
        print(f"[Controller] Action: {action.action}")

    def start(self):
        """
        启动控制器（如果需要）
        """
        pass

    def stop(self):
        """
        停止控制器（如果需要）
        """
        pass

    def reset(self):
        """
        复位机器人到初始位置
        
        从配置文件中读取复位参数，具体参数由子类实现定义。
        """
        raise NotImplementedError("reset method must be implemented by subclass")