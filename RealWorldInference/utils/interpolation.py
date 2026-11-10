#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Action Interpolation Utilities
动作插值工具类
"""

from abc import ABC, abstractmethod
import inspect
import sys
import numpy as np
from typing import Optional, Dict, Any, Type, List, Tuple


# ===================== 四元数 SLERP 工具 =====================


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    两个单位四元数之间的球面线性插值 (SLERP)。

    自动处理双覆盖（始终走最短弧）；当两四元数几乎重合时退化为
    归一化线性插值 (nlerp) 以避免 sin(theta) 趋零的数值问题。

    @param q0: (4,) 起点四元数 (xyzw)
    @param q1: (4,) 终点四元数 (xyzw)
    @param t: 插值参数，范围 [0, 1]
    @return: (4,) 插值后的单位四元数
    """
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = np.dot(q0, q1)

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / (np.linalg.norm(result) + 1e-12)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    result = a * q0 + b * q1
    return result / (np.linalg.norm(result) + 1e-12)


def _slerp_sequence(q0: np.ndarray, q1: np.ndarray, n_steps: int) -> np.ndarray:
    """
    生成 n_steps 个 SLERP 插值四元数（t 均匀分布，不含起点、含终点）。

    时间采样约定与标量插值器一致：t = linspace(0, 1, n_steps+1)[1:]，
    这样 prev（已执行过的起点）不会被重复发送。

    @param q0: (4,) 起点四元数 (xyzw)
    @param q1: (4,) 终点四元数 (xyzw)
    @param n_steps: 步数
    @return: (n_steps, 4) 插值后的四元数序列
    """
    if n_steps <= 0:
        return np.empty((0, 4), dtype=np.float64)
    ts = np.linspace(0, 1, n_steps + 1)[1:]
    return np.stack([_slerp(q0, q1, t) for t in ts], axis=0)


# 公开别名：便于 executor 等外部模块复用
slerp = _slerp
slerp_sequence = _slerp_sequence


def _resolve_steps(default_steps: int, num_steps: Optional[int] = None) -> int:
    steps = default_steps if num_steps is None else int(num_steps)
    return max(1, steps)


class Interpolator(ABC):
    @abstractmethod
    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        插值方法
        """
        raise NotImplementedError

class LinearInterpolation(Interpolator):
    """
    线性插值类
    
    用于在动作之间进行线性插值，确保动作平滑过渡
    """
    
    def __init__(self, arm_steps_length: list, publish_rate: int = 1000, action_rate: int = 0):
        """
        初始化线性插值器
        
        Args:
            arm_steps_length: 每个关节的步长限制列表 (7个元素，对应单臂7个关节)
            publish_rate: 发布频率（Hz），默认1000Hz
        """
        self.arm_steps_length = np.array(arm_steps_length)
        self.publish_rate = publish_rate
    
    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        对动作进行线性插值
        
        Args:
            prev_action: 上一个动作，形状为 (14,) 或 (7,)
            cur_action: 当前动作，形状为 (14,) 或 (7,)
            
        Returns:
            插值后的动作数组，形状为 (N, 14) 或 (N, 7)
        """
        prev_action = np.array(prev_action)
        cur_action = np.array(cur_action)
        
        # 确保两个动作形状一致
        if prev_action.shape != cur_action.shape:
            raise ValueError(f"Action shape mismatch: {prev_action.shape} vs {cur_action.shape}")
        
        # 如果是单臂（7个关节），扩展到双臂（14个关节）
        if len(prev_action) == 7:
            steps = np.array(list(self.arm_steps_length) + list(self.arm_steps_length))
        elif len(prev_action) == 14:
            steps = np.array(list(self.arm_steps_length) + list(self.arm_steps_length))
        else:
            raise ValueError(f"Unsupported action dimension: {len(prev_action)}")
        
        diff = np.abs(cur_action - prev_action)
        step = np.ceil(diff / steps).astype(int)
        max_step = _resolve_steps(int(np.max(step)), num_steps)
        
        if max_step <= 1:
            return cur_action[np.newaxis, :]
        
        new_actions = np.linspace(prev_action, cur_action, max_step + 1)
        return new_actions[1:]

class IDMInterpolation(Interpolator):
    """
    IDM 风格的异步线性插值类
    
    允许每个关节以独立的速度进行插值，而不是所有关节同步到达目标。
    """
    
    def __init__(self, arm_steps_length: list, publish_rate: int = 1000):
        """
        初始化 IDM 插值器
        
        Args:
            arm_steps_length: 每个关节的步长限制列表 (7个元素，对应单臂7个关节)
            publish_rate: 发布频率（Hz），默认1000Hz
        """
        self.arm_steps_length = np.array(arm_steps_length)
        self.publish_rate = publish_rate
        
        # 每个关节的硬编码速度因子
        # 针对双臂 (14维) 的速度因子
        self.speed_factors_14 = np.array([
            1, 1, 1, 1, 1, 1, 1.0,  # Left Arm: joints 1-6 + gripper
            1, 1, 1, 1, 1, 1, 1.0   # Right Arm: joints 1-6 + gripper
        ], dtype=np.float64)
        
        # 针对单臂 (7维) 的速度因子
        self.speed_factors_7 = np.array([
            1, 1, 1, 1, 1, 1, 1.0   # Arm: joints 1-6 + gripper
        ], dtype=np.float64)

    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        IDM 风格的异步线性插值，支持独立速度因子
        """
        prev_action = np.array(prev_action, dtype=np.float64)
        cur_action = np.array(cur_action, dtype=np.float64)
        
        # 维度检查与适配
        if prev_action.shape != cur_action.shape:
            raise ValueError(f"Action shape mismatch: {prev_action.shape} vs {cur_action.shape}")
            
        dim = len(prev_action)
        if dim == 7:
            speed_factors = self.speed_factors_7
            base_steps = self.arm_steps_length.astype(np.float64)
        elif dim == 14:
            speed_factors = self.speed_factors_14
            # 这里的 arm_steps_length 是单臂的，需要拼接成双臂
            base_steps = np.concatenate([self.arm_steps_length, self.arm_steps_length]).astype(np.float64)
        else:
            raise ValueError(f"Unsupported action dimension: {dim}")

        diff = cur_action - prev_action
        
        # 根据速度因子计算每个关节的最大步长距离
        if len(base_steps) == len(speed_factors):
            max_step_dist = base_steps * speed_factors
        else:
            max_step_dist = base_steps * 1.0
            
        # 独立计算每个关节所需的步数
        with np.errstate(divide='ignore', invalid='ignore'):
            required_steps = np.abs(diff) / max_step_dist
        required_steps[~np.isfinite(required_steps)] = 0
        
        # 向上取整以获取每个关节的整数步数
        per_joint_steps = np.ceil(required_steps).astype(int)
        per_joint_steps = np.maximum(1, per_joint_steps)
        
        # 总持续时间由最慢的关节决定；显式 num_steps 则让本次调用按统一步数完成。
        if num_steps is None:
            total_steps = _resolve_steps(int(np.max(per_joint_steps)), None)
        else:
            total_steps = _resolve_steps(int(np.max(per_joint_steps)), num_steps)
            per_joint_steps = np.where(np.abs(diff) > 0, total_steps, 1)
        
        if total_steps <= 1:
            return cur_action[np.newaxis, :]
            
        # 生成异步轨迹
        interpolated = []
        current = prev_action.copy()
        
        # 计算每个关节的增量
        increments = np.zeros_like(diff)
        mask = per_joint_steps > 0
        increments[mask] = diff[mask] / per_joint_steps[mask]
        
        for i in range(total_steps):
            # 仅更新尚未完成的关节
            active_mask = (i < per_joint_steps)
            
            current[active_mask] += increments[active_mask]
        
            # 修正已完成的关节
            finished_mask = (i == per_joint_steps - 1)
            if np.any(finished_mask):
                current[finished_mask] = cur_action[finished_mask]
            
            interpolated.append(current.copy())
        
        # 确保最终状态精确
        interpolated[-1] = cur_action
        
        return np.array(interpolated)


class TimeBasedInterpolation(Interpolator):
    """
    基于时间的线性插值类 (固定步数)
    
    无论动作幅度大小，都在固定的时间内（固定的步数）完成过渡。
    这保证了机器人严格按照预定的时间轴运动，适合复现高速动作。
    """
    def __init__(self, publish_rate: int, action_rate: int):
        """
        Args:
            publish_rate: 机器人底层控制频率 (Hz)，例如 1000Hz
            action_rate: 上层动作指令频率 (Hz)，例如 30Hz
        """
        self.publish_rate = publish_rate
        self.action_rate = action_rate
        # 计算两个动作点之间需要插入多少个控制点
        self.steps = int(publish_rate / action_rate)
        if self.steps < 1:
            self.steps = 1
            
    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成固定步数的插值轨迹
        """
        prev_action = np.array(prev_action)
        cur_action = np.array(cur_action)
        
        if prev_action.shape != cur_action.shape:
             raise ValueError(f"Action shape mismatch: {prev_action.shape} vs {cur_action.shape}")

        steps = _resolve_steps(self.steps, num_steps)
        # 生成固定步数的插值
        # linspace 包含起点和终点，我们需要去掉起点（因为起点是 prev_action，已经执行过了）
        # 所以生成 steps + 1 个点，然后取 [1:]
        new_actions = np.linspace(prev_action, cur_action, steps + 1)
        return new_actions[1:]


class PolynomialInterpolation(Interpolator):
    """
    基于时间的多项式插值类 (三次样条)
    
    使用三次多项式在两个动作点之间进行平滑插值。
    保持速度连续性，避免线性插值在转折点处的速度突变。
    """
    def __init__(self, publish_rate: int, action_rate: int):
        """
        Args:
            publish_rate: 机器人底层控制频率 (Hz)
            action_rate: 上层动作指令频率 (Hz)
        """
        self.publish_rate = publish_rate
        self.action_rate = action_rate
        self.steps = int(publish_rate / action_rate)
        if self.steps < 1:
            self.steps = 1
            
        # 记录上一次结束时的速度，用于保证连续性
        self.last_velocity = None
        
    def reset(self):
        """重置内部状态，清除历史速度信息"""
        self.last_velocity = None
        
    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        end_velocity: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成三次样条插值轨迹
        
        Args:
            prev_action: 上一时刻动作
            cur_action: 当前目标动作
            end_velocity: 期望的终点速度。如果为None，则使用启发式方法（平均速度）估计。
                         在Chunk的最后一步建议设为0，以避免停顿时抖动。
        """
        prev_action = np.array(prev_action, dtype=np.float64)
        cur_action = np.array(cur_action, dtype=np.float64)
        
        if prev_action.shape != cur_action.shape:
             raise ValueError(f"Action shape mismatch: {prev_action.shape} vs {cur_action.shape}")
             
        # 初始化速度
        if self.last_velocity is None:
            self.last_velocity = np.zeros_like(prev_action)
            
        # 时间参数
        steps = _resolve_steps(self.steps, num_steps)
        T = steps / self.publish_rate  # 总时间 (秒)
        t = np.linspace(0, T, steps + 1)[1:]  # 时间点 (不包含0)
        
        # 边界条件
        x0 = prev_action
        v0 = self.last_velocity
        x1 = cur_action
        
        # 确定终点速度
        if end_velocity is not None:
            v1 = end_velocity
        else:
            # 启发式终点速度：设为当前段的平均速度，以便平滑过渡到下一段
            # 这样在匀速运动时会退化为线性，在变速时会有平滑曲线
            v1 = (x1 - x0) / T
        # 三次样条系数计算
        # f(t) = a*t^3 + b*t^2 + c*t + d
        # d = x0
        # c = v0
        # a = (2*(x0-x1) + T*(v0+v1)) / T^3
        # b = (3*(x1-x0) - T*(2*v0+v1)) / T^2
        
        d = x0
        c = v0
        a = (2 * (x0 - x1) + T * (v0 + v1)) / (T ** 3)
        b = (3 * (x1 - x0) - T * (2 * v0 + v1)) / (T ** 2)
        
        # 计算轨迹
        # 使用广播机制计算所有关节的轨迹
        # t: (steps,) -> (steps, 1) 以便与 (dim,) 广播
        t_col = t[:, np.newaxis]
        
        new_actions = (a * (t_col ** 3) + 
                       b * (t_col ** 2) + 
                       c * t_col + 
                       d)
        
        # 更新末端速度供下一次使用
        # v(t) = 3*a*t^2 + 2*b*t + c
        self.last_velocity = (3 * a * (T ** 2) + 
                              2 * b * T + 
                              c)
                              
        return new_actions


class MinimumJerkInterpolation(Interpolator):
    """
    基于时间的 Minimum Jerk 插值类
    
    使用五次多项式在两个动作点之间进行插值，最小化加加速度 (Jerk)。
    这种插值方式在每个动作点处速度和加速度都为0，适合点到点的平滑运动，
    但如果用于连续轨迹，会导致速度在每个点都降为0（走走停停）。
    """
    def __init__(self, publish_rate: int, action_rate: int):
        """
        Args:
            publish_rate: 机器人底层控制频率 (Hz)
            action_rate: 上层动作指令频率 (Hz)
        """
        self.publish_rate = publish_rate
        self.action_rate = action_rate
        self.steps = int(publish_rate / action_rate)
        if self.steps < 1:
            self.steps = 1
            
    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成 Minimum Jerk 插值轨迹
        """
        prev_action = np.array(prev_action, dtype=np.float64)
        cur_action = np.array(cur_action, dtype=np.float64)
        
        if prev_action.shape != cur_action.shape:
             raise ValueError(f"Action shape mismatch: {prev_action.shape} vs {cur_action.shape}")
             
        steps = _resolve_steps(self.steps, num_steps)
        # 归一化时间 t \in (0, 1]
        t = np.linspace(0, 1, steps + 1)[1:]
        t_col = t[:, np.newaxis]
        
        # Minimum Jerk 轨迹公式 (假设 v0=a0=vf=af=0)
        # r(t) = x0 + (xf - x0) * (10t^3 - 15t^4 + 6t^5)
        
        dist = cur_action - prev_action
        poly = 10 * (t_col ** 3) - 15 * (t_col ** 4) + 6 * (t_col ** 5)
        
        new_actions = prev_action + dist * poly
                              
        return new_actions


class EefInterpolation(PolynomialInterpolation):
    """
    EEF 混合插值器：位置/头部/夹爪用三次样条，四元数段用 SLERP。

    继承自 PolynomialInterpolation：
    - 先调用父类 interpolate 得到整段 (N, D) 轨迹；
    - 再对 `quat_segments` 指定的每一段 4 维四元数列执行 SLERP，
      覆盖父类给出的线性/多项式插值结果。

    `quat_segments` 必须由调用方传入（通常从 action layout 派生）。
    """

    def __init__(
        self,
        publish_rate: int,
        action_rate: int,
        quat_segments: List[Tuple[int, int]],
    ):
        """
        @param publish_rate: 机器人底层控制频率 (Hz)
        @param action_rate: 上层动作指令频率 (Hz)
        @param quat_segments: 四元数段列表 [(start_index, length), ...]，
                              通常由 executor 侧的 action layout 派生后注入。
        """
        super().__init__(publish_rate=publish_rate, action_rate=action_rate)
        if not quat_segments:
            raise ValueError(
                "EefInterpolation requires a non-empty quat_segments list; "
                "executors should derive it from their action layout and pass "
                "it through the interpolation config."
            )
        self.quat_segments: List[Tuple[int, int]] = [tuple(s) for s in quat_segments]

    def interpolate(
        self,
        prev_action: np.ndarray,
        cur_action: np.ndarray,
        end_velocity: Optional[np.ndarray] = None,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成位置/标量三次样条 + 四元数 SLERP 的混合轨迹。

        @param prev_action: (D,) 上一时刻动作
        @param cur_action: (D,) 当前目标动作
        @param end_velocity: 期望终点速度，透传给父类 PolynomialInterpolation
        @return: (N, D) 混合插值后的轨迹
        """
        steps = super().interpolate(prev_action, cur_action, end_velocity=end_velocity, num_steps=num_steps)
        n_steps = len(steps)
        if n_steps == 0:
            return steps

        prev_arr = np.asarray(prev_action, dtype=np.float64)
        cur_arr = np.asarray(cur_action, dtype=np.float64)

        for qstart, qlen in self.quat_segments:
            q0 = prev_arr[qstart: qstart + qlen]
            q1 = cur_arr[qstart: qstart + qlen]
            steps[:, qstart: qstart + qlen] = _slerp_sequence(q0, q1, n_steps)

        return steps


# ===================== 工厂函数 =====================


def build_interpolator_from_config(
    config: Dict[str, Any],
    *,
    default_class: str = "PolynomialInterpolation",
) -> Interpolator:
    """
    根据配置构建插值器实例
    
    Args:
        config: interpolation 子配置，通常来自 slave/config.yaml 中的
                `action.interpolation` 部分。
                要求至少包含键 `class`，其它键会作为 kwargs 传入构造函数。
        default_class: 当未指定 class 时使用的默认插值类名称。
        
    Returns:
        对应插值类的实例。
        
    典型用法（在 robot_client 中）::
    
        interp_cfg = cfg["action"].get("interpolation", {})
        interpolator = build_interpolator_from_config(interp_cfg)
    
    设计原则：
    - 类名字符串到具体实现的映射集中维护在本文件 `_INTERPOLATION_CLASS_REGISTRY` 中
    - config 中的所有其它键将被原样作为关键字参数传入构造函数
    """
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError(f"interpolation config must be a dict, got {type(config)}")

    class_name = config.get("class", default_class)
    if not isinstance(class_name, str):
        raise TypeError(f"interpolation.class must be a string, got {type(class_name)}")

    # 直接在当前模块命名空间中按名字查找类，避免额外维护一份 registry
    module = sys.modules[__name__]
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        raise ValueError(
            f"Unknown interpolation class '{class_name}'. "
            "Make sure the class is defined in utils.interpolation "
            "and the name in config matches exactly."
        )

    # 其余字段全部作为 kwargs 传入构造函数，
    # 但只传递目标类构造函数实际接受的参数，
    # 这样 config 中可以同时存在 action_rate 和 arm_steps_length 等不同插值类的参数而不会报错
    kwargs = {k: v for k, v in config.items() if k != "class"}

    sig = inspect.signature(cls.__init__)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if not has_var_keyword:
        valid_params = {
            name for name, p in sig.parameters.items()
            if name != "self"
        }
        ignored = set(kwargs) - valid_params
        if ignored:
            import logging as _logging
            _logging.getLogger(__name__).debug(
                "Interpolation class '%s' ignoring unused config keys: %s",
                class_name, ignored,
            )
        kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(**kwargs)
