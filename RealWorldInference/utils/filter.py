#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Filtering Utilities
信号滤波工具类
"""

import numpy as np
import scipy.signal as signal
from typing import Optional, Union, List, Tuple

from utils.interpolation import slerp

class LowPassFilter:
    """
    巴特沃斯低通滤波器类 (Butterworth Low-Pass Filter)
    
    用于对动作序列或实时信号进行低通滤波，去除高频噪声，使动作更平滑。
    支持离线序列滤波（零相位）和实时单步滤波。
    """
    
    def __init__(self, cutoff_freq: float, sampling_rate: float, order: int = 2, ignore_indices: Optional[List[int]] = None):
        """
        初始化低通滤波器
        
        Args:
            cutoff_freq: 截止频率 (Hz)，建议设置为采样率的 1/10 到 1/5
            sampling_rate: 采样率 (Hz)，即数据发布的频率
            order: 滤波器阶数，默认为 2，阶数越高滚降越陡峭但延迟越大
            ignore_indices: 需要忽略的维度索引列表（不进行滤波），例如 [6, 13]
        """
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        self.ignore_indices = ignore_indices
        
        # 计算奈奎斯特频率
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        
        # 边界检查
        if normal_cutoff >= 1:
            normal_cutoff = 0.99
        
        # 设计 Butterworth 滤波器，使用 SOS (Second-Order Sections) 格式以获得更好的数值稳定性
        self.sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        
        # 实时滤波的状态缓存
        self._zi: Optional[np.ndarray] = None
        
    def filter_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        对完整的动作轨迹进行离线滤波 (零相位)
        
        使用 filtfilt 进行双向滤波，不会产生相位延迟，适合处理已录制好的轨迹。
        
        Args:
            trajectory: 动作轨迹数组，形状为 (N, D) 或 (N,)
                       其中 N 是时间步数，D 是动作维度 (如 7 或 14)
            
        Returns:
            滤波后的轨迹数组，形状与输入相同
        """
        trajectory = np.array(trajectory)
        
        # 处理空数组
        if trajectory.size == 0:
            return trajectory
            
        # 处理一维数组的情况
        if trajectory.ndim == 1:
            return signal.sosfiltfilt(self.sos, trajectory)
            
        # 对每一列（每个维度）分别进行滤波
        # axis=0 表示沿着时间轴滤波
        filtered = signal.sosfiltfilt(self.sos, trajectory, axis=0)
        
        # 如果有忽略的维度，恢复原始值
        if self.ignore_indices is not None and trajectory.ndim > 1:
            # 确保索引有效
            valid_indices = [i for i in self.ignore_indices if i < trajectory.shape[1]]
            if valid_indices:
                filtered[:, valid_indices] = trajectory[:, valid_indices]
                
        return filtered

    def update(self, current_value: np.ndarray) -> np.ndarray:
        """
        实时单步滤波
        
        用于实时控制循环中，输入当前的观测值或指令，输出平滑后的值。
        注意：实时滤波必然会引入一定的相位延迟。
        
        Args:
            current_value: 当前时刻的动作值，形状为 (D,) 或 标量
            
        Returns:
            滤波后的当前值，形状与输入相同
        """
        current_value = np.array(current_value)
        
        # 确定需要滤波的维度
        filter_mask = None
        if self.ignore_indices is not None and current_value.ndim > 0:
            filter_mask = np.ones(current_value.shape[0], dtype=bool)
            valid_indices = [i for i in self.ignore_indices if i < current_value.shape[0]]
            filter_mask[valid_indices] = False
            
            # 如果没有需要滤波的维度，直接返回
            if not np.any(filter_mask):
                return current_value
                
            # 提取需要滤波的部分
            value_to_filter = current_value[filter_mask]
        else:
            value_to_filter = current_value
        
        # 初始化状态
        if self._zi is None:
            # 计算初始状态，使得滤波器从当前的稳态开始，避免启动时的瞬态跳变
            zi_template = signal.sosfilt_zi(self.sos) # shape (n_sections, 2)
            
            if value_to_filter.ndim == 0:
                self._zi = zi_template * value_to_filter
            else:
                # 扩展 zi 的形状以匹配输入维度
                # zi shape: (n_sections, 2, D_filtered)
                self._zi = zi_template[:, :, np.newaxis] * value_to_filter[np.newaxis, np.newaxis, :]
        
        # 进行滤波
        if value_to_filter.ndim == 0:
            filtered_val, self._zi = signal.sosfilt(self.sos, [value_to_filter], zi=self._zi)
            filtered_val = filtered_val[0]
        else:
            # sosfilt 期望输入是 (n_samples, n_channels)
            # 这里我们处理单步，所以输入 reshape 为 (1, D_filtered)
            filtered_val, self._zi = signal.sosfilt(self.sos, value_to_filter[np.newaxis, :], zi=self._zi, axis=0)
            filtered_val = filtered_val[0]
            
        # 如果有忽略的维度，合并结果
        if filter_mask is not None:
            result = current_value.copy()
            result[filter_mask] = filtered_val
            return result
        else:
            return filtered_val

            
    def reset(self):
        """
        重置实时滤波器的状态
        
        在开始新的轨迹或任务时调用。
        """
        self._zi = None

class EMAFilter:
    """
    指数移动平均滤波器 (Exponential Moving Average Filter)
    
    用于对动作序列或实时信号进行平滑处理。
    相比巴特沃斯滤波器，EMA计算量更小，但频响特性不如巴特沃斯陡峭。
    """
    
    def __init__(self, cutoff_freq: float, sampling_rate: float, alpha: Optional[float] = None, ignore_indices: Optional[List[int]] = None):
        """
        初始化 EMA 滤波器
        
        @param cutoff_freq: 截止频率 (Hz)，用于计算平滑系数 alpha
        @param sampling_rate: 采样率 (Hz)
        @param alpha: 可选，直接指定平滑系数 (0 < alpha <= 1)。如果指定，忽略 cutoff_freq。
                      alpha 越小，平滑程度越高，延迟越大。
        @param ignore_indices: 需要忽略的维度索引列表（不进行滤波）
        """
        self.sampling_rate = sampling_rate
        self.ignore_indices = ignore_indices
        
        if alpha is not None:
            self.alpha = alpha
        else:
            # 根据截止频率计算 alpha
            # alpha = 2 * pi * fc / (fs + 2 * pi * fc)
            omega_c = 2 * np.pi * cutoff_freq
            self.alpha = omega_c / (sampling_rate + omega_c)
            
        # 边界检查
        self.alpha = np.clip(self.alpha, 0.0, 1.0)
        
        # IIR coefficients for filtfilt (b, a)
        # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
        self.b = np.array([self.alpha])
        self.a = np.array([1, -(1 - self.alpha)])
        
        # 实时滤波的状态缓存
        self._last_value: Optional[np.ndarray] = None
        
    def filter_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        对完整的动作轨迹进行离线滤波 (零相位)
        
        @param trajectory: 动作轨迹数组，形状为 (N, D) 或 (N,)
        @return: 滤波后的轨迹数组
        """
        trajectory = np.array(trajectory)
        
        if trajectory.size == 0:
            return trajectory
            
        # 处理极短序列，避免 filtfilt 报错
        # filtfilt 需要数据长度大于 padlen (默认 3 * max(len(a), len(b)) = 6)
        if trajectory.shape[0] <= 6:
            return trajectory
            
        try:
            # 确定需要滤波的维度
            if self.ignore_indices is not None and trajectory.ndim > 1:
                # 创建副本以避免修改输入
                result = trajectory.copy()
                
                # 找出需要滤波的列
                filter_indices = [i for i in range(trajectory.shape[1]) if i not in self.ignore_indices]
                
                if not filter_indices:
                    return result
                    
                # 只对选定的列进行滤波
                filtered_cols = signal.filtfilt(self.b, self.a, trajectory[:, filter_indices], axis=0)
                result[:, filter_indices] = filtered_cols
                return result
            else:
                # filtfilt 实现零相位滤波 (前向+后向)
                if trajectory.ndim == 1:
                    return signal.filtfilt(self.b, self.a, trajectory)
                
                # axis=0 表示沿着时间轴滤波
                return signal.filtfilt(self.b, self.a, trajectory, axis=0)
        except Exception:
            return trajectory

    def update(self, current_value: np.ndarray) -> np.ndarray:
        """
        实时单步滤波
        
        @param current_value: 当前时刻的动作值
        @return: 滤波后的当前值
        """
        current_value = np.array(current_value)
        
        if self._last_value is None:
            self._last_value = current_value
            return current_value
            
        # EMA 更新公式
        # self._last_value = self.alpha * current_value + (1 - self.alpha) * self._last_value
        
        if self.ignore_indices is not None and current_value.ndim > 0:
            # 创建掩码
            update_mask = np.ones(current_value.shape[0], dtype=bool)
            valid_indices = [i for i in self.ignore_indices if i < current_value.shape[0]]
            update_mask[valid_indices] = False
            
            # 只更新需要滤波的部分
            # new_val = alpha * curr + (1-alpha) * last
            # => new_val = last + alpha * (curr - last)
            
            # 计算增量
            delta = current_value - self._last_value
            
            # 只在非忽略的维度上应用 EMA
            self._last_value[update_mask] += self.alpha * delta[update_mask]
            
            # 对于忽略的维度，直接更新为当前值
            self._last_value[~update_mask] = current_value[~update_mask]
        else:
            self._last_value = self.alpha * current_value + (1 - self.alpha) * self._last_value
            
        return self._last_value
        
    def reset(self):
        """
        重置滤波器状态
        """
        self._last_value = None


class QuaternionAwareEMAFilter:
    """
    Quaternion-aware EMA for single-step action filtering.

    - scalar dimensions (xyz / qpos): standard EMA
    - quaternion segments: SLERP(prev, current, alpha)
    - ignore_indices: passthrough dimensions, typically grippers
    """

    def __init__(
        self,
        cutoff_freq: float = 0.0,
        sampling_rate: float = 1.0,
        alpha: Optional[float] = 0.2,
        quat_segments: Optional[List[Tuple[int, int]]] = None,
        ignore_indices: Optional[List[int]] = None,
    ):
        self.sampling_rate = sampling_rate
        self.quat_segments = [tuple(seg) for seg in (quat_segments or [])]
        self.ignore_indices = ignore_indices or []

        if alpha is None:
            omega_c = 2 * np.pi * cutoff_freq
            self.alpha = omega_c / (sampling_rate + omega_c)
        else:
            self.alpha = float(alpha)
        self.alpha = float(np.clip(self.alpha, 0.0, 1.0))

        self._last_value: Optional[np.ndarray] = None

    def update(self, current_value: np.ndarray) -> np.ndarray:
        current = np.asarray(current_value, dtype=np.float64)

        if self._last_value is None or self._last_value.shape != current.shape:
            self._last_value = current.copy()
            return current.copy()

        if current.ndim == 0:
            self._last_value = self.alpha * current + (1.0 - self.alpha) * self._last_value
            return self._last_value.copy()

        result = self._last_value + self.alpha * (current - self._last_value)

        if current.ndim == 1:
            for start, length in self.quat_segments:
                end = start + length
                if length == 4 and 0 <= start and end <= current.shape[0]:
                    result[start:end] = slerp(self._last_value[start:end], current[start:end], self.alpha)

            valid_ignore = [i for i in self.ignore_indices if 0 <= i < current.shape[0]]
            if valid_ignore:
                result[valid_ignore] = current[valid_ignore]

        self._last_value = result.copy()
        return result

    def reset(self):
        self._last_value = None
