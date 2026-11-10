"""
Executor base class.
Defines the unified interface for action execution strategies.
All execution modes (sync/async/rtc/simple_rtc) inherit from this.
"""

# Enable PEP 604 union syntax (`X | Y`) on Python 3.8+. Libero conda env
# still ships 3.8; future annotations makes all annotations lazy strings.
from __future__ import annotations

import abc
import numpy as np
from typing import Any, Dict, Optional

from utils.message import Action, Observation, WireConfig
from utils.logging_utils import setup_logger

logger = setup_logger("Executor")


def parse_plan_step_max_delta(config: Dict[str, Any]):
    """
    从配置中读取 plan_step_max_delta。

    @param config: slave 全局配置
    @return: None（禁用）、float（标量）或 np.ndarray（按关节）
    """
    action_cfg = config.get("action", {}) or {}
    delta_cfg = action_cfg.get("plan_step_max_delta", None)
    if delta_cfg is None:
        return None
    if isinstance(delta_cfg, dict):
        if not delta_cfg.get("enabled", True):
            return None
        raw = delta_cfg.get("values", None)
    else:
        raw = delta_cfg
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return np.array([float(v) for v in raw])


def clamp_plan_from_start(
    plan: np.ndarray,
    anchor: np.ndarray,
    max_delta,
) -> np.ndarray:
    """
    从 anchor 位置开始，对 plan 的每一步做累积 delta clamp。

    @param plan: (N, D) plan 步数组
    @param anchor: (D,) 当前实际位置，作为约束起点
    @param max_delta: 每步最大跳变量，标量或 (D,) 数组（单臂7维自动扩展到双臂14维）
    @return: 约束后的 plan 数组（副本）
    """
    dim = plan.shape[1] if plan.ndim > 1 else plan.shape[0]
    md = np.asarray(max_delta)
    if md.ndim == 0:
        md = np.full(dim, float(md))
    elif md.shape[0] == 7 and dim == 14:
        md = np.concatenate([md, md])

    clamped = plan.copy()
    prev = anchor.copy()
    for i in range(len(clamped)):
        delta = clamped[i] - prev
        clamped[i] = prev + np.clip(delta, -md, md)
        prev = clamped[i]
    return clamped


class ObservationBuffer:
    """Time-stacked observation image buffer owned by the executor.

    The buffer stores each view as a single numpy array:
    ``Dict[str, np.ndarray]`` where values are shaped ``[T, H, W, C]``.
    """

    def __init__(
        self,
        video_length: int = 1,
        video_action_rate: int = 1,
        image_format: str = "jpeg",
        jpeg_quality: int = 95,
    ):
        self.video_length = int(video_length)
        self.video_action_rate = max(1, int(video_action_rate))
        self.image_format = image_format
        self.jpeg_quality = int(jpeg_quality)
        self.images: Dict[str, np.ndarray] = {}
        self._action_steps = 0

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        wire_config: Optional[WireConfig] = None,
    ) -> "ObservationBuffer":
        obs_cfg = config.get("observation", {}) or {}
        wire = wire_config
        video_length = obs_cfg.get("video_length", getattr(wire, "video_length", 1))
        video_action_rate = obs_cfg.get("video_action_rate", getattr(wire, "video_action_rate", 1))
        image_format = obs_cfg.get("image_format", getattr(wire, "image_format", "jpeg"))
        jpeg_quality = obs_cfg.get("jpeg_quality", getattr(wire, "jpeg_quality", 95))
        return cls(
            video_length=video_length,
            video_action_rate=video_action_rate,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )

    def reset(self):
        self.images.clear()
        self._action_steps = 0

    def configure(self, wire_config: WireConfig) -> None:
        """Replace buffer parameters from an authoritative WireConfig.

        Unlike ``from_config``, this ignores ``observation`` dict overrides so
        a session-level handshake (e.g. CMD_START) can fully replace defaults.
        """
        self.video_length = int(wire_config.video_length)
        self.video_action_rate = max(1, int(wire_config.video_action_rate))
        self.image_format = wire_config.image_format
        self.jpeg_quality = int(wire_config.jpeg_quality)
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.video_length != 1

    def seed(self, obs: Observation):
        """Prime the buffer with the first observation of a session."""
        if self.images or not obs.images:
            return
        self._append_images(obs.images)

    def should_capture_after_action(self) -> bool:
        """Advance the waypoint counter; return True if a frame should be stored.

        Call once per consumed waypoint. When this returns False, callers can
        skip ``controller.get_observation()`` unless another hook needs it.
        """
        self._action_steps += 1
        return self._action_steps % self.video_action_rate == 0

    def record_after_action(self, obs: Observation) -> None:
        """Append ``obs.images`` after a positive ``should_capture_after_action``."""
        if obs.images:
            self._append_images(obs.images)

    def update_after_action(self, obs: Observation):
        """Record one post-action observation, respecting ``video_action_rate``.

        Prefer ``should_capture_after_action`` + ``record_after_action`` when
        the caller wants to avoid ``get_observation`` on skipped steps.
        """
        if not self.should_capture_after_action():
            return
        self.record_after_action(obs)

    def to_images(self) -> Dict[str, np.ndarray]:
        if self.video_length == -1:
            images = self._format_images(self.images)
            self.images = {}
            return images
        return self._format_images(self.images)

    def prepare_current_images(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize current camera views to time-batched wire arrays."""
        return {name: self._as_time_batch(image) for name, image in images.items()}

    def _append_images(self, images: Dict[str, np.ndarray]):
        for name, image in images.items():
            batch = self._as_time_batch(image)
            if name in self.images:
                self.images[name] = np.concatenate([self.images[name], batch], axis=0)
            else:
                self.images[name] = batch.copy()

            if self.video_length > 0 and self.images[name].shape[0] > self.video_length:
                self.images[name] = self.images[name][-self.video_length:]

    @staticmethod
    def _as_time_batch(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3:
            return arr[np.newaxis, ...]
        if arr.ndim >= 4:
            return arr
        logger.error(f"Image array has unexpected dimensions: {arr.shape}")
        return arr.reshape((1,) + arr.shape)

    @staticmethod
    def _format_images(images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        formatted: Dict[str, np.ndarray] = {}
        for name, batch in images.items():
            formatted[name] = batch
        return formatted


class BaseExecutor(abc.ABC):
    """Action executor base class."""

    def __init__(
        self,
        config: Dict[str, Any],
        controller: Any,
    ):
        """
        @param config: slave config dict
        @param controller: hardware controller instance
        """
        self.config = config
        self.controller = controller
        self._running = True
        self._paused = False
        self.prev_action: Optional[np.ndarray] = None
        self.last_commanded_action: Optional[np.ndarray] = None

        action_cfg = config.get("action", {})
        self.publish_rate = action_cfg.get("publish_rate", 200)
        interp_cfg = action_cfg.get("interpolation", {}) or {}
        self.use_interpolation = interp_cfg.get("enabled", True)

        self.plan_step_max_delta = parse_plan_step_max_delta(config)

    @abc.abstractmethod
    def execute(self, action: Action):
        """
        Execute an Action.

        - SyncExecutor: blocks until all steps are published.
        - RTC executors: injects the new chunk and returns quickly.
        """

    def start(self):
        """
        Called once when a session starts (after reset).
        RTC executors start their background action loop here.
        """

    def wait_ready_for_inference(self) -> None:
        """
        Block (in a worker thread) until the executor is ready for the
        next inference cycle.

        - SyncExecutor: returns after the previous execute() completes.
        - SimpleRtcExecutor: returns when execution reaches s_min.
        """

    def before_get_observation(self) -> None:
        """
        Hook called immediately before the agent reads a fresh controller
        observation.

        Default: no-op. Executors may override this to intentionally delay
        capture relative to their readiness signal.
        """

    def prepare_obs(self, obs: Observation) -> Observation:
        """
        Enrich an observation with executor-specific metadata before
        it is sent to the policy server.

        - SyncExecutor: pass-through.
        - RTC executors: attach action_constraint, rtc_s, rtc_d, etc.

        @param obs: raw observation from the controller
        @return: enriched observation (may be mutated in-place)
        """
        return obs

    def apply_wire_config(self, wire_config: WireConfig) -> None:
        """Apply policy wire encoding to the observation buffer.

        Default: no-op. Executors that own an ``ObservationBuffer`` override
        this to reconfigure frame stacking / image encoding for a session.
        """

    def on_inference_started(self):
        """
        Called right after the observation is sent and inference is in-flight.
        Default: no-op. Override if the executor needs to react.
        """

    def pause(self):
        """Pause execution."""
        self._paused = True
        logger.info("Executor paused")

    def resume(self):
        """Resume execution."""
        self._paused = False
        logger.info("Executor resumed")

    def stop(self):
        """Stop execution."""
        self._running = False
        logger.info("Executor stopped")

    def is_publish_thread_active(self) -> bool:
        """Whether this executor currently owns a live publishing worker."""
        return False

    def wait_until_stopped(self, timeout: float = 1.0) -> bool:
        """Wait for publishing workers to exit; workerless executors are stopped."""
        return not self.is_publish_thread_active()

    def wait_until_idle(self, timeout: float) -> bool:
        """Wait until all submitted actions are consumed.

        Blocking executors have no queued work after ``execute`` returns.
        """
        return True

    def _init_prev_action_from_robot(self):
        """Read current robot state and set as prev_action.

        obs.state is a numpy array (already selected by action_type in hardware layer).
        Called at the start of execution when prev_action is None, so the
        first action chunk goes through interpolation instead of jumping
        directly to the target position (which can be dangerously fast).
        """
        if self.prev_action is not None:
            return
        try:
            obs = self.controller.get_observation()
            if obs and obs.state is not None:
                self.prev_action = np.asarray(obs.state, dtype=np.float64).copy()
                logger.info(f"prev_action initialized from robot state (dim={len(self.prev_action)})")
                return
        except Exception as e:
            logger.warning(f"Failed to read robot state: {e}")

    def reset(self):
        """Reset state for a new session."""
        self.prev_action = None
        self._paused = False
        self._running = True
        logger.info("Executor reset")
