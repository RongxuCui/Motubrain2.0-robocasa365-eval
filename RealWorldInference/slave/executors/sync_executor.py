"""
SyncExecutor - synchronous action execution.

Mirrors the logic: interpolate -> filter -> publish.
Blocks until the entire action chunk is executed before returning.

The executor is action-space agnostic: it forwards a flat numpy vector to
``controller.publish_action`` and reads state via ``controller.get_observation``.
The controller routes by ``action_type`` internally.
"""

import threading
import time
import numpy as np
from typing import Any, Callable, Dict, Optional

from slave.executors.base import BaseExecutor, ObservationBuffer, clamp_plan_from_start
from utils.message import Action, Observation, WireConfig
from utils.logging_utils import setup_logger
import utils.interpolation as interpolation
from utils.filter import LowPassFilter

logger = setup_logger("SyncExecutor")


class SyncExecutor(BaseExecutor):
    """Synchronous (blocking) action executor."""

    def __init__(
        self,
        config: Dict[str, Any],
        controller: Any,
        wire_config: Optional[WireConfig] = None,
    ):
        super().__init__(config, controller)
        self.observation_buffer = ObservationBuffer.from_config(config, wire_config)
        self.on_action_step: Optional[Callable[[Observation], None]] = None

        # Action layout comes from the robot profile via controller.layout.
        controller_cfg = config.get("controller", {})
        self.action_type = controller_cfg["action_type"]

        layout_cfg = controller_cfg["layout"]
        self.action_dim = int(layout_cfg["dim"])
        # asdict serializes tuples as lists; restore them for the interpolator.
        quat_segments = layout_cfg.get("quat_segments")
        self.quat_segments = (
            [tuple(seg) for seg in quat_segments] if quat_segments else None
        )
        self.no_filter_indices = list(layout_cfg.get("no_filter_indices") or [])
        xyz_segments = layout_cfg.get("xyz_segments")
        self.xyz_segments = (
            [tuple(seg) for seg in xyz_segments] if xyz_segments else []
        )
        self.jump_threshold = float(layout_cfg["jump_threshold"])

        action_cfg = config.get("action", {})
        obs_cfg = config.get("observation", {})
        obs_rate = obs_cfg.get("publish_rate", 30)

        # Interpolation setup
        interp_cfg = dict(action_cfg.get("interpolation", {}) or {})
        interp_cfg.setdefault("publish_rate", self.publish_rate)
        interp_cfg.setdefault("action_rate", obs_rate)
        if self.quat_segments is not None:
            interp_cfg.setdefault("quat_segments", self.quat_segments)
        self.interpolator = interpolation.build_interpolator_from_config(interp_cfg)

        # Filter setup
        filter_cfg = config.get("filter", {}) or {}
        self.filter_enabled = filter_cfg.get("enabled", True)
        cutoff = filter_cfg.get("cutoff_freq_ratio", 0.03) * self.publish_rate
        ignore_indices = filter_cfg.get("ignore_indices", self.no_filter_indices)
        self.filter = LowPassFilter(
            cutoff_freq=cutoff,
            sampling_rate=self.publish_rate,
            ignore_indices=ignore_indices,
        )

        self._inference_ready = threading.Event()
        self._inference_ready.set()

        logger.info(
            f"SyncExecutor init: action_type={self.action_type} "
            f"dim={self.action_dim} rate={self.publish_rate}Hz interp={self.use_interpolation}"
        )

    # ---- scheduling interface ----

    def wait_ready_for_inference(self) -> None:
        self._inference_ready.wait()
        self._inference_ready.clear()

    def resume(self):
        """Resume execution.

        除了清掉 _paused，还要 set _inference_ready：pause 期间 wait_ready
        可能正卡在 wait()，clear 之后没人 set 会一直阻塞。直接放行让推理
        循环重新发起 obs 即可，不会有数据不一致——execute() 自己做了 jump
        检测，会从机器人当前位置重新对齐。
        """
        super().resume()
        self._inference_ready.set()

    def prepare_obs(self, obs: Observation) -> Observation:
        """Apply sync-executor observation buffering and wire encoding."""
        if self.observation_buffer.enabled:
            self.observation_buffer.seed(obs)
            obs.images = self.observation_buffer.to_images()
        else:
            obs.images = self.observation_buffer.prepare_current_images(obs.images)
        obs.image_format = self.observation_buffer.image_format
        obs.jpeg_quality = self.observation_buffer.jpeg_quality
        return obs

    def apply_wire_config(self, wire_config: WireConfig) -> None:
        self.observation_buffer.configure(wire_config)
        logger.info("SyncExecutor wire_config applied: %s", wire_config.to_dict())

    # ---- controller adapters ----

    def _publish_action(self, action: np.ndarray):
        self.controller.publish_action(Action(action=action))

    def _get_current_state(self) -> Optional[np.ndarray]:
        try:
            obs = self.controller.get_observation()
            if obs is None or obs.state is None:
                return None
            return np.asarray(obs.state, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Failed to read robot state: {e}")
            return None

    def _check_position_jump(self):
        """
        Detect jump between current robot state and prev_action.
        If jump exceeds threshold, resets filter and interpolator.

        EEF mode uses xyz_segments derived from the action layout, so the
        check works for any robot whose EEF triplets sit at known offsets.
        Qpos mode uses max absolute joint diff.
        """
        if self.prev_action is None:
            return

        current_state = self._get_current_state()
        if current_state is None:
            return

        if self.action_type == "eef" and self.xyz_segments:
            dists = [
                float(np.linalg.norm(current_state[s:s + l] - self.prev_action[s:s + l]))
                for s, l in self.xyz_segments
            ]
            max_dist = max(dists) if dists else 0.0

            if max_dist > self.jump_threshold:
                logger.warning(
                    f"EEF position jump detected ({max_dist:.4f} m), "
                    f"resetting filter & interpolator"
                )
                self.prev_action = current_state.copy()
                self.filter.reset()
                if hasattr(self.interpolator, "reset"):
                    self.interpolator.reset()
        else:
            max_diff = float(np.max(np.abs(current_state - self.prev_action)))

            if max_diff > self.jump_threshold:
                logger.warning(
                    f"Qpos jump detected ({max_diff:.3f} rad), "
                    f"resetting filter & interpolator"
                )
                self.prev_action = current_state.copy()
                self.filter.reset()
                if hasattr(self.interpolator, "reset"):
                    self.interpolator.reset()

    def _init_prev_action_from_robot(self):
        """Read current robot state as interpolation anchor."""
        if self.prev_action is not None:
            return

        state = self._get_current_state()
        if state is not None:
            self.prev_action = state.copy()
            logger.info(
                f"prev_action initialized from robot state (dim={len(self.prev_action)})"
            )

    # ---- execution ----

    def execute(self, action: Action):
        """
        Execute action chunk synchronously.
        Blocks until all steps are published to hardware, then signals
        ``_inference_ready`` so the Agent can send the next observation.
        """
        try:
            recv_ts = time.time()
            action_ts = getattr(action, "timestamp", None)
            if action_ts is not None:
                logger.info(
                    f"[action-ts] SyncExecutor received action frame_ts={action_ts:.6f} "
                    f"recv_ts={recv_ts:.6f} latency={recv_ts - action_ts:.6f}s"
                )
            else:
                logger.info(f"[action-ts] SyncExecutor received action frame_ts=None recv_ts={recv_ts:.6f}")

            chunks = action.action
            if chunks.ndim == 1:
                chunks = chunks[np.newaxis, :]

            self._check_position_jump()
            self._init_prev_action_from_robot()

            if self.plan_step_max_delta is not None:
                anchor = self.prev_action if self.prev_action is not None else chunks[0]
                chunks = clamp_plan_from_start(chunks, anchor, self.plan_step_max_delta)

            sleep_time = 1.0 / self.publish_rate
            executed = 0

            for chunk in chunks:
                if not self._running:
                    break
                while self._paused and self._running:
                    time.sleep(0.05)

                if self.use_interpolation and self.prev_action is not None:
                    steps = self.interpolator.interpolate(self.prev_action, chunk)
                else:
                    steps = chunk[np.newaxis, :] if chunk.ndim == 1 else chunk

                for step in steps:
                    if not self._running:
                        break
                    while self._paused and self._running:
                        time.sleep(0.05)
                    filtered = self.filter.update(step) if self.filter_enabled else step
                    self.controller.publish_action(Action(action=filtered, timestamp=action_ts))
                    time.sleep(sleep_time)

                self._on_action_step()
                self.prev_action = chunk.copy()
                executed += 1

            logger.info(f"SyncExecutor: executed {executed}/{len(chunks)} chunks")
        finally:
            self._inference_ready.set()

    def reset(self):
        super().reset()
        self.observation_buffer.reset()
        self.filter.reset()
        if hasattr(self.interpolator, "reset"):
            self.interpolator.reset()
        self._inference_ready.set()

    def _on_action_step(self):
        """Capture one post-action observation for buffering and observers.

        ``video_action_rate`` gates camera reads for the observation buffer:
        skipped steps do not call ``get_observation`` unless ``on_action_step``
        still needs a frame.
        """
        capture_for_buffer = (
            self.observation_buffer.enabled
            and self.observation_buffer.should_capture_after_action()
        )
        need_hook = self.on_action_step is not None
        if not capture_for_buffer and not need_hook:
            return
        try:
            obs = self.controller.get_observation()
            if obs is None:
                return
            if capture_for_buffer:
                self.observation_buffer.record_after_action(obs)
            if need_hook:
                self.on_action_step(obs)
        except Exception as e:
            logger.warning(f"Action step observation hook failed: {e}")
