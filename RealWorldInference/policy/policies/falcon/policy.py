"""Bridge RealWorldInference's policy interface to Soma inference.

All model logic (image preprocessing, multiview handling, diffusion, RTC, etc.)
lives in Soma. This adapter only converts between Observation/Action and the
Soma dictionary interface.

Config example (config_falcon.yaml):
    policy:
      type: "falcon"
      config_path: "/path/to/ss-soma/configs/inference/policy.py"
      checkpoint_path: "/path/to/step-50000-ema.safetensors"

连接参数（url / insecure）统一在 configs/server.yaml policy: 段。
"""

from collections.abc import Mapping
import logging
import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np

_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from policy.policies.base_policy import BasePolicy, Observation, Action, WireConfig
from policy.policies.falcon.astribot_camera import (
    AstribotHeadCameraExtrinsicProvider,
)
from utils.episode_recorder import compose_standard_views
from utils.image import save_video

logger = logging.getLogger(__name__)

_INTERNAL_OBS_KWARGS = {
    "capture_timestamp",
    "obs_send_timestamp",
    "obs_step",
    "prepare_timestamp",
    "timestamp_semantics",
    "plan_timestamp_source",
}


class FalconPolicy(BasePolicy):
    """Convert RealWorldInference observations and actions for Soma."""

    # Stash the loaded cfg so get_wire_config() can read from it.
    _cfg = None
    _inference_seed: int | None = None

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        policy_id: str = "policy_falcon",
        device: str = "cuda",
        return_video: bool = False,
        inference_seed: int | None = None,
        **kwargs,
    ):
        # Ensure Soma loggers (and all other libraries) have a handler.
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        config_path = str(Path(config_path).resolve())

        from soma.inference import build_policy
        from soma.inference.config import EmbodiedConfig

        logger.info(f"Loading config: {config_path}")
        cfg = EmbodiedConfig.load(config_path, checkpoint=checkpoint_path)
        cfg.load_model_config()
        if return_video and not cfg.inference.decode_video:
            logger.info("return_video enabled; forcing cfg.inference.decode_video=True")
            cfg.inference.decode_video = True
        self._cfg = cfg
        self._policy_id = policy_id
        self._pipeline_server_handle = None
        self._camera_key = getattr(cfg.policy.action, "camera_key", None)
        self._camera_extrinsic_provider = None
        camera_provider = os.environ.get("RWI_CAMERA_EXTRINSIC_PROVIDER")
        if camera_provider:
            if not self._camera_key:
                raise ValueError(
                    "RWI_CAMERA_EXTRINSIC_PROVIDER requires a configured "
                    "policy action camera_key"
                )
            if camera_provider != "astribot_s1_joint_state":
                raise ValueError(
                    "Unknown RWI_CAMERA_EXTRINSIC_PROVIDER: "
                    f"{camera_provider!r}"
                )
            calibration_path = os.environ.get(
                "RWI_ASTRIBOT_HEAD_CAMERA_CALIB_PATH"
            )
            if not calibration_path:
                raise ValueError(
                    "RWI_ASTRIBOT_HEAD_CAMERA_CALIB_PATH is required for "
                    "astribot_s1_joint_state camera extrinsics"
                )
            self._camera_extrinsic_provider = (
                AstribotHeadCameraExtrinsicProvider(calibration_path)
            )
            logger.info(
                "Camera extrinsic provider: %s -> %s",
                camera_provider,
                self._camera_key,
            )

        pipeline_server_cfg = cfg.inference.pipeline_server
        if (
            pipeline_server_cfg.enabled
            and pipeline_server_cfg.launch is not None
        ):
            from soma.inference.pipeline_server.launcher import (
                ensure_pipeline_server,
            )

            logger.info(
                "Ensuring managed pipeline server is ready at %s",
                pipeline_server_cfg.endpoint,
            )
            self._pipeline_server_handle = ensure_pipeline_server(
                cfg,
                config_path,
            )
            logger.info(
                "Pipeline server ready deployment_id=%s pid=%d "
                "started_by_this_policy=%s",
                self._pipeline_server_handle.deployment_id,
                self._pipeline_server_handle.pid,
                self._pipeline_server_handle.started_by_caller,
            )

        logger.info(f"Building policy (checkpoint: {checkpoint_path})...")
        self._policy = build_policy(cfg, device=device)
        logger.info("Policy ready.")

        # State
        self._cached_images: Optional[dict] = None
        self._cached_instruction: Optional[str] = None
        self._cached_state: Optional[dict] = None
        self._cached_obs_timestamp: Optional[float] = None
        self._extra_kwargs: dict = {}
        self._return_video = return_video
        self._inference_seed = inference_seed

    def set_session_id(self, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")
        inference_id = f"{self._policy_id}:{session_id}"
        if hasattr(self._policy, "set_inference_id"):
            self._policy.set_inference_id(inference_id)

    def close(self) -> None:
        if hasattr(self._policy, "close"):
            try:
                self._policy.close()
            except Exception as exc:
                logger.warning("Failed to close Falcon policy cleanly: %s", exc)

    def reset(self):
        self._cached_images = None
        self._cached_instruction = None
        self._cached_state = None
        self._cached_obs_timestamp = None
        self._extra_kwargs = {}
        if hasattr(self._policy, 'reset'):
            self._policy.reset()
        logger.info("Policy reset")

    def update_obs(self, observation: Observation):
        recv_ts = time.time()
        plan_ts = getattr(observation, "timestamp", None)
        capture_ts = (observation.kwargs or {}).get("capture_timestamp")
        if plan_ts is not None:
            transport_latency = None if capture_ts is None else recv_ts - capture_ts
            logger.info(
                f"[obs-ts] FalconPolicy received obs plan_ts={plan_ts:.6f} "
                f"capture_ts={capture_ts} recv_ts={recv_ts:.6f} "
                f"plan_lag={recv_ts - plan_ts:.6f}s transport_latency={transport_latency}"
            )
        else:
            logger.info(
                f"[obs-ts] FalconPolicy received obs plan_ts=None "
                f"capture_ts={capture_ts} recv_ts={recv_ts:.6f}"
            )
        self._cached_obs_timestamp = plan_ts

        self._cached_images = {
            name: np.asarray(img, dtype=np.uint8)
            for name, img in observation.images.items()
        }
        if observation.instruction:
            self._cached_instruction = observation.instruction

        # Cache state for action postprocessor (eef, qpos, etc.)
        self._cached_state = observation.state

        self._extra_kwargs = dict(observation.kwargs) if observation.kwargs else {}
        if (
            self._camera_extrinsic_provider is not None
            and self._camera_key not in self._extra_kwargs
        ):
            if observation.joint_state is None:
                raise KeyError(
                    "Astribot camera extrinsic provider requires observation "
                    "joint_state"
                )
            self._extra_kwargs[self._camera_key] = (
                self._camera_extrinsic_provider(observation.joint_state)
            )
        if getattr(observation, 'action_constraint', None) is not None:
            self._extra_kwargs["action_constraint"] = observation.action_constraint

        # Pass state through for postprocessor
        if self._cached_state is not None:
            self._extra_kwargs["state"] = self._cached_state

    def _prepare_prediction_video(self, video) -> np.ndarray:
        """Convert decoded single- or multiview output to one debug video."""
        if isinstance(video, (list, tuple)):
            view_names = list(getattr(self._policy, "view_names", ()) or ())
            if len(view_names) != len(video):
                raise ValueError(
                    "Decoded video view count does not match policy views: "
                    f"video={len(video)}, view_names={view_names}"
                )
            views = [np.asarray(view, dtype=np.uint8) for view in video]
            if any(view.ndim != 4 or view.shape[-1] != 3 for view in views):
                raise ValueError(
                    "Each decoded video view must have shape [T,H,W,3], got "
                    f"{[view.shape for view in views]}"
                )
            frame_counts = {view.shape[0] for view in views}
            if len(frame_counts) != 1:
                raise ValueError(
                    "Decoded video views must have the same frame count, got "
                    f"{[view.shape for view in views]}"
                )
            if len(views) == 1:
                return views[0]
            standard_roles = (
                ("head_camera", "cam_high", "head"),
                ("left_camera", "left_wrist", "cam_left_wrist"),
                ("right_camera", "right_wrist", "cam_right_wrist"),
            )
            if all(set(aliases).intersection(view_names) for aliases in standard_roles):
                return self._stitch_multiview_video(views, view_names)
            frames = [
                compose_standard_views({
                    name: view[frame_index]
                    for name, view in zip(view_names, views)
                })
                for frame_index in range(views[0].shape[0])
            ]
            return np.stack(frames, axis=0)

        video_array = np.asarray(video, dtype=np.uint8)
        if video_array.ndim != 4 or video_array.shape[-1] != 3:
            raise ValueError(
                "Decoded video must have shape [T,H,W,3], got "
                f"{video_array.shape}"
            )
        return video_array

    def _save_prediction_video(self, video, out_path: str, fps: int = 8):
        """Save a model prediction as a single-view or composed multiview MP4."""
        video_array = self._prepare_prediction_video(video)
        save_video(out_path, list(video_array), fps=fps)
        frame_count, height, width, _ = video_array.shape
        logger.info(
            "Prediction video saved: %s (%d frames, %dx%d)",
            out_path,
            frame_count,
            width,
            height,
        )

    @staticmethod
    def _squeeze_single_frame(arr: np.ndarray) -> np.ndarray:
        """Keep existing single-frame policies compatible with wire [1,H,W,3]."""
        return arr[0] if arr.ndim == 4 and arr.shape[0] == 1 else arr

    @classmethod
    def _squeeze_single_frame_images(cls, images: dict) -> dict:
        return {name: cls._squeeze_single_frame(img) for name, img in images.items()}

    def _dump_debug_inference(
        self,
        *,
        policy_images: dict,
        video_views: list[np.ndarray] | None,
        model_context: dict,
        result: dict,
    ) -> None:
        output_dir = os.environ.get("RWI_DEBUG_DUMP_DIR")
        if not output_dir:
            return
        payload = {
            "instruction": np.asarray(self._cached_instruction),
            "raw_action": np.asarray(result.get("raw_action")),
            "action": np.asarray(result["action"]),
        }
        state = self._cached_state
        if isinstance(state, Mapping):
            for name, value in state.items():
                payload[f"state_{name}"] = np.asarray(value)
        elif state is not None:
            payload["state"] = np.asarray(state)
        if self._camera_key and self._camera_key in model_context:
            payload[self._camera_key] = np.asarray(
                model_context[self._camera_key]
            )
        for name, frames in policy_images.items():
            payload[f"wire_{name}"] = np.asarray(frames, dtype=np.uint8)
        if video_views is not None:
            for index, frames in enumerate(video_views):
                payload[f"composed_view_{index}"] = np.asarray(
                    frames,
                    dtype=np.uint8,
                )
        observation_wrapper = getattr(
            self._policy,
            "observation_wrapper",
            None,
        )
        if observation_wrapper is not None:
            prepared = observation_wrapper.prepare(
                policy_images,
                video_views=video_views,
            )
            for index, frames in enumerate(prepared):
                payload[f"prepared_view_{index}"] = np.asarray(
                    frames,
                    dtype=np.uint8,
                )

        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"inference_{time.time_ns()}.npz"
        np.savez_compressed(path, **payload)
        logger.info(
            "Inference debug dump saved: %s raw_action=%s action=%s",
            path,
            payload["raw_action"].shape,
            payload["action"].shape,
        )

    @staticmethod
    def _stitch_multiview_video(
        videos: list,
        view_names: list[str],
    ) -> np.ndarray:
        """Stitch head/wrists, with an optional torso row when present."""
        import cv2

        aliases = {
            "left": ("left_camera", "left_wrist", "cam_left_wrist"),
            "head": ("head_camera", "cam_high", "head"),
            "torso": ("torso_camera", "cam_torso", "torso"),
            "right": ("right_camera", "right_wrist", "cam_right_wrist"),
        }
        indexed = dict(zip(view_names, videos))

        def pick(role: str, required: bool = True) -> np.ndarray | None:
            for name in aliases[role]:
                if name in indexed:
                    value = np.asarray(indexed[name], dtype=np.uint8)
                    if value.ndim != 4 or value.shape[-1] != 3:
                        raise ValueError(
                            f"Generated {role} video must have shape [T,H,W,3], "
                            f"got {value.shape}"
                        )
                    return value
            if required:
                raise ValueError(
                    f"Generated multiview video is missing {role} view; "
                    f"view_names={view_names}"
                )
            return None

        left = pick("left")
        head = pick("head")
        torso = pick("torso", required=False)
        right = pick("right")
        frame_counts = {len(left), len(head), len(right)}
        if torso is not None:
            frame_counts.add(len(torso))
        if len(frame_counts) != 1:
            raise ValueError(
                "Generated multiview videos have different frame counts: "
                f"left={len(left)}, head={len(head)}, "
                f"torso={None if torso is None else len(torso)}, "
                f"right={len(right)}"
            )

        stitched_frames = []
        for frame_idx in range(len(head)):
            left_frame = left[frame_idx]
            head_frame = head[frame_idx]
            right_frame = right[frame_idx]
            head_h, head_w = head_frame.shape[:2]
            wrist_size = (head_w // 2, head_h // 2)
            left_resized = cv2.resize(
                left_frame, wrist_size, interpolation=cv2.INTER_LINEAR
            )
            right_resized = cv2.resize(
                right_frame, wrist_size, interpolation=cv2.INTER_LINEAR
            )
            wrist_row = np.concatenate([left_resized, right_resized], axis=1)
            bottom = np.concatenate([head_frame, wrist_row], axis=0)

            if torso is None:
                stitched_frames.append(bottom)
                continue

            torso_frame = torso[frame_idx]
            target_w = max(torso_frame.shape[1], bottom.shape[1])
            if torso_frame.shape[1] != target_w:
                new_h = max(
                    1,
                    int(round(torso_frame.shape[0] * target_w / torso_frame.shape[1])),
                )
                torso_frame = cv2.resize(
                    torso_frame,
                    (target_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            if bottom.shape[1] != target_w:
                new_h = max(
                    1,
                    int(round(bottom.shape[0] * target_w / bottom.shape[1])),
                )
                bottom = cv2.resize(
                    bottom,
                    (target_w, new_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            stitched_frames.append(np.concatenate([torso_frame, bottom], axis=0))

        return np.stack(stitched_frames, axis=0)

    def get_action(self) -> Action:
        if self._cached_images is None:
            raise RuntimeError("No observation. Call update_obs() first.")
        if self._cached_instruction is None:
            raise RuntimeError("No instruction in Observation.")

        extra = dict(self._extra_kwargs)
        for key in _INTERNAL_OBS_KWARGS:
            extra.pop(key, None)
        out_video_path = extra.pop("out_video_path", None)
        policy_images = self._squeeze_single_frame_images(self._cached_images)

        # Multiview policy: extract per-view frames in view_names order.
        view_names = getattr(self._policy, "view_names", None)
        video_views = None
        # Soma composes single-view camera dictionaries itself.
        if view_names and len(view_names) > 1:
            video_views = []
            for name in view_names:
                frame = policy_images.get(name)
                if frame is None:
                    raise RuntimeError(
                        f"Multiview policy expects view '{name}' but it is missing from "
                        f"observation images (got: {list(policy_images.keys())})"
                    )
                video_views.append(frame)
            extra["video_views"] = video_views

        result = self._policy.infer(
            policy_images,
            self._cached_instruction,
            seed=self._inference_seed,
            **extra,
        )

        self._dump_debug_inference(
            policy_images=policy_images,
            video_views=video_views,
            model_context=extra,
            result=result,
        )
        video = result.get("video")
        raw_video = video
        if isinstance(video, (list, tuple)):
            video = self._prepare_prediction_video(video)

        if out_video_path:
            if video is None:
                raise RuntimeError(
                    "Prediction video was requested, but infer() did not return one. "
                    "Set inference.decode_video=True in the policy config."
                )
            self._save_prediction_video(video, out_video_path)

        actions = result["action"]  # (N, action_dim) float32
        logger.info(f"Action shape: {actions.shape}")

        extra = {}
        if self._return_video:
            if video is None:
                logger.warning("return_video is enabled, but infer() did not return a video.")
            else:
                extra["generated_video"] = np.asarray(video, dtype=np.uint8)
                if isinstance(raw_video, (list, tuple)):
                    # Replay metrics compare each view before preview resizing.
                    extra["generated_video_views"] = [
                        np.asarray(view, dtype=np.uint8) for view in raw_video
                    ]

        return Action(action=actions, timestamp=self._cached_obs_timestamp, **extra)

    def get_wire_config(self) -> WireConfig:
        """Pull wire config from Soma's observation configuration."""
        obs = self._cfg.policy.observation
        action = self._cfg.policy.action
        return WireConfig(
            video_action_rate=getattr(action, "video_action_freq", 1),
            video_length=getattr(obs, "video_length", 1),
            image_format=getattr(obs, "image_format", "jpeg"),
            jpeg_quality=getattr(obs, "jpeg_quality", 95),
        )
