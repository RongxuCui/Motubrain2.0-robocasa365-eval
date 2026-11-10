"""Episode video recorder for simulator workers.

The recorder is intentionally independent from controllers. Controllers expose
current observations; workers decide whether those observations should be saved
as evaluation artifacts.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from utils.image import save_video
from utils.message import Observation


class EpisodeRecorder:
    """Collect observation frames and save a single MP4 per episode."""

    def __init__(
        self,
        enabled: bool = False,
        fps: int = 30,
        view_mode: str = "auto",
        max_frames: Optional[int] = None,
    ):
        self.enabled = enabled
        self.fps = fps
        self.view_mode = view_mode
        self.max_frames = max_frames
        self._frames: list[np.ndarray] = []

    def reset(self):
        self._frames.clear()

    def add_obs(self, obs: Optional[Observation]):
        if not self.enabled or obs is None or not obs.images:
            return
        frame = self._observation_to_frame(obs)
        if frame is None:
            return
        if self.max_frames is not None and len(self._frames) >= self.max_frames:
            return
        self._frames.append(frame)

    def save(self, path: str):
        if not self.enabled or not self._frames:
            return
        save_video(path, self._frames, fps=self.fps)

    @staticmethod
    def _latest_frame(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.uint8)
        return arr[-1] if arr.ndim == 4 else arr

    def _observation_to_frame(self, obs: Observation) -> Optional[np.ndarray]:
        images = {
            name: self._latest_frame(image)
            for name, image in obs.images.items()
        }
        if not images:
            return None

        if self.view_mode == "composite" or (
            self.view_mode == "auto" and len(images) > 1
        ):
            return compose_standard_views(images)

        first = next(iter(images.values()))
        return np.asarray(first, dtype=np.uint8)


def compose_standard_views(images: Dict[str, np.ndarray]) -> np.ndarray:
    """Compose standard robot views into one debug frame.

    Default layout:
      - If standard tri-view exists: high camera on top, left/right wrist below.
      - Otherwise: concatenate all available views horizontally.
    """
    if "composite" in images:
        return np.asarray(images["composite"], dtype=np.uint8)

    if "cam_high" in images:
        head = np.asarray(images["cam_high"], dtype=np.uint8)
        left = images.get("cam_left_wrist")
        right = images.get("cam_right_wrist")
        if left is not None and right is not None:
            cv2 = __import__("cv2")
            hh, hw = head.shape[:2]
            half_h, half_w = hh // 2, hw // 2
            left_resized = cv2.resize(np.asarray(left, dtype=np.uint8), (half_w, half_h))
            right_resized = cv2.resize(np.asarray(right, dtype=np.uint8), (half_w, half_h))
            bottom = np.concatenate([left_resized, right_resized], axis=1)
            return np.concatenate([head, bottom], axis=0)
        return head

    frames = [np.asarray(frame, dtype=np.uint8) for frame in images.values()]
    target_h = min(frame.shape[0] for frame in frames)
    if any(frame.shape[0] != target_h for frame in frames):
        cv2 = __import__("cv2")
        frames = [
            cv2.resize(frame, (int(frame.shape[1] * target_h / frame.shape[0]), target_h))
            for frame in frames
        ]
    return np.concatenate(frames, axis=1)


def episode_video_path(output_dir: str, task: str, episode: int, success: bool) -> str:
    safe_task = task.replace("/", "__")
    video_dir = os.path.join(output_dir, "exec_videos")
    name = f"{safe_task}_ep{episode:02d}_{'ok' if success else 'fail'}.mp4"
    return os.path.join(video_dir, name)


def prediction_video_path(
    output_dir: str,
    task: str,
    episode: int,
    chunk_index: int,
) -> str:
    """Return the path for one model-predicted action-chunk video."""
    safe_task = task.replace("/", "__")
    video_dir = os.path.join(output_dir, "prediction_videos")
    name = f"{safe_task}_ep{episode:02d}_chunk{chunk_index:04d}.mp4"
    return os.path.join(video_dir, name)
