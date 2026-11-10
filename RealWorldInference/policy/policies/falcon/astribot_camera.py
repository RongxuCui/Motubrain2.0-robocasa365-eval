"""Astribot S1 head-camera extrinsics derived from runtime joint state."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


_TORSO_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.12), (-1.5708, 1.5708, 0.0)),
    ((-0.38, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.390, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, 0.0, -1.5708)),
)
_TORSO_EEF_XYZ = (0.0, 0.0, 0.34475)
_HEAD_FIXED_XYZ = (0.0, 0.0, 0.1229)
_HEAD_JOINT_2_RPY = (-1.5708, 0.0, 0.0)
_SUPPORTED_JOINT_STATE_DIMS = (20, 22)


def _transform(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _rotate_z(theta: float) -> np.ndarray:
    return _transform(rpy=(0.0, 0.0, float(theta)))


class AstribotHeadCameraExtrinsicProvider:
    """Compute ``T_base_camera`` using the training URDF's torso/head chain."""

    def __init__(self, calibration_path: str) -> None:
        path = Path(calibration_path).expanduser().resolve()
        with path.open(encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        matrix = np.asarray(calibration["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                f"Astribot head-camera calibration must be a finite 4x4 matrix, got {matrix.shape}"
            )
        self._head_camera = matrix

    def __call__(self, joint_state: np.ndarray) -> np.ndarray:
        joints = np.asarray(joint_state, dtype=np.float64).reshape(-1)
        if joints.size not in _SUPPORTED_JOINT_STATE_DIMS:
            raise ValueError(
                "Astribot camera FK requires joint_state with exactly 20 or "
                f"22 values, got {joints.size}"
            )
        if not np.isfinite(joints).all():
            raise ValueError("Astribot camera FK joint_state contains NaN or Inf")

        base_torso = np.eye(4, dtype=np.float64)
        for joint, (xyz, rpy) in zip(
            joints[:4],
            _TORSO_JOINT_ORIGINS,
            strict=True,
        ):
            base_torso = base_torso @ _transform(xyz, rpy) @ _rotate_z(joint)
        base_torso = base_torso @ _transform(_TORSO_EEF_XYZ)

        torso_head = (
            _transform(_HEAD_FIXED_XYZ)
            @ _rotate_z(joints[4])
            @ _transform(rpy=_HEAD_JOINT_2_RPY)
            @ _rotate_z(joints[5])
        )
        return (base_torso @ torso_head @ self._head_camera).astype(np.float32)
