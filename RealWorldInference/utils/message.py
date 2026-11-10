"""Wire-format messages between policy server, worker, and controller.

`Observation.images[view]` is normalized to `[T, H, W, 3]` at the executor
wire boundary. Controllers may still produce current-frame `[H, W, 3]`
observations before executor normalization. The on-wire encoding is controlled
by `image_format`:

    "raw":  pickle the ndarray directly (fast, large, lossless)
    "png":  per-frame PNG encode (slow, moderate size, lossless)
    "jpeg": per-frame JPEG encode (fast, smallest, lossy)

`WireConfig` is a handshake object describing what a policy expects on the
wire. Sim workers query `GET /config` on startup and pass the resulting
WireConfig to the executor/ObservationBuffer. The SEP slave agent can also
initialize from ``observation.*`` in the robot config, and optionally apply a
runtime override from ``CMD_START.payload.wire_config``.
"""

import io
import pickle
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import numpy as np

from utils.image import encode_view, decode_view


# ---------------------------------------------------------------------------
# Numpy cross-version pickle compatibility.
#
# numpy 2.0 renamed ``numpy.core`` → ``numpy._core``. Pickling an ndarray
# under numpy 2.x and unpickling under 1.x (or vice versa) raises
# ModuleNotFoundError because the internal module path changed.
#
# Instead of converting every ndarray to a manual dict on the *serialization*
# side, we fix it on the *deserialization* side with a custom Unpickler that
# maps between the two module prefixes. Serialization stays plain pickle.
# ---------------------------------------------------------------------------

class _CrossNumpyUnpickler(pickle.Unpickler):
    """Unpickler that handles numpy.core ↔ numpy._core module rename."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                module = "numpy.core" + module[len("numpy._core"):]
            elif module.startswith("numpy.core"):
                module = "numpy._core" + module[len("numpy.core"):]
            else:
                raise
            return super().find_class(module, name)


def _loads(data: bytes) -> Any:
    """pickle.loads with numpy cross-version compatibility."""
    return _CrossNumpyUnpickler(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------
# Wire config (policy ↔ controller handshake)
# ---------------------------------------------------------------------------

@dataclass
class WireConfig:
    """How the executor should prepare and send Observations to this policy.

    Attributes:
        video_action_rate: Capture one frame per `video_action_rate`.
        video_length: How many buffered frames get_observation() returns.
            1  = latest frame only (non-AR / frame-stacking with N=1)
            N  = latest N frames
            -1 = all frames buffered since last get_observation()  (AR chunk-by-chunk)
        image_format: Wire encoding for image ndarrays. "raw" | "png" | "jpeg".
        jpeg_quality: JPEG quality 0..100 (only used when image_format="jpeg").
    """
    video_action_rate: int = 1
    video_length: int = 1
    image_format: str = "jpeg"
    jpeg_quality: int = 95

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "WireConfig":
        field_names = {field.name for field in fields(WireConfig)}
        return WireConfig(**{k: d[k] for k in d if k in field_names})

    @staticmethod
    def from_observation_config(obs_cfg: Optional[dict]) -> "WireConfig":
        """Build WireConfig from a robot ``observation`` dict (unknown keys ignored)."""
        return WireConfig.from_dict(obs_cfg or {})


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation:
    """Single observation sent from controller/executor to policy server.
    Controllers create `images[view]` as `[H, W, 3]`. Before transmission,
    executors normalize each view to `[T, H, W, 3]` (single-frame observations
    use `T=1`). Encoding format is controlled per-Observation via
    `image_format`.
    """

    def __init__(
        self,
        images: Dict[str, np.ndarray],
        state=None,
        instruction: Optional[str] = None,
        t5_embeddings: Optional[np.ndarray] = None,
        action_constraint: Optional[np.ndarray] = None,
        joint_state: Optional[np.ndarray] = None,
        image_format: str = "jpeg",
        jpeg_quality: int = 95,
        timestamp: Optional[float] = None,
        **kwargs,
    ):
        self.images = images
        self.state = state
        self.instruction = instruction
        self.t5_embeddings = t5_embeddings
        self.action_constraint = action_constraint
        # Robot-space joint positions used as IK warm-start seed for debug
        # preview rendering. Optional — controllers that don't populate it
        # will simply give the IK adapter no seed.
        self.joint_state = joint_state
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.timestamp = timestamp
        self.kwargs = kwargs

    def to_bytes(self) -> bytes:
        encoded_images = {
            name: encode_view(arr, self.image_format, self.jpeg_quality)
            for name, arr in self.images.items()
        }
        payload = {
            "image_format": self.image_format,
            "images": encoded_images,
            "state": self.state,
            "instruction": self.instruction,
            "t5_embeddings": self.t5_embeddings,
            "action_constraint": self.action_constraint,
            "joint_state": self.joint_state,
            "timestamp": self.timestamp,
            **(self.kwargs or {}),
        }
        return pickle.dumps(payload)

    @staticmethod
    def from_bytes(data: bytes) -> "Observation":
        d = _loads(data)
        fmt = d.get("image_format", "png")
        images = {name: decode_view(v, fmt) for name, v in d["images"].items()}
        reserved = {"image_format", "images", "state", "instruction", "t5_embeddings", "action_constraint", "joint_state", "timestamp"}
        kwargs = {k: v for k, v in d.items() if k not in reserved}
        return Observation(
            images=images,
            state=d.get("state"),
            instruction=d.get("instruction"),
            t5_embeddings=d.get("t5_embeddings"),
            action_constraint=d.get("action_constraint"),
            joint_state=d.get("joint_state"),
            image_format=fmt,
            timestamp=d.get("timestamp"),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action:
    def __init__(self, action: np.ndarray, timestamp: Optional[float] = None, **kwargs):
        self.action = action
        self.timestamp = timestamp
        self.kwargs = kwargs

    def to_bytes(self) -> bytes:
        payload = {"action": self.action, "timestamp": self.timestamp, **(self.kwargs or {})}
        return pickle.dumps(payload)

    @staticmethod
    def from_bytes(data: bytes) -> "Action":
        d = _loads(data)
        kwargs = {k: v for k, v in d.items() if k not in ("action", "timestamp")}
        return Action(action=np.asarray(d["action"]), timestamp=d.get("timestamp"), **kwargs)
