"""Unified image & video I/O utilities.

All image encoding, decoding, format conversion, and video writing lives here.
Callers should never import cv2 or imageio directly for these tasks.

Convention: current-frame images are **RGB uint8 [H, W, 3]** in memory; wire
views may be time-stacked as **[T, H, W, 3]**. BGR conversion is handled
internally when calling cv2 encode/decode.
"""

import base64
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Single-frame encode / decode
# ---------------------------------------------------------------------------

def encode_frame(frame: np.ndarray, fmt: str = "jpeg", quality: int = 95) -> bytes:
    """Encode a single [H, W, 3] RGB uint8 frame to bytes.

    Args:
        frame: RGB image array.
        fmt: "raw" | "png" | "jpeg".
        quality: JPEG quality 0..100 (ignored for raw/png).

    Returns:
        Encoded bytes.
    """
    if fmt == "raw":
        return frame.tobytes()
    bgr = frame[..., ::-1]  # RGB -> BGR for cv2
    if fmt == "png":
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    elif fmt == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        raise ValueError(f"Unknown image format: {fmt!r}")
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for format {fmt!r}")
    return buf.tobytes()


def decode_frame(data: bytes, fmt: str = "jpeg", shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Decode bytes back to a single [H, W, 3] RGB uint8 frame.

    Args:
        data: Encoded bytes.
        fmt: "raw" | "png" | "jpeg".
        shape: Required for "raw" format — the (H, W, 3) shape.

    Returns:
        RGB uint8 ndarray.
    """
    if fmt == "raw":
        if shape is None:
            raise ValueError("shape is required for raw format decoding")
        return np.frombuffer(data, dtype=np.uint8).reshape(shape)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return bgr[..., ::-1]  # BGR -> RGB


# ---------------------------------------------------------------------------
# Multi-frame (view) encode / decode — for wire protocol
# ---------------------------------------------------------------------------

def encode_view(arr: np.ndarray, fmt: str = "jpeg", quality: int = 95) -> dict:
    """Encode a view ndarray ([H,W,3] or [T,H,W,3]) to a wire-format dict.

    Returns:
        {"shape": tuple, "frames": [bytes, ...]}
    """
    if arr.ndim == 3:
        return {"shape": arr.shape, "frames": [encode_frame(arr, fmt, quality)]}
    if arr.ndim == 4:
        return {"shape": arr.shape, "frames": [encode_frame(arr[i], fmt, quality) for i in range(arr.shape[0])]}
    raise ValueError(f"Image ndarray must be 3D or 4D, got shape {arr.shape}")


def decode_view(d: dict, fmt: str = "jpeg") -> np.ndarray:
    """Decode a wire-format dict back to [H,W,3] or [T,H,W,3].

    Args:
        d: {"shape": tuple, "frames": [bytes, ...]}
        fmt: "raw" | "png" | "jpeg".
    """
    shape = tuple(d["shape"])
    frames_bytes = d["frames"]
    if fmt == "raw":
        frame_shape = shape if len(shape) == 3 else shape[1:]
        frames = [decode_frame(b, fmt, shape=frame_shape) for b in frames_bytes]
    else:
        frames = [decode_frame(b, fmt) for b in frames_bytes]
    if len(shape) == 3:
        return frames[0]
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# JPEG + Base64 (for WebSocket / web preview streaming)
# ---------------------------------------------------------------------------

def rgb_to_jpeg_b64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB frame to a Base64-encoded JPEG string.

    Common pattern used by WebSocket streaming (slave agent, web dashboard,
    debug preview). Consolidates the scattered RGB→BGR→imencode→b64 logic.

    Args:
        frame: [H, W, 3] RGB uint8.
        quality: JPEG quality 0..100.

    Returns:
        Base64 ASCII string of the JPEG bytes.
    """
    data = encode_frame(frame, fmt="jpeg", quality=quality)
    return base64.b64encode(data).decode("ascii")


def rgb_to_jpeg_data_uri(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB frame to a data URI string for HTML <img> tags.

    Returns:
        "data:image/jpeg;base64,..." string.
    """
    return f"data:image/jpeg;base64,{rgb_to_jpeg_b64(frame, quality)}"


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def save_video(
    path: str,
    frames: Sequence[np.ndarray],
    fps: int = 20,
    codec: str = "libx264",
) -> None:
    """Save a list of RGB uint8 frames as an MP4 video (H.264).

    Uses imageio + ffmpeg. Never use cv2.VideoWriter — its mp4v codec is
    unviewable in most preview tools (VSCode, browsers).

    Args:
        path: Output file path (should end with .mp4).
        frames: Sequence of [H, W, 3] uint8 RGB ndarrays.
        fps: Frames per second.
        codec: FFmpeg codec name.
    """
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import imageio.v2 as imageio
    imageio.mimwrite(path, frames, fps=fps, codec=codec)
