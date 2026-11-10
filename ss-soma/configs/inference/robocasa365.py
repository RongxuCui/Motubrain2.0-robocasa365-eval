"""Direct single-GPU RoboCasa365 policy for a four-step memory student.

Sampling uses CMSolver on the trained four-step grid, without CFG or DiT
caching. Supply the student safetensors checkpoint separately at launch.

The checkpoint's training snapshot supplies the compressed-history stride,
prediction window and action frequency. The VAE and text-encoder weights
come from ``SOMA_VAE_PATH`` and ``SOMA_TEXT_ENCODER_PATH``. Preserve
all captured observations for causal five-frame history encoding. Actions
remain in the raw 12-D layout consumed by the RWI RoboCasa365 adapter.
"""

import os

from morph.convs.image import ResizeConfig
from soma.data import conditions
from soma.inference.config import ActionConfig
from soma.inference.config import CompileConfig
from soma.inference.config import GeometryConfig
from soma.inference.config import InferenceConfig
from soma.inference.config import MultiviewConfig
from soma.inference.config import ObservationConfig
from soma.inference.config import PolicyConfig
from soma.inference.config import SimpleRTCConfig
from soma.inference.config import SparseMemoryConfig
from soma.inference.config import TextEncoderConfig
from soma.inference.config import VaeConfig
from soma.inference.diffusion import solvers


# Wire camera -> training camera, in the training packing order.
_VIEW_EMBED_ALIASES = {
    "cam_agentview_left": "cam_high",
    "cam_eye_in_hand": "cam_right_wrist",
    "cam_agentview_right": "cam_side",
}
_VIEW_NAMES = tuple(_VIEW_EMBED_ALIASES)
_MAIN_RESIZE = ResizeConfig(
    mode="area",
    target_area=120_000,
    multiple=16,
    interpolation="bicubic",
)
_WRIST_RESIZE = ResizeConfig(
    mode="area",
    target_area=30_000,
    multiple=16,
    interpolation="bicubic",
)


policy = PolicyConfig(
    type="soma.inference.policies.motubrain.MotubrainPolicy",
    dit_factory="soma.inference.runtime.motubrain.build_model",
    vae=VaeConfig(
        vae_pth=os.environ["SOMA_VAE_PATH"],
        precision="bf16",
        incremental_encode=False,
    ),
    text_encoder=TextEncoderConfig(
        pretrained_model_name_or_path=os.environ["SOMA_TEXT_ENCODER_PATH"],
    ),
    geometry=GeometryConfig(
        config_path=os.environ["DA3_CONFIG_PATH"],
        checkpoint_path=os.environ["DA3_CHECKPOINT_PATH"],
        layer_map={
            layer: layer
            for layer in (10, 12, 14, 16, 18, 20, 22, 24)
        },
        views={
            name: conditions.ConditionViewConfig(name)
            for name in _VIEW_NAMES
        },
        image_size=(280, 504),
        required_image_names=_VIEW_NAMES,
    ),
    observation=ObservationConfig(
        resize=_MAIN_RESIZE,
        num_frames=13,
        # Memory consumes every captured observation since the last request.
        video_length=-1,
        image_format="raw",
        jpeg_quality=95,
    ),
    action=ActionConfig(
        video_action_freq=6,
        profile=None,
    ),
    prompt_builder=None,
    multiview=MultiviewConfig(
        num_views=len(_VIEW_NAMES),
        view_names=list(_VIEW_NAMES),
        rope_mode="per_view",
        view_resizes={
            "cam_agentview_left": _MAIN_RESIZE,
            "cam_eye_in_hand": _WRIST_RESIZE,
            "cam_agentview_right": _MAIN_RESIZE,
        },
        view_embed_aliases=_VIEW_EMBED_ALIASES,
    ),
    sparse_memory=SparseMemoryConfig(enabled=True),
)


inference = InferenceConfig(
    steps=4,
    timeshift=6.0,
    action_timeshift=1.0,
    solver=solvers.CMSolver,
    cfg_scale=None,
    dit_cache_threshold=None,
    dit_cache_threshold_2=None,
    decode_video=False,
    action_smoothing_window=0,
    simple_rtc=SimpleRTCConfig(enabled=False),
    compile=CompileConfig(),
    context_parallel_size=1,
    int8=False,
)
