"""Data wrappers for RoboCasa365 (robocasa v1.0.1).

Converts between robosuite observation dicts and RealWorldInference messages.
Parallels hardware/robocasa/data_wrapper.py but follows the 365 conventions:

Cameras keep RoboCasa's own naming rather than being remapped onto the
aloha-style cam_high / cam_left_wrist / cam_right_wrist convention:
    robot0_agentview_left_image   -> cam_agentview_left
    robot0_agentview_right_image  -> cam_agentview_right
    robot0_eye_in_hand_image      -> cam_eye_in_hand

The v0.2 adapter mapped eye_in_hand onto "cam_left_wrist", which misdescribes
it: RoboCasa has exactly one wrist camera and there is no left/right pair, so
that name implies a second wrist view that does not exist. The two agentviews
are both third-person, so calling one "cam_high" and the other "cam_high_right"
loses the left/right distinction that the source names carry.

Proprio has two modes:
  - "base_relative" (default) mirrors the official 365 GR00T mapping in
    robocasa/wrappers/gym_wrapper.py:map_obs —
      [gripper_qpos(2), base_to_eef_pos(3), base_to_eef_quat(4),
       base_pos(3), base_quat(4)]  = 16 dims
    Base-relative eef is the right choice for a mobile base: the same reach
    looks identical regardless of where the robot parked.
  - "absolute" reproduces the older world-frame layout used by the v0.2 adapter:
      [gripper_qpos(2), eef_pos(3), eef_quat(4)]  = 9 dims

Pick whichever matches how the checkpoint was trained.

Actions: PandaOmron runs HYBRID_MOBILE_BASE, a 12-dim action laid out (per
robocasa/utils/env_utils.py:convert_action) as
    [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
A 7-dim manipulation-only policy is padded with `base_and_mode_action`, whose
last element must stay < 0 to keep the arm controller (not the base) active —
see robosuite HybridMobileBase.set_goal.
"""

import numpy as np

from utils.message import Observation, Action


# Camera view -> RealWorldInference image key. These names are the policy input
# contract, not just video labels: FalconPolicy looks up its `view_names`
# against these keys (policy/policies/falcon/policy.py), so they must match
# whatever the checkpoint was trained with.
VIEW_MAP = {
    "robot0_agentview_left_image": "cam_agentview_left",
    "robot0_agentview_right_image": "cam_agentview_right",
    "robot0_eye_in_hand_image": "cam_eye_in_hand",
}

# Proprio key order per mode. Every key is a robosuite observable for a
# WheeledRobot (see robosuite/robots/mobile_robot.py + robot.py).
PROPRIO_KEYS = {
    "base_relative": (
        "robot0_gripper_qpos",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_base_pos",
        "robot0_base_quat",
    ),
    "absolute": (
        "robot0_gripper_qpos",
        "robot0_eef_pos",
        "robot0_eef_quat",
    ),
}


def _maybe_flip(img: np.ndarray, flip: bool) -> np.ndarray:
    if not flip:
        return img
    # np.flipud returns a view; copy so downstream pickle/jpeg encoding is safe.
    return np.ascontiguousarray(np.flipud(img))


class ObservationWrapper:
    @staticmethod
    def wrap(raw_obs: dict, flip_images: bool = True,
             proprio: str = "base_relative") -> Observation:
        """Convert a robosuite obs dict into an Observation.

        Args:
            raw_obs: dict from env.step()/env.reset() with uint8 image arrays and
                low-dim proprio keys.
            flip_images: flip vertically to undo robosuite's bottom-up images.
            proprio: "base_relative" or "absolute" (see module docstring).

        Returns:
            Observation with `images` keyed by RealWorldInference camera names and
            a flat float32 `state`.
        """
        if proprio not in PROPRIO_KEYS:
            raise ValueError(
                f"proprio must be one of {sorted(PROPRIO_KEYS)}, got {proprio!r}"
            )

        images = {}
        for rc_key, out_key in VIEW_MAP.items():
            if rc_key in raw_obs:
                img = np.asarray(raw_obs[rc_key], dtype=np.uint8)
                images[out_key] = _maybe_flip(img, flip_images)

        keys = PROPRIO_KEYS[proprio]
        missing = [k for k in keys if k not in raw_obs]
        if missing:
            raise KeyError(
                f"proprio={proprio!r} needs {missing}, absent from the observation. "
                f"Available low-dim keys: "
                f"{sorted(k for k in raw_obs if not k.endswith('_image'))}"
            )
        state = np.concatenate(
            [np.asarray(raw_obs[k], dtype=np.float32).reshape(-1) for k in keys]
        )

        return Observation(images=images, state=state)


# ARIO GR00T action layout (PandaOmron_modality.json), as emitted by the
# robocasa365-baseline checkpoint:
#   [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper_close(1)]
# NOTE base_motion is itself [base_xyz(3), torso(1)] — see unmap_action, which
# routes base_motion[0:3] to the "base" part and base_motion[3:4] to "torso".
_ARIO = {
    "base_motion": slice(0, 4),
    "control_mode": slice(4, 5),
    "eef_pos": slice(5, 8),
    "eef_rot": slice(8, 11),
    "gripper_close": slice(11, 12),
}


class ActionWrapper:
    """Convert a policy Action into a raw robosuite action array."""

    @staticmethod
    def _ario_to_env(a: np.ndarray, create_action_vector) -> np.ndarray:
        """Map a 12-dim ARIO GR00T action onto the raw robosuite env layout.

        Rather than hard-coding the flat env order, we build the same named
        part dict as robocasa/wrappers/gym_wrapper.py:unmap_action and hand it to
        the robot's ``create_action_vector`` (robosuite HybridMobileBase). That
        routine places each part at the offset defined by the live controller's
        ``_action_split_indexes`` and puts base_mode at [-1], so the result is
        bitwise-identical to the official 365 GR00T eval — no assumption about
        the torso/base/gripper ordering.

        Two transforms mirror unmap_action:
          - base_motion(4) splits into base=[0:3] (x/y/yaw) and torso=[3:4];
          - the binary dims (gripper_close, control_mode) are in robosuite's
            native [-1, 1] convention in this dataset (gripper: -1=open,
            +1=close; control_mode: <0=arm, >0=base), NOT the 0/1 encoding that
            gym_wrapper.unmap_action thresholds at 0.5. We threshold at 0 to a
            clean {-1, +1}; thresholding at 0.5 would misclassify any output in
            [0, 0.5) (robosuite only looks at the sign anyway).
        The eef and base/torso dims are otherwise continuous and pass through.

        Args:
            a: 12-dim ARIO GR00T action vector.
            create_action_vector: env.robots[0].create_action_vector — maps a
                {part_name: sub_action} dict to the env's flat action array.
        """
        if a.shape[0] != 12:
            raise ValueError(
                f"action_layout='ario_groot' expects a 12-dim policy action, "
                f"got {a.shape[0]} dims"
            )
        base_motion = a[_ARIO["base_motion"]]
        gripper = 1.0 if a[_ARIO["gripper_close"]][0] >= 0.0 else -1.0
        base_mode = 1.0 if a[_ARIO["control_mode"]][0] >= 0.0 else -1.0
        action_dict = {
            # OSC_POSE arm controller: 6-dim delta [pos(3), rot(3)].
            "right": np.concatenate(
                [a[_ARIO["eef_pos"]], a[_ARIO["eef_rot"]]]
            ).astype(np.float32),
            "right_gripper": np.array([gripper], dtype=np.float32),
            "base": base_motion[0:3].astype(np.float32),
            "torso": base_motion[3:4].astype(np.float32),
            "base_mode": base_mode,
        }
        return np.asarray(create_action_vector(action_dict), dtype=np.float32)

    @staticmethod
    def unwrap(action: Action, env_action_dim: int,
               base_and_mode_action: np.ndarray,
               action_layout: str = "ario_groot",
               vec_factory=None) -> np.ndarray:
        """Convert a policy Action into the env's raw action array.

        action_layout:
          "ario_groot": policy emits the 12-dim ARIO GR00T layout. Requires
              vec_factory=env.robots[0].create_action_vector, which places each
              named part at the correct controller index — identical to the
              official gym_wrapper path (see _ario_to_env).
          "env": policy already emits the raw env order. A vector matching
              env_action_dim passes through; a 7-dim manipulation action is
              padded with base_and_mode_action (base_motion(4) + control_mode(1)).
        """
        a = np.asarray(action.action, dtype=np.float32).reshape(-1)

        if action_layout == "ario_groot":
            if vec_factory is None:
                raise ValueError(
                    "action_layout='ario_groot' requires vec_factory="
                    "env.robots[0].create_action_vector"
                )
            mapped = ActionWrapper._ario_to_env(a, vec_factory)
            if mapped.shape[0] != env_action_dim:
                raise ValueError(
                    f"mapped ARIO action has {mapped.shape[0]} dims but the env "
                    f"expects {env_action_dim}"
                )
            return mapped

        if action_layout != "env":
            raise ValueError(
                f"action_layout must be 'ario_groot' or 'env', got {action_layout!r}"
            )

        if a.shape[0] == env_action_dim:
            return a

        pad = np.asarray(base_and_mode_action, dtype=np.float32).reshape(-1)
        if a.shape[0] + pad.shape[0] == env_action_dim:
            return np.concatenate([a, pad])

        raise ValueError(
            f"cannot map a {a.shape[0]}-dim policy action to the env's "
            f"{env_action_dim}-dim action space "
            f"(base_and_mode_action has {pad.shape[0]} dims)"
        )
