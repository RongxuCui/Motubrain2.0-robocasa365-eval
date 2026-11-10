"""RoboCasa365 (robocasa v1.0.1) simulator controller.

Same Controller interface as the other adapters (set_task / reset /
get_observation / publish_action). Differs from hardware/robocasa/ in three ways
that matter:

1. Env construction goes through `robocasa.utils.env_utils.create_env(split=...)`
   instead of a hand-rolled `robosuite.make`. create_env owns the mapping from
   split -> (obj_instance_split, layout_and_style_ids), so we stay aligned with
   the upstream eval protocol: "target" = target objects + the 10 held-out
   scenes, "pretrain" = pretrain objects + the 2500-scene pool.

2. Per-task horizons come from the registry via `get_task_horizon(task)` rather
   than a hardcoded table. 365 horizons range from 300 to well past 1000 steps;
   the v0.2 adapter's flat 500-700 would silently truncate long composite tasks
   and report them as failures.

3. The env is built once per task and reused across episodes. robosuite's
   `base.reset()` already destroys the sim and re-runs `_load_model()` when
   hard_reset=True (robosuite/environments/base.py:270-274), and the kitchen
   draws a fresh layout/style from the env rng each time
   (kitchen.py:595). Rebuilding per episode is redundant and expensive at
   317-task scale. Set rebuild_env_each_episode: true to restore the old
   behavior.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from hardware.controller import Controller
from hardware.robocasa365.data_wrapper import ObservationWrapper, ActionWrapper
from utils.message import Observation, Action
from utils.logging_utils import setup_logger

logger = setup_logger("RoboCasa365Controller")


_import_done = False


def _robosuite_egl_device_ids() -> tuple[Optional[str], Optional[str]]:
    """Resolve physical import and logical runtime EGL device identifiers.

    robosuite validates MUJOCO_EGL_DEVICE_ID against the physical identifiers
    in CUDA_VISIBLE_DEVICES at import time. Its EGL context later enumerates
    only the visible devices and therefore requires the corresponding logical
    ordinal. Convert between those two interpretations without changing the
    worker's CUDA isolation.
    """
    requested_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return requested_device, requested_device

    physical_devices = [
        device.strip() for device in visible_devices.split(",")
    ]
    if not physical_devices or any(
        not device.isdigit() for device in physical_devices
    ):
        return requested_device, requested_device

    if requested_device is None:
        return None, "0"
    if requested_device in physical_devices:
        return (
            requested_device,
            str(physical_devices.index(requested_device)),
        )
    if requested_device.isdigit():
        logical_device = int(requested_device)
        if 0 <= logical_device < len(physical_devices):
            return physical_devices[logical_device], requested_device
    return requested_device, requested_device


def _set_egl_device(device_id: Optional[str]) -> None:
    if device_id is None:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    else:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = device_id


def _ensure_robocasa365(robocasa_dir: Optional[str],
                        robosuite_dir: Optional[str]) -> None:
    """Put the local sim checkouts ahead of anything installed in the env.

    Importing robocasa is what registers the kitchen envs with robosuite.
    """
    global _import_done
    if _import_done:
        return
    for path in (robosuite_dir, robocasa_dir):
        if path and path not in sys.path:
            sys.path.insert(0, path)

    original_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    import_egl_device, runtime_egl_device = _robosuite_egl_device_ids()
    _set_egl_device(import_egl_device)
    try:
        import robocasa  # noqa: F401

        # Guard against pointing at a v0.2 checkout: the task registry was
        # renamed in 1.0, and a silent mismatch here produces confusing
        # downstream errors.
        from robocasa.utils import dataset_registry

        if not hasattr(dataset_registry, "ATOMIC_TASK_DATASETS"):
            raise RuntimeError(
                f"robocasa at {robocasa_dir!r} has no ATOMIC_TASK_DATASETS — "
                "this looks like robocasa v0.2 (SINGLE_STAGE/MULTI_STAGE "
                "registry). Point ROBOCASA365_DIR at a v1.0+ checkout, or use "
                "the `robocasa` adapter instead of `robocasa365`."
            )
    except Exception:
        _set_egl_device(original_egl_device)
        raise

    _set_egl_device(runtime_egl_device)
    _import_done = True


def resolve_task_set(task_set: str, robocasa_dir: Optional[str] = None,
                     robosuite_dir: Optional[str] = None) -> List[str]:
    """Expand a TASK_SET_REGISTRY key into its task-name list.

    Valid keys include target50, atomic_seen, composite_seen, composite_unseen,
    pretrain50/100/200/300, all_atomic_tasks, all_composite_tasks, all_tasks,
    and lifelong_learning_phase1..4.
    """
    _ensure_robocasa365(robocasa_dir, robosuite_dir)
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    if task_set not in TASK_SET_REGISTRY:
        raise ValueError(
            f"unknown task_set {task_set!r}. Available: "
            f"{sorted(TASK_SET_REGISTRY)}"
        )
    return list(TASK_SET_REGISTRY[task_set])


# The three target-split subsets, in report order. target50 is exactly their
# concatenation (dataset_registry.py: target50=atomic_seen + composite_seen +
# composite_unseen), and they are pairwise disjoint, so grouping an eval by
# these keys partitions a target50 run with nothing double-counted.
TARGET_SUBSETS = ("atomic_seen", "composite_seen", "composite_unseen")


def task_subset_map(robocasa_dir: Optional[str] = None,
                    robosuite_dir: Optional[str] = None) -> Dict[str, str]:
    """Map each target-split task name to its subset key.

    Tasks outside the target split (e.g. pretrain-only tasks) are absent; the
    caller decides how to report those.
    """
    _ensure_robocasa365(robocasa_dir, robosuite_dir)
    from robocasa.utils.dataset_registry import TARGET_TASKS

    return {
        task: subset
        for subset in TARGET_SUBSETS
        for task in TARGET_TASKS[subset]
    }


def summarize_by_subset(task_results: Dict[str, dict],
                        robocasa_dir: Optional[str] = None,
                        robosuite_dir: Optional[str] = None) -> Dict[str, dict]:
    """Group per-task results into the three target-split subsets.

    Args:
        task_results: {task_name: {"success_rate", "successes", "total", ...}}.
        robocasa_dir / robosuite_dir: forwarded to task_subset_map.

    Returns:
        {subset: {"success_rate", "successes", "total", "num_tasks", "tasks"}}
        in TARGET_SUBSETS order, holding only subsets that have results. Tasks
        outside the target split (e.g. a pretrain50 run) land under "other", so
        the breakdown never silently drops a task.

    `success_rate` is the unweighted mean over that subset's tasks, matching how
    the headline avg_success_rate is computed. `successes`/`total` are episode
    sums, which differ from the mean only if tasks ran unequal episode counts.
    """
    subset_of = task_subset_map(robocasa_dir, robosuite_dir)

    grouped: Dict[str, list] = {}
    for task in sorted(task_results):
        grouped.setdefault(subset_of.get(task, "other"), []).append(task)

    summary = {}
    for subset in (*TARGET_SUBSETS, "other"):
        tasks = grouped.get(subset)
        if not tasks:
            continue
        rates = [task_results[t]["success_rate"] for t in tasks]
        summary[subset] = {
            "success_rate": sum(rates) / len(rates),
            "successes": sum(int(task_results[t]["successes"]) for t in tasks),
            "total": sum(int(task_results[t]["total"]) for t in tasks),
            "num_tasks": len(tasks),
            "tasks": tasks,
        }
    return summary


class RoboCasa365Controller(Controller):
    """RoboCasa365 MuJoCo kitchen env as a Controller."""

    def __init__(self, config: Dict[str, Any], in_collect: bool = False):
        config = dict(config)
        config.setdefault("action_type", "eef")
        super().__init__(config, in_collect)

        self._robocasa_dir = config.get("robocasa365_dir")
        self._robosuite_dir = config.get("robosuite_dir")
        self._robots = config.get("robots", "PandaOmron")
        self._split = config.get("split", "target")
        if self._split not in ("pretrain", "target", "all"):
            raise ValueError(
                f'split must be one of "pretrain"/"target"/"all", got '
                f"{self._split!r}"
            )
        self._cam_h = int(config.get("camera_height", 224))
        self._cam_w = int(config.get("camera_width", 224))
        self._flip_images = bool(config.get("flip_images", True))
        self._randomize_cameras = bool(config.get("randomize_cameras", False))
        self._proprio = config.get("proprio", "base_relative")
        self._max_steps_default = int(config.get("max_steps", 500))
        self._num_steps_wait = int(config.get("num_steps_wait", 10))
        self._base_and_mode_action = np.asarray(
            config.get("base_and_mode_action", [0.0, 0.0, 0.0, 0.0, -1.0]),
            dtype=np.float32,
        )
        self._action_layout = config.get("action_layout", "ario_groot")
        self._deterministic = bool(config.get("deterministic", True))
        self._base_seed = int(config.get("seed", 195))
        self._rebuild_each_episode = bool(
            config.get("rebuild_env_each_episode", False)
        )

        self._env = None
        self._task_name: Optional[str] = None
        self._episode_idx = 0
        self._step_count = 0
        self._max_steps = self._max_steps_default
        self._success = False
        self._instruction: Optional[str] = None
        self._latest_raw_obs: Optional[dict] = None

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def set_task(self, task_name: str):
        """Select a task. Call before reset().

        `task_name` must be a key in ATOMIC_TASK_DATASETS or
        COMPOSITE_TASK_DATASETS, e.g. "CloseBlenderLid", "MixCakeFrosting".
        """
        _ensure_robocasa365(self._robocasa_dir, self._robosuite_dir)

        # Drop any env built for the previous task.
        self._close_env()

        self._task_name = task_name
        self._episode_idx = 0
        self._max_steps = self._resolve_horizon(task_name)
        logger.info(
            f"Task: {task_name}  split={self._split}  max_steps={self._max_steps}"
        )

    def _resolve_horizon(self, task_name: str) -> int:
        """Per-task horizon from the registry, falling back to config max_steps."""
        try:
            from robocasa.utils.dataset_registry_utils import get_task_horizon

            return int(get_task_horizon(task_name))
        except Exception as e:
            logger.warning(
                f"no registry horizon for {task_name} ({e}); "
                f"falling back to max_steps={self._max_steps_default}"
            )
            return self._max_steps_default

    # ------------------------------------------------------------------
    # Env lifecycle
    # ------------------------------------------------------------------

    def _close_env(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _build_env(self):
        """Build the env for the current task via the upstream create_env."""
        _ensure_robocasa365(self._robocasa_dir, self._robosuite_dir)
        from robocasa.utils.env_utils import create_env

        # One seed per (task, split) run. robosuite seeds its rng once at
        # construction, and each reset() draws the next layout/style from that
        # rng, so a single seed gives a deterministic *sequence* of episodes.
        seed = self._base_seed if self._deterministic else None

        self._env = create_env(
            env_name=self._task_name,
            robots=self._robots,
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            camera_widths=self._cam_w,
            camera_heights=self._cam_h,
            seed=seed,
            split=self._split,
            randomize_cameras=self._randomize_cameras,
            translucent_robot=False,
        )
        logger.info(
            f"Built env: task={self._task_name} split={self._split} "
            f"seed={seed} action_dim={self._env.action_dim}"
        )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset into a fresh episode, settle physics, fetch the first obs."""
        if self._task_name is None:
            raise RuntimeError("Call set_task() before reset()")

        if self._env is None or self._rebuild_each_episode:
            self._close_env()
            self._build_env()

        self._env.reset()

        # Settle physics with zero actions before the policy takes over.
        zero_action = np.zeros(self._env.action_dim, dtype=np.float32)
        raw_obs = None
        for _ in range(self._num_steps_wait):
            raw_obs, _, _, _ = self._env.step(zero_action)

        if raw_obs is None:
            # num_steps_wait == 0: pull observations without advancing the sim.
            raw_obs = self._env._get_observations(force_update=True)

        # The task-conditioned instruction is only valid after reset.
        try:
            self._instruction = self._env.get_ep_meta().get("lang", None) or None
        except Exception as e:
            logger.warning(f"get_ep_meta failed: {e}")
            self._instruction = None

        self._step_count = 0
        self._success = False
        self._latest_raw_obs = raw_obs
        self._episode_idx += 1
        logger.info(
            f"Episode {self._episode_idx}: task={self._task_name} "
            f"layout={getattr(self._env, 'layout_id', None)} "
            f"style={getattr(self._env, 'style_id', None)} "
            f"instr={self._instruction!r}"
        )

    def get_observation(self) -> Optional[Observation]:
        if self._env is None or self._latest_raw_obs is None:
            return None
        obs = ObservationWrapper.wrap(
            self._latest_raw_obs,
            flip_images=self._flip_images,
            proprio=self._proprio,
        )
        obs.instruction = self._instruction
        return obs

    def publish_action(self, action: Action):
        if self._env is None:
            return
        # ario_groot layout routes the action through the robot's own
        # create_action_vector so the env layout is derived from the live
        # controller (identical to the official gym_wrapper eval), not assumed.
        vec_factory = (
            self._env.robots[0].create_action_vector
            if self._action_layout == "ario_groot"
            else None
        )
        raw_action = ActionWrapper.unwrap(
            action,
            env_action_dim=int(self._env.action_dim),
            base_and_mode_action=self._base_and_mode_action,
            action_layout=self._action_layout,
            vec_factory=vec_factory,
        )
        raw_obs, _reward, _done, _info = self._env.step(raw_action)
        self._latest_raw_obs = raw_obs
        self._step_count += 1

        # Latch success: some tasks can transiently satisfy then break the
        # condition, and the benchmark scores "did it ever succeed".
        try:
            if self._env._check_success():
                self._success = True
        except Exception as e:
            logger.warning(f"_check_success failed: {e}")

    def is_episode_done(self) -> bool:
        if self._env is None:
            return True
        return self._success or self._step_count >= self._max_steps

    def get_episode_result(self) -> dict:
        return {
            "task": self._task_name,
            "episode": self._episode_idx - 1,
            "success": bool(self._success),
            "steps": int(self._step_count),
            "split": self._split,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        pass

    def stop(self):
        self._close_env()
