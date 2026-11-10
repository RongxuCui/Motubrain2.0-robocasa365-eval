"""RoboCasa365 evaluation worker.

Combines RoboCasa365Controller + HTTPClientPolicy + SyncExecutor. Pulls task
names from a shared flock queue so multiple workers on different GPUs run in
parallel. run_eval.sh builds the queue from a --task-set and starts one worker
per GPU.

RoboCasa needs the `robocasa` conda env (robosuite, mujoco, robocasa v1.0+);
the PolicyServer runs separately in the policy env.

Usage:
    python -m hardware.robocasa365.worker \
        --policy-url http://localhost:24851 \
        --queue /path/to/task_queue.txt \
        --num-episodes 50 \
        --output /path/to/results
"""

import argparse
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.robocasa365.controller import RoboCasa365Controller
from policy.policies.http_client.policy import HTTPClientPolicy
from slave.executors.sync_executor import SyncExecutor
from utils.episode_recorder import EpisodeRecorder, episode_video_path
from utils.logging_utils import setup_logger

logger = setup_logger("RoboCasa365Worker")


def build_executor_config(config: dict) -> dict:
    """Adapt hardware/robocasa365/config.yaml to the executor schema.

    The executor sees the action *as the policy emits it* — the reorder from ARIO
    to the raw env layout happens downstream in controller.publish_action
    (data_wrapper.ActionWrapper), so the layout described here must match the
    policy output layout, selected by config action_layout:

      "ario_groot" (robocasa365-baseline default): the 12-dim ARIO GR00T action
        [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper_close(1)]
        -> dim=12, eef xyz at [5:8], binary dims (control_mode=4, gripper=11)
        excluded from low-pass filtering.
      "env": the raw robosuite order
        [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
        -> dim=12, eef xyz at [0:3], binary dims (gripper=6, control_mode=11).

    Jump detection compares obs.state (16-dim base-relative proprio) against the
    12-dim delta action — different spaces, so it is not meaningful for this
    adapter. We disable it with an effectively-infinite jump_threshold rather
    than leaving the stale 7-dim [(0,3)] segments in place (interpolation and
    filtering are already off in config.yaml). The xyz_segments/no_filter_indices
    are still set correctly so smoothing behaves if it is ever enabled.
    """
    layout = config.get("action_layout", "ario_groot")
    if layout == "ario_groot":
        xyz_segments = [(5, 3)]
        no_filter_indices = [4, 11]   # control_mode, gripper_close (binary)
    elif layout == "env":
        xyz_segments = [(0, 3)]
        no_filter_indices = [6, 11]   # gripper, control_mode (binary)
    else:
        raise ValueError(
            f"action_layout must be 'ario_groot' or 'env', got {layout!r}"
        )

    executor_config = copy.deepcopy(config.get("executor", {}) or {})
    controller_cfg = executor_config.setdefault("controller", {})
    controller_cfg.setdefault("action_type", config.get("action_type", "eef"))
    layout_cfg = controller_cfg.setdefault("layout", {})

    layout_cfg.setdefault("dim", int(config.get("action_dim", 12)))
    layout_cfg.setdefault("quat_segments", None)
    layout_cfg.setdefault("no_filter_indices", no_filter_indices)
    layout_cfg.setdefault("xyz_segments", xyz_segments)
    # 1e9: jump detection is not meaningful across state/action spaces here.
    layout_cfg.setdefault("jump_threshold",
                          float(config.get("eef_jump_threshold", 1e9)))

    return executor_config


def pop_task(queue_file: str, lock_file: str) -> str:
    """Atomically pop the first task from the queue file."""
    with open(lock_file, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            with open(queue_file, "r") as f:
                lines = f.readlines()
            if not lines:
                return ""
            task = lines[0].strip()
            with open(queue_file, "w") as f:
                f.writelines(lines[1:])
            return task
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def run_worker(policy_url: str, queue_file: str, lock_file: str,
               config: dict, num_episodes: int, output_dir: str,
               save_video: bool = False):
    parsed = urlparse(policy_url)
    policy = HTTPClientPolicy(host=parsed.hostname, port=parsed.port or 24851)
    wire_cfg = policy.get_wire_config()
    logger.info(f"Policy wire config: {wire_cfg.to_dict()}")

    controller = RoboCasa365Controller(config)

    executor_config = build_executor_config(config)
    executor = SyncExecutor(executor_config, controller, wire_config=wire_cfg)
    recorder = EpisodeRecorder(enabled=save_video, fps=30)
    executor.on_action_step = recorder.add_obs

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    split = config.get("split", "target")

    while True:
        task = pop_task(queue_file, lock_file)
        if not task:
            logger.info("Queue empty, exiting.")
            break

        logger.info(f"Task: {task} ({num_episodes} episodes)")
        try:
            controller.set_task(task)
        except Exception as e:
            logger.error(f"{task}: set_task failed, skipping: {e}")
            continue

        successes = 0
        task_start = time.time()

        for ep in range(num_episodes):
            ep_start = time.time()
            try:
                controller.reset()
            except Exception as e:
                logger.warning(f"{task} ep{ep}: reset failed: {e}")
                continue

            policy.reset()
            executor.reset()
            recorder.reset()

            while not controller.is_episode_done():
                obs = controller.get_observation()
                if obs is None:
                    break
                policy.update_obs(executor.prepare_obs(obs))
                action = policy.get_action()
                executor.execute(action)

            result = controller.get_episode_result()
            elapsed = time.time() - ep_start

            if result["success"]:
                successes += 1
            status = "Success" if result["success"] else "Fail"
            logger.info(f"  {task} ep{ep}: {status} ({result['steps']} steps, "
                        f"{elapsed:.1f}s) SR={successes}/{ep + 1}")
            recorder.save(episode_video_path(output_dir, task, ep, result["success"]))

        task_elapsed = time.time() - task_start
        sr = successes / max(num_episodes, 1)
        all_results[task] = {
            "success_rate": sr,
            "successes": successes,
            "total": num_episodes,
            "split": split,
            "time": round(task_elapsed, 1),
        }
        logger.info(f">>> {task}: SR={successes}/{num_episodes} ({sr * 100:.0f}%) "
                    f"in {task_elapsed:.0f}s")

        controller.stop()

        with open(os.path.join(output_dir, "results.json"), "w") as f:
            avg_sr = float(np.mean([r["success_rate"] for r in all_results.values()]))
            json.dump({"split": split, "avg_success_rate": avg_sr,
                       "tasks": all_results}, f, indent=2)

    logger.info("Worker done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--queue", required=True, help="Path to task_queue.txt")
    parser.add_argument("--lock", default=None, help="Path to queue.lock")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--split", default=None,
                        choices=["pretrain", "target", "all"],
                        help="Override the split from config.yaml.")
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--save-video", action="store_true",
                        help="Save simulator execution videos under output/exec_videos.")
    args = parser.parse_args()

    lock_file = args.lock or (args.queue + ".lock")

    from hardware import get_config
    config = get_config("robocasa365")
    if args.split:
        config["split"] = args.split

    run_worker(
        policy_url=args.policy_url,
        queue_file=args.queue,
        lock_file=lock_file,
        config=config,
        num_episodes=args.num_episodes,
        output_dir=args.output,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
