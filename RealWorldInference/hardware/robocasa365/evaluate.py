#!/usr/bin/env python3
"""RoboCasa365 batch evaluation using FalconPolicy + RoboCasa365Controller.

Single-process evaluator. For multi-GPU runs use run_eval.sh (which drives
worker.py against a shared task queue).

Task selection follows the upstream 365 protocol: name a `--task-set` from
robocasa.utils.dataset_registry.TASK_SET_REGISTRY and a `--split`, rather than
listing tasks by hand. `--tasks` is still available for spot checks.

Usage:
    # One task set, target split
    CUDA_VISIBLE_DEVICES=0 python -m hardware.robocasa365.evaluate \
        --config-path /path/to/ss-inference/configs/embodied/robocasa365.py \
        --checkpoint /path/to/step-50000-ema.safetensors \
        --task-set target50 --split target --num-episodes 50

    # Spot-check a couple of tasks
    CUDA_VISIBLE_DEVICES=0 python -m hardware.robocasa365.evaluate \
        --policy-url http://localhost:24851 \
        --tasks CloseBlenderLid CloseCabinet --num-episodes 5

    # List what task sets exist and exit
    python -m hardware.robocasa365.evaluate --list-task-sets
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.robocasa365.controller import (
    RoboCasa365Controller,
    resolve_task_set,
    summarize_by_subset,
)
from utils.logging_utils import setup_logger

logger = setup_logger("RoboCasa365Eval")


def evaluate_task(policy, controller, task_name, num_episodes):
    """Evaluate one task for num_episodes. Returns a result dict."""
    controller.set_task(task_name)
    successes = 0
    results = []

    for ep in range(num_episodes):
        t0 = time.time()
        try:
            controller.reset()
        except Exception as e:
            logger.warning(f"{task_name} ep{ep}: reset failed: {e}")
            results.append({"episode": ep, "success": False, "reason": "reset_failed"})
            continue

        policy.reset()

        while not controller.is_episode_done():
            obs = controller.get_observation()
            if obs is None:
                break
            policy.update_obs(obs)
            action = policy.get_action()
            controller.publish_action(action)

        ep_result = controller.get_episode_result()
        elapsed = time.time() - t0

        if ep_result["success"]:
            successes += 1

        results.append({**ep_result, "time": round(elapsed, 1)})
        status = "\033[92mSuccess\033[0m" if ep_result["success"] else "\033[91mFail\033[0m"
        logger.info(
            f"{task_name} ep{ep}: {status} ({ep_result['steps']} steps, {elapsed:.1f}s) "
            f"SR={successes}/{ep + 1} ({successes / (ep + 1) * 100:.0f}%)"
        )

    controller.stop()
    # Denominator is attempted episodes, so reset failures can't silently
    # inflate the success rate.
    return {
        "task": task_name,
        "success_rate": successes / max(num_episodes, 1),
        "successes": successes,
        "total": num_episodes,
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-set", default=None,
                        help="TASK_SET_REGISTRY key, e.g. target50 / atomic_seen / "
                             "composite_unseen / all_tasks. Defaults to config.yaml.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Explicit task names; overrides --task-set.")
    parser.add_argument("--split", default=None,
                        choices=["pretrain", "target", "all"],
                        help="Evaluation split. Defaults to config.yaml.")
    parser.add_argument("--num-episodes", type=int, default=50,
                        help="Episodes per task (upstream protocol uses 50).")
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--list-task-sets", action="store_true",
                        help="Print available task sets with their sizes and exit.")

    # Policy: either local (FalconPolicy) or remote (HTTPClientPolicy)
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--policy-url", help="PolicyServer URL, e.g. http://localhost:24851")
    policy_group.add_argument("--config-path", help="ss-inference config path (local mode)")
    parser.add_argument("--checkpoint", help="Model checkpoint (required for local mode)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from hardware import get_config
    controller_config = get_config("robocasa365")

    if args.list_task_sets:
        from hardware.robocasa365.controller import _ensure_robocasa365
        _ensure_robocasa365(
            controller_config.get("robocasa365_dir"),
            controller_config.get("robosuite_dir"),
        )
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        for name, tasks in TASK_SET_REGISTRY.items():
            print(f"  {name:32s} {len(tasks):4d} tasks")
        return

    if not (args.policy_url or args.config_path):
        parser.error("one of --policy-url or --config-path is required")

    if args.split:
        controller_config["split"] = args.split
    split = controller_config.get("split", "target")

    # Resolve the task list before loading the policy — a bad task-set name
    # should fail in seconds, not after a multi-minute checkpoint load.
    if args.tasks:
        tasks = list(args.tasks)
        task_set_name = "custom"
    else:
        task_set_name = args.task_set or controller_config.get("task_set", "target50")
        tasks = resolve_task_set(
            task_set_name,
            controller_config.get("robocasa365_dir"),
            controller_config.get("robosuite_dir"),
        )

    # Build policy
    if args.policy_url:
        from policy.policies.http_client.policy import HTTPClientPolicy
        from urllib.parse import urlparse
        parsed = urlparse(args.policy_url)
        policy = HTTPClientPolicy(host=parsed.hostname, port=parsed.port or 24851)
        logger.info(f"Using remote policy: {args.policy_url}")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required with --config-path")
        from policy.policies.falcon.policy import FalconPolicy
        logger.info(f"Loading local policy: {args.config_path}")
        policy = FalconPolicy(
            config_path=args.config_path,
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    logger.info("Policy ready")

    controller = RoboCasa365Controller(controller_config)

    os.makedirs(args.output, exist_ok=True)
    all_results = {}

    logger.info(
        f"Evaluating task_set={task_set_name} split={split}: "
        f"{len(tasks)} tasks x {args.num_episodes} episodes"
    )

    def save(path, payload):
        with open(os.path.join(args.output, path), "w") as f:
            json.dump(payload, f, indent=2)

    for task in tasks:
        logger.info(f"\n{'=' * 50}\nTask: {task}\n{'=' * 50}")

        result = evaluate_task(policy, controller, task, args.num_episodes)
        all_results[task] = result

        sr = result["success_rate"]
        logger.info(f">>> {task}: SR = {result['successes']}/{result['total']} ({sr * 100:.1f}%)")

        # Incremental summary, so a crashed run still leaves usable numbers.
        summary = {
            t: {"success_rate": r["success_rate"], "successes": r["successes"],
                "total": r["total"]}
            for t, r in all_results.items()
        }
        save("results.json", {
            "task_set": task_set_name,
            "split": split,
            "avg_success_rate": float(np.mean([r["success_rate"] for r in all_results.values()])),
            "tasks": summary,
        })

    avg_sr = float(np.mean([r["success_rate"] for r in all_results.values()]))
    by_subset = summarize_by_subset(
        all_results,
        controller_config.get("robocasa365_dir"),
        controller_config.get("robosuite_dir"),
    )

    logger.info(f"\n{'=' * 50}")
    logger.info(f"FINAL: task_set={task_set_name} split={split} Avg SR = {avg_sr * 100:.1f}%")
    for subset, s in by_subset.items():
        logger.info(f"  {subset:20s} {s['success_rate'] * 100:5.1f}%  "
                    f"{s['successes']:3d}/{s['total']:3d} episodes  "
                    f"({s['num_tasks']} tasks)")
    for subset, s in by_subset.items():
        logger.info(f"[{subset}]")
        for task in s["tasks"]:
            result = all_results[task]
            logger.info(f"  {task:36s} {result['successes']:2d}/{result['total']:2d} "
                        f"({result['success_rate'] * 100:.0f}%)")
    logger.info(f"{'=' * 50}")

    save("results_full.json", {
        "task_set": task_set_name, "split": split,
        "avg_success_rate": avg_sr, "by_subset": by_subset,
        "tasks": all_results,
    })
    logger.info(f"Results saved to {args.output}/")


if __name__ == "__main__":
    main()
