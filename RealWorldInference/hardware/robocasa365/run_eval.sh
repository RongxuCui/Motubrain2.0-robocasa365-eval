#!/bin/bash
# ============================================================================
# RoboCasa365 Evaluation — client-server, multi-GPU parallel.
#
# Each GPU runs: PolicyServer (policy env) + Worker (robocasa env).
# Workers share a flock task queue; each queue line is one task name and the
# worker runs --num-episodes episodes on it.
#
# Task selection follows the upstream 365 protocol: name a task set from
# TASK_SET_REGISTRY plus a split, rather than hardcoding task names.
#
# Usage:
#   bash run_eval.sh \
#       --config-path /path/to/ss-inference/configs/embodied/robocasa365.py \
#       --checkpoint /path/to/step-50000-ema.safetensors \
#       --task-set target50 \
#       --split target \
#       --num-gpus 8 \
#       --num-episodes 50
#
#   # list available task sets and exit
#   bash run_eval.sh --list-task-sets
# ============================================================================

set -eo pipefail
ulimit -Sn 65536
echo "[eval] nofile soft limit: $(ulimit -Sn)"

CONFIG_PATH=""
CHECKPOINT=""
NUM_EPISODES=50
NUM_GPUS=8
BASE_PORT=24851
POLICY_CONFIG=""
TASK_SET=""
TASKS=""
SPLIT=""
SAVE_VIDEO=0
LIST_TASK_SETS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config-path)     CONFIG_PATH="$2";    shift 2 ;;
        --checkpoint)      CHECKPOINT="$2";     shift 2 ;;
        --num-episodes)    NUM_EPISODES="$2";   shift 2 ;;
        --num-gpus)        NUM_GPUS="$2";       shift 2 ;;
        --base-port)       BASE_PORT="$2";      shift 2 ;;
        --policy-config)   POLICY_CONFIG="$2";  shift 2 ;;
        --task-set)        TASK_SET="$2";       shift 2 ;;
        --tasks)           TASKS="$2";          shift 2 ;;
        --split)           SPLIT="$2";          shift 2 ;;
        --save-video)      SAVE_VIDEO=1;        shift ;;
        --list-task-sets)  LIST_TASK_SETS=1;    shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RWAI_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Default straight to the shared sim tree rather than probing the repo-relative
# ../sim first (as the other adapters do): only the provisioned checkout has the
# ~23G of downloaded kitchen assets, and a tree with code but an empty
# models/assets/objects fails at object sampling instead of at import.
: "${SIM_ROOT:?set SIM_ROOT to the simulator root holding robocasa/ and robosuite-master/}"
SIM_ROOT="$(cd "${SIM_ROOT}" && pwd)"
# RoboCasa365 = the v1.0+ checkout, distinct from robocasa-50 (v0.2).
ROBOCASA365_DIR="${ROBOCASA365_DIR:-${SIM_ROOT}/robocasa}"
ROBOCASA365_DIR="$(cd "${ROBOCASA365_DIR}" && pwd)"
# robosuite-master, not robosuite: robocasa 1.0.1 imports
# robosuite.utils.mjcf_utils.get_elements, which only exists on robosuite master
# (1.5.2). The 1.5.1 checkouts in this workspace lack it and fail at import.
ROBOCASA365_ROBOSUITE_DIR="${ROBOCASA365_ROBOSUITE_DIR:-${SIM_ROOT}/robosuite-master}"
ROBOCASA365_ROBOSUITE_DIR="$(cd "${ROBOCASA365_ROBOSUITE_DIR}" && pwd)"
# Conda install that provides the PolicyServer environment.
# Optional PyPI index for the one extra wheel the policy env needs
# (POLICY_DEPS_INDEX_URL). Unset uses pip's configured default index.
: "${POLICY_CONDA_BASE:?set POLICY_CONDA_BASE to the conda install holding the policy env}"
POLICY_CONDA_SH="${POLICY_CONDA_BASE}/etc/profile.d/conda.sh"
# The worker's conda base defaults to the policy one. It is only used to source
# conda.sh before activating ROBOCASA365_ENV by absolute path, so sharing one
# install is enough; set ROBOCASA365_CONDA_BASE to separate them.
ROBOCASA365_CONDA_BASE="${ROBOCASA365_CONDA_BASE:-${POLICY_CONDA_BASE}}"
ROBOCASA365_CONDA_SH="${ROBOCASA365_CONDA_BASE}/etc/profile.d/conda.sh"

# Conda envs. The sim worker's env may live at an absolute path outside any
# conda base's envs/ dir, so set ROBOCASA365_ENV explicitly in that case.
POLICY_ENV="${POLICY_ENV:-${POLICY_CONDA_BASE}/envs/policy}"
ROBOCASA365_ENV="${ROBOCASA365_ENV:-${ROBOCASA365_CONDA_BASE}/envs/robocasa365}"
ROBOCASA365_POLICY_DEPS="${ROBOCASA365_POLICY_DEPS:-/tmp/robocasa365-policy-deps}"

PYTHONPATH_WORKER="${RWAI_DIR}:${ROBOCASA365_DIR}:${ROBOCASA365_ROBOSUITE_DIR}:${PYTHONPATH:-}"

run_in_sim_env() {
    # Run python in the robocasa env with the sim trees importable.
    # Body is a subshell: `conda activate` and `cd` must not leak into the
    # caller, which would break the PolicyServer launch below (it needs the
    # policy env) and every later relative path.
    (
        source "${ROBOCASA365_CONDA_SH}"
        conda activate "${ROBOCASA365_ENV}"
        cd "${RWAI_DIR}"
        ROBOCASA365_DIR="${ROBOCASA365_DIR}" \
        ROBOCASA365_ROBOSUITE_DIR="${ROBOCASA365_ROBOSUITE_DIR}" \
        PYTHONPATH="${PYTHONPATH_WORKER}" \
        BENCH_DIR="${BENCH_DIR:-}" \
        NUM_EPISODES="${NUM_EPISODES:-}" \
        TASK_SET_LABEL="${TASK_SET_LABEL:-}" \
        python "$@"
    )
}

if [[ "${LIST_TASK_SETS}" -eq 1 ]]; then
    run_in_sim_env -m hardware.robocasa365.evaluate --list-task-sets
    exit 0
fi

[[ -z "$CONFIG_PATH" ]] && { echo "ERROR: --config-path required"; exit 1; }
[[ -z "$CHECKPOINT" ]] && { echo "ERROR: --checkpoint required"; exit 1; }

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_DIR="${RWAI_DIR}/eval_results/robocasa365/${TIMESTAMP}"
LOG_DIR="${BENCH_DIR}/logs"
mkdir -p "${LOG_DIR}"

QUEUE_FILE="${BENCH_DIR}/task_queue.txt"
LOCK_FILE="${BENCH_DIR}/queue.lock"
SAVE_VIDEO_ARGS=()
[[ "${SAVE_VIDEO}" -eq 1 ]] && SAVE_VIDEO_ARGS+=(--save-video)
SPLIT_ARGS=()
[[ -n "${SPLIT}" ]] && SPLIT_ARGS+=(--split "${SPLIT}")

# ─── Build task queue ───
# Explicit --tasks wins; otherwise expand --task-set through TASK_SET_REGISTRY
# so the task list always matches the upstream benchmark definition.
if [[ -n "${TASKS}" ]]; then
    echo "${TASKS}" | tr ',' '\n' | tr ' ' '\n' | grep -v '^$' > "${QUEUE_FILE}"
    TASK_SET_LABEL="custom"
else
    TASK_SET_LABEL="${TASK_SET:-<config default>}"
    TASK_SET_ARGS=()
    [[ -n "${TASK_SET}" ]] && TASK_SET_ARGS+=("${TASK_SET}")
    # Importing robocasa/robosuite prints warnings to stdout (e.g. "mimicgen
    # environments not imported ..."). Tag each real task name with a TASK<TAB>
    # prefix and keep only tagged lines below, so nothing a library prints to
    # stdout can leak into the queue as a bogus task name.
    QUEUE_RAW="${QUEUE_FILE}.raw"
    run_in_sim_env - "${TASK_SET_ARGS[@]}" > "${QUEUE_RAW}" << 'PYEOF'
import sys
from hardware import get_config
from hardware.robocasa365.controller import resolve_task_set

cfg = get_config("robocasa365")
task_set = sys.argv[1] if len(sys.argv) > 1 else cfg.get("task_set", "target50")
for task in resolve_task_set(
    task_set, cfg.get("robocasa365_dir"), cfg.get("robosuite_dir")
):
    print(f"TASK\t{task}")
PYEOF
    grep -P '^TASK\t' "${QUEUE_RAW}" | cut -f2- > "${QUEUE_FILE}"
    rm -f "${QUEUE_RAW}"
fi

TOTAL_TASKS="$(wc -l < "${QUEUE_FILE}")"
if [[ "${TOTAL_TASKS}" -eq 0 ]]; then
    echo "ERROR: task queue is empty."
    exit 1
fi
cp "${QUEUE_FILE}" "${BENCH_DIR}/tasks_expected.txt"

# ─── Write policy config ───
if [[ -z "${POLICY_CONFIG}" ]]; then
    POLICY_CONFIG="${BENCH_DIR}/policy_config.yaml"
    cat > "${POLICY_CONFIG}" << EOF
policy:
  type: falcon
  policy_id: falcon_robocasa365
  config_path: "${CONFIG_PATH}"
  checkpoint_path: "${CHECKPOINT}"
server:
  host: "0.0.0.0"
  port: ${BASE_PORT}
EOF
fi

exec > >(tee -a "${BENCH_DIR}/log.txt") 2>&1

echo "============================================================"
echo "  RoboCasa365 Evaluation (${NUM_GPUS} GPUs)"
echo "  Robocasa:   ${ROBOCASA365_DIR}"
echo "  Config:     ${CONFIG_PATH}"
echo "  Checkpoint: $(basename "${CHECKPOINT}")"
echo "  Task set:   ${TASK_SET_LABEL} (${TOTAL_TASKS} tasks)"
echo "  Split:      ${SPLIT:-<config default>}"
echo "  Episodes:   ${NUM_EPISODES}"
echo "  Save video: ${SAVE_VIDEO_ARGS[*]:-(disabled)}"
echo "  Output:     ${BENCH_DIR}"
echo "  $(date)"
echo "============================================================"

# ─── Cleanup trap ───
SERVER_PIDS=()
WORKER_PIDS=()

cleanup() {
    echo "[eval] Cleaning up..."
    for pid in "${WORKER_PIDS[@]}"; do kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

prepare_policy_dependencies() {
    # The policy stack is pure PyTorch; the only extra wheel is Depth-Anything-3's
    # `addict`, installed into ROBOCASA365_POLICY_DEPS so the conda env stays clean.
    local policy_python="${POLICY_ENV}/bin/python"
    local da3_root="${DEPTH_ANYTHING_3_ROOT:-}"
    local da3_src=""

    [[ -x "${policy_python}" ]] || {
        echo "ERROR: missing policy interpreter: ${policy_python}" >&2
        return 1
    }
    if [[ -n "${da3_root}" ]]; then
        da3_src="${da3_root}/src"
        [[ -r "${da3_src}/depth_anything_3/model/dinov2/dinov2.py" ]] || {
            echo "ERROR: Depth-Anything-3 source is missing under ${da3_src}" >&2
            return 1
        }
    fi
    mkdir -p "${ROBOCASA365_POLICY_DEPS}"
    if [[ -n "${da3_src}" ]] && ! \
        PYTHONNOUSERSITE=1 PYTHONPATH="${ROBOCASA365_POLICY_DEPS}" \
        "${policy_python}" -c 'import addict' >/dev/null 2>&1; then
        PYTHONNOUSERSITE=1 "${policy_python}" -m pip install \
            ${POLICY_DEPS_INDEX_URL:+--index-url "${POLICY_DEPS_INDEX_URL}"} \
            --disable-pip-version-check --no-deps \
            --target "${ROBOCASA365_POLICY_DEPS}" "addict==2.4.0"
    fi
}

prepare_policy_dependencies

# ─── Start PolicyServers (policy env) ───
echo "[eval] Starting ${NUM_GPUS} PolicyServers..."

WORKSPACE_DIR="$(cd "${RWAI_DIR}/.." && pwd)"
PYTHONPATH_SERVER="${ROBOCASA365_POLICY_DEPS}"
POLICY_IMPORT_PROBE='from soma.inference.runtime.motubrain import build_model'
if [[ -n "${DEPTH_ANYTHING_3_ROOT:-}" ]]; then
    PYTHONPATH_SERVER="${PYTHONPATH_SERVER}:${DEPTH_ANYTHING_3_ROOT}/src"
    POLICY_IMPORT_PROBE+='; from depth_anything_3.model.dinov2 import dinov2'
fi
PYTHONPATH_SERVER="${PYTHONPATH_SERVER}:${RWAI_DIR}:${WORKSPACE_DIR}/ss-soma:${WORKSPACE_DIR}/ss-falcon:${WORKSPACE_DIR}/ss-morph:${WORKSPACE_DIR}/ss-modelzoo:${PYTHONPATH:-}"
PYTHONNOUSERSITE=1 PYTHONPATH="${PYTHONPATH_SERVER}" \
    "${POLICY_ENV}/bin/python" -c "${POLICY_IMPORT_PROBE}"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="${LOG_DIR}/server_gpu${i}.log"
    GPU_POLICY_CFG="${BENCH_DIR}/policy_config_gpu${i}.yaml"
    sed "s/port: ${BASE_PORT}/port: ${PORT}/" "${POLICY_CONFIG}" > "${GPU_POLICY_CFG}"
    (
        source "${POLICY_CONDA_SH}"
        conda activate "${POLICY_ENV}"
        cd "${RWAI_DIR}"
        PYTHONNOUSERSITE=1 PYTHONPATH="${PYTHONPATH_SERVER}" \
        ROBOCASA365_PIPELINE_SERVER_LOG_PATH="${LOG_DIR}/pipeline_server.log" \
        CUDA_VISIBLE_DEVICES="${i}" python -m policy.policy_http_server \
            --config "${GPU_POLICY_CFG}" \
        > "${LOG}" 2>&1
    ) &
    SERVER_PIDS+=($!)
    echo "  GPU ${i}: PolicyServer port=${PORT} (PID ${SERVER_PIDS[-1]})"
done

# ─── Wait for servers to be ready ───
echo "[eval] Waiting for PolicyServers..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    for attempt in $(seq 1 360); do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "  GPU ${i}: ready ($((attempt * 5))s)"
            break
        fi
        if ! kill -0 "${SERVER_PIDS[$i]}" 2>/dev/null; then
            echo "  GPU ${i}: PolicyServer died!"
            tail -20 "${LOG_DIR}/server_gpu${i}.log"
            exit 1
        fi
        sleep 5
    done
    if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "  GPU ${i}: timeout after 30min!"
        tail -20 "${LOG_DIR}/server_gpu${i}.log"
        exit 1
    fi
done
echo "[eval] All PolicyServers ready."

# ─── Start Workers (robocasa env) ───
echo "[eval] Starting ${NUM_GPUS} Workers..."

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG="${LOG_DIR}/worker_gpu${i}.log"
    (
        source "${ROBOCASA365_CONDA_SH}"
        conda activate "${ROBOCASA365_ENV}"
        cd "${RWAI_DIR}"
        # CUDA_VISIBLE_DEVICES remaps physical GPU i to logical device 0, which
        # is the only valid EGL device ordinal inside this worker process.
        ROBOCASA365_DIR="${ROBOCASA365_DIR}" \
        ROBOCASA365_ROBOSUITE_DIR="${ROBOCASA365_ROBOSUITE_DIR}" \
        PYTHONPATH="${PYTHONPATH_WORKER}" \
        MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
        CUDA_VISIBLE_DEVICES="${i}" python -m hardware.robocasa365.worker \
            --policy-url "http://localhost:${PORT}" \
            --queue "${QUEUE_FILE}" \
            --lock "${LOCK_FILE}" \
            --num-episodes "${NUM_EPISODES}" \
            --output "${BENCH_DIR}/gpu${i}" \
            "${SPLIT_ARGS[@]}" \
            "${SAVE_VIDEO_ARGS[@]}" \
        > "${LOG}" 2>&1
    ) &
    WORKER_PIDS+=($!)
    echo "  GPU ${i}: Worker (PID ${WORKER_PIDS[-1]})"
done

# ─── Wait for all workers ───
echo "[eval] Running..."
BENCH_START=$(date +%s)
WORKER_FAILURES=0

for i in "${!WORKER_PIDS[@]}"; do
    if ! wait "${WORKER_PIDS[$i]}"; then
        echo "[eval] ERROR: Worker GPU ${i} exited unsuccessfully."
        WORKER_FAILURES=$((WORKER_FAILURES + 1))
    fi
done

BENCH_END=$(date +%s)
BENCH_DURATION=$((BENCH_END - BENCH_START))

# ─── Merge results ───
echo "[eval] Merging results..."

MERGE_STATUS=0
run_in_sim_env - << 'PYEOF' || MERGE_STATUS=$?
import json
import os
import sys

import numpy as np

bench_dir = os.environ["BENCH_DIR"]
expected_episodes = int(os.environ["NUM_EPISODES"])
with open(os.path.join(bench_dir, "tasks_expected.txt")) as f:
    expected_tasks = [line.strip() for line in f if line.strip()]

all_results = {}
for d in sorted(os.listdir(bench_dir)):
    rf = os.path.join(bench_dir, d, "results.json")
    if os.path.exists(rf):
        with open(rf) as f:
            all_results.update(json.load(f).get("tasks", {}))

if not all_results:
    print("No results found. Check logs.")
    raise SystemExit(1)

avg_sr = float(np.mean([r["success_rate"] for r in all_results.values()]))
missing_tasks = sorted(set(expected_tasks) - set(all_results))
unexpected_tasks = sorted(set(all_results) - set(expected_tasks))
wrong_episode_counts = {
    task: result.get("total")
    for task, result in all_results.items()
    if result.get("total") != expected_episodes
}
complete = (
    len(all_results) == len(expected_tasks)
    and not missing_tasks
    and not unexpected_tasks
    and not wrong_episode_counts
)
print(f"Task set: {os.environ.get('TASK_SET_LABEL')}   tasks: {len(all_results)}")
print(f"Avg SR: {avg_sr * 100:.1f}%")
print(f"Complete: {complete} ({len(all_results)}/{len(expected_tasks)} tasks)")
if missing_tasks:
    print(f"Missing tasks: {missing_tasks}")
if unexpected_tasks:
    print(f"Unexpected tasks: {unexpected_tasks}")
if wrong_episode_counts:
    print(f"Wrong episode counts: {wrong_episode_counts}")

from hardware import get_config
from hardware.robocasa365.controller import summarize_by_subset

cfg = get_config("robocasa365")
by_subset = summarize_by_subset(
    all_results, cfg.get("robocasa365_dir"), cfg.get("robosuite_dir")
)

print()
for subset, s in by_subset.items():
    print(f"  {subset:20s} {s['success_rate'] * 100:5.1f}%  "
          f"{s['successes']:3d}/{s['total']:3d} episodes  "
          f"({s['num_tasks']} tasks)")

print()
for subset, s in by_subset.items():
    print(f"[{subset}]")
    for task in s["tasks"]:
        r = all_results[task]
        print(f"  {task:36s} {r['successes']:2d}/{r['total']:2d} "
              f"({r['success_rate'] * 100:.0f}%)")

with open(os.path.join(bench_dir, "results_merged.json"), "w") as f:
    json.dump({"task_set": os.environ.get("TASK_SET_LABEL"),
               "avg_success_rate": avg_sr,
               "complete": complete,
               "expected_tasks": len(expected_tasks),
               "expected_episodes_per_task": expected_episodes,
               "missing_tasks": missing_tasks,
               "unexpected_tasks": unexpected_tasks,
               "wrong_episode_counts": wrong_episode_counts,
               "num_tasks": len(all_results),
               "by_subset": by_subset,
               "tasks": all_results}, f, indent=2)

if not complete:
    raise SystemExit(2)
PYEOF

if [[ "${WORKER_FAILURES}" -ne 0 || "${MERGE_STATUS}" -ne 0 ]]; then
    echo "[eval] ERROR: incomplete evaluation: worker_failures=${WORKER_FAILURES}, merge_status=${MERGE_STATUS}"
    exit 1
fi

echo "============================================================"
echo "  Done in $((BENCH_DURATION / 60))m$((BENCH_DURATION % 60))s"
echo "  Results: ${BENCH_DIR}"
echo "============================================================"
