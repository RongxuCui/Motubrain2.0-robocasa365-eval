# RoboCasa365 Evaluation

A self-contained bundle for evaluating one embodied policy — a four-step memory
student sampled with CMSolver, with depth-anything-3 geometry conditioning and
sparse visual memory — on the [RoboCasa](https://robocasa.ai) 365-task benchmark,
on a local multi-GPU machine. Pure PyTorch, no proprietary kernel packages.

`run_eval.sh` starts one PolicyServer (env `rc365-policy`, serves actions over
HTTP) and one simulation worker (env `rc365-sim`, runs MuJoCo) per GPU and hands
them a shared task queue, merging their per-worker `results.json` at the end. The
two environments cannot be merged: robocasa asserts `numpy == 2.2.5` and
`mujoco == 3.3.1` at import time, while the policy stack needs a different numpy.

## 1. Install the two environments

Each yml installs its pinned pip list (`requirements-policy.txt` /
`requirements-sim.txt`), so the only open choice is torch's CUDA variant: the
pinned `torch==2.9.1` installs its default build, while the reference run used the
CUDA 13.0 build (2.9 and 2.12 were both exercised end to end).

```bash
conda env create -f environment-policy.yml                       # rc365-policy, py3.13
conda activate rc365-policy
# only if you need a different CUDA build than the default wheel's
pip install --force-reinstall torch==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu130

conda env create -f environment-sim.yml                          # rc365-sim, py3.11
cd /path/to/sim                                                  # this is $SIM_ROOT
git clone https://github.com/robocasa/robocasa.git
git -C robocasa checkout f970a0aac4675a7e97ed5771cade4ade7044b2af          # 1.0.1
git clone https://github.com/ARISE-Initiative/robosuite robosuite-master
git -C robosuite-master checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c  # 1.5.2
conda activate rc365-sim
pip install --no-deps -e ./robosuite-master
pip install --no-deps -e ./robocasa

# headless rendering, or every task is skipped with
# 'NoneType' object has no attribute 'eglQueryString'
sudo apt-get install -y libegl1 libgl1 libglvnd0 libglib2.0-0
```

`--no-deps` keeps the source installs from replacing the pinned numpy or mujoco.
Pin by commit, not tag: robocasa reports `1.0.1` but its newest tag is `v1.0`, 19
commits behind, and robosuite must be master (1.5.2) because 1.5.1 lacks
`utils.mjcf_utils.get_elements`. Both checkouts must sit under the one directory
passed as `SIM_ROOT` in step 3, with the RoboCasa assets downloaded as that project
documents, roughly 23 GB; a checkout with code but an empty
`models/assets/objects` fails late, during object sampling. Depth-Anything-3 ships
as the submodule `third_party/Depth-Anything-3`: clone this bundle with
`--recursive`, or run `git submodule update --init`.

## 2. Download the four weight sets

The checkpoint holds only the DiT; the VAE, text encoder and depth model are
frozen pretrained models loaded separately. `hf` ships with the pinned
`huggingface-hub==0.36.0`, so run these from `rc365-policy`, and check the sizes
afterwards: a different revision changes results with no error at all.

| What | Repository | Verify |
| --- | --- | --- |
| Policy checkpoint | `Motubrain/Motubrain2.0-robocasa365` | `step-2000.safetensors` 39047282448 B; `config.yaml` 1794 B |
| Text encoder | `google/t5gemma-9b-9b-prefixlm-it` — gated, Gemma license | 9 shards totalling 40666951336 B; index `total_size` 40668637184 |
| VAE | `Wan-AI/Wan2.2-TI2V-5B`, file `Wan2.2_VAE.pth` | 2818839170 B |
| Depth model | `depth-anything/DA3-LARGE` — CC BY-NC 4.0 | `model.safetensors` 1643843860 B |

```bash
hf auth login     # required for the gated T5Gemma repo

hf download Motubrain/Motubrain2.0-robocasa365   --local-dir /path/to/motubrain
hf download google/t5gemma-9b-9b-prefixlm-it     --local-dir /path/to/t5gemma
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir /path/to/wan
hf download depth-anything/DA3-LARGE             --local-dir /path/to/da3
```

Add `--include "weights/*"` to the first command to fetch only the checkpoint
and its config. Three cautions:

- Note the `-it` suffix and `prefixlm`, **not** `ul2`: all four T5Gemma 9B-9B
  variants ship nine shards of identical total size, so size cannot tell them apart.
- `Wan2.2-TI2V-5B` is the only Wan2.2 repository publishing `Wan2.2_VAE.pth`; the
  other releases ship `Wan2.1_VAE.pth`.
- Use plain `DA3-LARGE`, **not** `DA3-LARGE-1.1`: upstream deprecates the original
  in favour of the retrain, but this policy was trained against the original.

**Preserve the `weights/` layout.** The policy reads its architecture from a
`config.yaml` it finds by walking up to five parent directories from the
checkpoint, so keep the tree as published: `<local-dir>/weights/config.yaml`
beside `<local-dir>/weights/ckpts/step-2000.safetensors`, the path to pass to
`--checkpoint`. Otherwise startup fails with `No config.yaml found in parent
directories of ...`.

Local weights are read in place and need no scratch space; only remote
(`scheme://`) artefacts are copied into `LOCAL_CACHE_DIR` (`/tmp/ckpt_cache`).
`SOMA_ARTIFACT_CACHE=all` also copies local paths to local disk (~41 GB for the
encoders); `0` disables caching entirely.

## 3. Start an evaluation

```bash
# required: the run aborts immediately if any of these is unset
export SIM_ROOT=/path/to/sim                      # holds robocasa/, robosuite-master/
export POLICY_CONDA_BASE="$HOME/miniconda3"       # has etc/profile.d/conda.sh
export SOMA_TEXT_ENCODER_PATH=/path/to/t5gemma
export SOMA_VAE_PATH=/path/to/wan/Wan2.2_VAE.pth
export DA3_CONFIG_PATH=/path/to/da3/config.json
export DA3_CHECKPOINT_PATH=/path/to/da3/model.safetensors
export DEPTH_ANYTHING_3_ROOT="$PWD/third_party/Depth-Anything-3"
# set explicitly: the defaults name envs that environment-*.yml does not create
export POLICY_ENV="$POLICY_CONDA_BASE/envs/rc365-policy"
export ROBOCASA365_ENV="$POLICY_CONDA_BASE/envs/rc365-sim"

cd RealWorldInference/hardware/robocasa365
bash run_eval.sh \
    --config-path  ../../../ss-soma/configs/inference/robocasa365.py \
    --checkpoint   /path/to/motubrain/weights/ckpts/step-2000.safetensors \
    --task-set     atomic_seen \
    --num-gpus     8 \
    --num-episodes 50
```

Also optional: `ROBOCASA365_CONDA_BASE` (defaults to `POLICY_CONDA_BASE`),
`ROBOCASA365_DIR` and `ROBOCASA365_ROBOSUITE_DIR` (the two `SIM_ROOT`
subdirectory names), `ROBOCASA365_POLICY_DEPS`, `LOCAL_CACHE_DIR`,
`SOMA_ARTIFACT_CACHE`, `POLICY_DEPS_INDEX_URL`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--config-path` | required | Policy config (the cm4 config above) |
| `--checkpoint` | required | Policy `.safetensors` |
| `--task-set` | config default | A key of robocasa's `TASK_SET_REGISTRY` |
| `--tasks` | unset | Explicit comma-separated task names; overrides `--task-set` |
| `--split` | config default | `pretrain`, `target` or `all` |
| `--num-episodes` | 50 | Episodes per task |
| `--num-gpus` | 8 | PolicyServer/worker pairs, one full GPU each |
| `--base-port` | 24851 | First PolicyServer port |
| `--policy-config` | generated | Supply your own policy YAML |
| `--save-video` | off | Write execution videos |
| `--list-task-sets` | — | Print the available task sets and exit |

Common task sets: `atomic_seen`, `composite_seen`, `composite_unseen`,
`target50`, `all_tasks` — one per invocation, so the full benchmark is several
runs. For scale, 18 tasks × 50 episodes on 8 GPUs takes roughly four hours.

A bad path or a missing library only surfaces once the model has loaded, so
smoke-test with `--tasks CloseFridge --num-gpus 1 --num-episodes 2` first: that
exercises model load, DA3, renderer, executor and merge in a few minutes. Servers
must answer `/health` before any worker starts (30 minute budget, a few minutes
per GPU), so an empty `logs/worker_gpu*.log` early on is normal while
`logs/server_gpu0.log` works towards `Policy ready.`

## 4. Read the results

Results land in `RealWorldInference/eval_results/robocasa365/<timestamp>/`:

| Path | Contents |
| --- | --- |
| `results_merged.json` | Per-task and per-subset success rates |
| `log.txt` | Controller log and the final summary |
| `gpu<N>/results.json` | One worker's tasks |
| `logs/server_gpu<N>.log` | PolicyServer startup and model loading |
| `logs/worker_gpu<N>.log` | Simulator and episode progress |
| `gpu<N>/exec_videos/` | Present with `--save-video` |

A run is complete only when `results_merged.json` has `"complete": true`. The
launcher's exit code is not a reliable signal: its cleanup step can report
failure after a run has finished normally.

Scene sampling is seeded (`seed: 195`, once per task), but **diffusion noise is
unseeded**, so success rates move between identical runs — several percentage
points on an 18-task set is normal. Treat one run as an estimate and compare
configurations over repeated runs, not single numbers. (An `inference_seed` in a
`--policy-config` YAML fixes the draw, but then samples one noise vector rather
than averaging over the distribution.)

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `SIM_ROOT: set SIM_ROOT to ...` | A required variable is unset; see step 3 |
| `No config.yaml found in parent directories of ...` | Checkpoint moved out of its `weights/` layout; see step 2 |
| `missing policy interpreter: .../bin/python` | `POLICY_ENV` does not point at the `rc365-policy` env |
| Import error for `robosuite.utils.mjcf_utils.get_elements` | robosuite 1.5.1 instead of master (1.5.2) |
| Worker dies at object sampling | RoboCasa assets not downloaded |
| `libEGL.so.1: cannot open shared object file` | OpenGL system packages missing |
| PolicyServer never becomes ready | Read `logs/server_gpu0.log`; usually a missing weight path |
| `No space left on device` while starting | Only with `SOMA_ARTIFACT_CACHE=all` or remote weights; set `LOCAL_CACHE_DIR` |
| `GPU N: timeout after 30min!` | Model load exceeded the budget — slow storage or a remote download |
