# HANDOFF — GraspNet training project context

Quick-start context for continuing this work on another machine (or in a new
session). Read this first, then `CLAUDE.md`, `migration/README.md`,
`docs/setup_env.md`, `docs/dataset_download.md`, and `ToDo.md`.

## 1. Goal & hard constraints

- Train the **GraspNet baseline** (paper-faithful) in this container; export the
  trained weights to another PC for real/Sim testing. **This container does
  training only** (no inference/eval/deploy).
- **Top constraint:** nothing here may harm the host or other containers. The
  host has **no resource isolation** (cgroup unlimited, all GPUs visible) and a
  **shared disk** (`/workspace` is a host bind-mount). So we self-limit: pin one
  GPU, cap CPU threads / DataLoader workers, enforce a disk floor + live monitor,
  and gate every resource-consuming step behind a report + user approval.
- **Guarded operations** (CLAUDE.md §18, enforced by `.claude/hooks/
  pre-guarded-ops.sh`): `pkill -f unicorn`/gunicorn and setting nginx
  `worker_processes` to a manual count require explicit approval even in auto
  mode.

## 2. Conventions in effect

`CLAUDE.md` governs: MIT comment style, 80-col, Ruff on tracked Python, and the
**§4 workflow** — every task gets a `ToDo.md` entry + `gh` issue + a branch +
a PR. serena MCP is configured (`.mcp.json`). `.claude/` has hooks
(pre-write-guard, secret-scan, env-guard, post-write-lint/debug-remind,
guarded-ops, Stop check).

## 3. Environment (reproducible)

- Dedicated conda env **`graspnet`** (Python 3.8) — does NOT touch `base`/
  `peallab`. `torch==1.13.1+cu117` (ships `sm_86` kernels), `numpy`, `scipy`,
  `tqdm`, `tensorboard`, `six`. Open3D/graspnetAPI intentionally NOT installed.
- CUDA toolkit **11.8** at `/usr/local/cuda-11.8`; GPUs are 3× RTX 3090 (sm_86).
- Fork **`H-Freax/GraspNet_Pointnet2_PyTorch1.13.1`** cloned at `graspnet/`
  (gitignored). `pointnet2` + `knn` CUDA extensions built with CUDA 11.8,
  `TORCH_CUDA_ARCH_LIST=8.6`, `MAX_JOBS=4`. Full recipe: `docs/setup_env.md`.
- **Vendored patches** in the gitignored fork (documented in setup_env.md):
  `graspnet/dataset/graspnet_dataset.py` (define missing `BASE_DIR`; add
  `scene_ids` arg), `graspnet/train.py` (adapted entrypoint with import
  bootstrap + `--scenes/--skip_eval/--max_iters/--dry_run` + paper LR schedule),
  `graspnet/dataset/generate_tolerance_label.py` (flat-import bootstrap).

## 4. Data (status: integrity-verified)

- Dataset root: `/workspace/data/graspnet/` (gitignored, NOT migrated — ~122 GB).
- **Acquisition switched to the public HF mirror `saic3d/graspnet`** (verified
  byte-identical to the official zips via sha256 vs HF LFS oids). Use
  `scripts/download_dataset.sh` (disk gate + sha256 check + extract). No
  streaming-capable distribution exists anywhere (all archive-only); the
  loader reads per-scene files from disk.
- Zips present and **all CRC-OK**: `train_1/2/4.zip`, `grasp_label.zip`,
  `collision_label.zip`; `train_3` fetched from the HF mirror (scenes
  0060-0089). No `test_*` zips yet (fetch via the script when full training
  starts).
- Extracted so far: `scenes/scene_0000..0029` (realsense), `grasp_label/`,
  `collision_label/`.
- **tolerance labels** generated: 88 `.npy` at
  `graspnet/dataset/tolerance/` (loader reads them via `BASE_DIR`).
- Loader assumes scenes 0000-0099 for `split='train'`; we have 0000-0029, so the
  minimal validation restricts to `--scenes 0-29` and `--skip_eval` (no test
  data).

## 5. Target gripper

**DH Robotics PGC-140-50**, parallel, ~50 mm stroke. ≤ GraspNet's 75 mm max
grasp width → the **standard baseline is sufficient** (no gripper-specific
retraining); filter predicted grasps by width at deploy time on the other PC.

## 6. Training config (decided, paper-faithful)

Single GPU, `batch_size=4`, Adam `lr=0.001` decayed at epochs 60/100 (the CVPR
2020 paper trained on one RTX 2080, batch 4). Multi-GPU was smoke-tested
(DataParallel and DDP both work) but is NOT used since the paper is single-GPU.

## 7. Our scripts (tracked)

- `scripts/preflight.py` — read-only GPU/disk/RAM gate; `--allow-busy` warns
  instead of aborting on a shared GPU.
- `scripts/train_graspnet.sh` — single-GPU launcher; modes
  `scratch`/`finetune-official`/`finetune-custom`; flags
  `--gpu/--scenes/--skip_eval/--max_iters/--allow_busy`; runs preflight first,
  starts a background disk monitor that kills training if free disk < floor.
- `claude_test/graspnet_ddp_smoke.py` — synthetic multi-GPU smoke test.

## 8. Work done (issues / PRs)

| Task | What | PR |
|------|------|----|
| 1 | Repo bootstrap (CLAUDE rules, .mcp.json) | merged |
| 2 | Plan review (PLAN-graspnet.md) | #3 merged |
| 3 | Env build + multi-GPU smoke test | #5 |
| 4 | Dataset doc + single-GPU wrappers | #7 |
| 5 | Guarded-ops rule + hook | #9 |
| 6 | Minimal-validation prep | #11 |
| 7 | Migration kit + integrity check | #13 |

PRs #5→#7→#9→#11→#13 are **stacked** (each based on the previous) and not yet
merged; merge in order, or retarget to `main`.

## 9. Pending / next steps

- **Minimal validation run** is GATED on user GPU permission (all 3 GPUs are
  currently shared by other users). When a GPU is free/chosen:
  ```
  scripts/train_graspnet.sh --gpu <N> --mode scratch --camera realsense \
    --batch_size 4 --num_workers 2 --max_epoch 1 \
    --scenes 0-29 --skip_eval --max_iters 50 --allow_busy
  ```
  Success = data loads, 1 step runs (no other-user OOM), `runs/.../checkpoint.tar`
  saved, disk floor held.
- **Full training later** needs: download `train_3.zip` (scenes 0060-0089),
  restore the full train scene range, and (for eval) the `test_seen` scenes
  100-129; then run without `--scenes/--skip_eval/--max_iters`.

## 10. Migration

See `migration/README.md`. Host `docker commit`/`save` for the env image (the
conda env + built extensions + CUDA are in the image); bundle the host
bind-mount artifacts (`/workspace/tools` = repo + fork source + tolerance); the
~122 GB dataset is excluded (re-acquire on target per `docs/dataset_download.md`).
`migration/environment.yml` + `requirements-lock.txt` are the env spec fallback.
