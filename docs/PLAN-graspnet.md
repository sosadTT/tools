# Plan: GraspNet-baseline TRAINING-ONLY Environment

> Status: **awaiting administrator review/approval.** No GraspNet setup,
> download, build, or training has been executed. This document exists so the
> administrator can review the plan before any resource-consuming action.

## Context

Goal: clone `graspnet/graspnet-baseline` in this container, train / fine-tune
it, and export **only the trained weights** to another PC for real/Sim testing.
This container does **training only** (no inference, evaluation, or deployment).

**Top constraint (from the user):** nothing this container does may negatively
affect the host server or other containers. Natural resource use from training
is allowed **only after reporting to the user and getting authorization**.

Why this matters — **the environment has no resource isolation** (verified by
read-only inspection):

- cgroup `memory.max = max`, `cpu.max = max`, and both `CUDA_VISIBLE_DEVICES`
  and `NVIDIA_VISIBLE_DEVICES` are unset. The container can consume all 3 GPUs,
  all 503 GB RAM, and all 128 CPU cores. Protection depends entirely on us
  self-limiting.
- The container `/` (overlay) and `/workspace` are on the **same physical disk**
  (`nvme0n1p2`, a bind mount from host `/home/appeal-s1/...`). It is 80%+ used
  with roughly 320–355 GB free, and that free space **fluctuates** because the
  disk is shared. Filling it directly harms the host and other users. The full
  dataset is 152 GB, so it must **never** be downloaded in full here.

Environment facts: 3× RTX 3090 (24 GB, Ampere sm_86, currently idle), CUDA
toolkit / nvcc **11.8** at `/usr/local/cuda-11.8`, gcc 11.4 (within nvcc 11.8's
supported range), conda (`base` / `peallab`; `peallab` has no torch and must not
be touched).

## User decisions

- **Dataset:** form-gated at graspnet.net. We only **document the download
  procedure**; the user downloads and places the data. No automatic download.
- **Training scope:** the wrappers must support all three modes (the user will
  iterate by trial and error): (a) fine-tune from the official checkpoint,
  (b) train from scratch, (c) fine-tune on the user's custom data.
- **Camera:** the test environment is mainly a single RGB-D camera
  (RealSense-like), possibly 2+. Training defaults to **realsense single
  camera**, with Kinect / dual camera easy to switch on.
- **GPU:** pin to a **single, user-specified index**. The index is decided at
  execution time after a fresh `nvidia-smi` check of others' usage. Never use
  all GPUs.
- **Initial data:** validate the pipeline on a **minimal subset first, then
  expand** (conservative).

## Approach

Base on the fork **`H-Freax/GraspNet_Pointnet2_PyTorch1.13.1`** rather than the
original repo. The original targets PyTorch 1.6, which has no Ampere (sm_86)
kernels and fails with "no kernel image is available". The fork ports the code
to an sm_86-capable torch and already consolidates the CUDA-extension build
patches (`AT_CHECK`→`TORCH_CHECK`, removal of `THC/THC.h`,
`THCState_getCurrentStream`→`at::cuda::getCurrentCUDAStream`). Less custom code
is safer.

Dedicated conda env `graspnet` (Python 3.8) + `torch==1.13.1+cu117` (ships sm_86
kernels). Build the `pointnet2` and `knn` extensions with the **local CUDA 11.8
nvcc** (a cu117 torch and cu118-built extensions are ABI-compatible within the
CUDA 11.x series). `graspnetAPI` / Open3D are evaluation-only and are **not
installed** (saves dependencies and disk). Pinned support libs: `numpy==1.23.5`,
`scipy==1.10.1`, `Pillow<10`, `tqdm`, `tensorboard`.

## Resource-control design (the safety core)

**GPU** — pin a single user-confirmed index via `CUDA_VISIBLE_DEVICES`. Before
launch the wrapper prints per-GPU `nvidia-smi` usage; if the chosen GPU is busy
it **refuses to launch**, and if no index is given it **prints the table and
stops** (never falls back to all GPUs).

**CPU / RAM** — `num_workers=2` (down from the repo's hardcoded 4);
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4` plus `torch.set_num_threads(4)`;
`MAX_JOBS=4` for the build (prevents 128-job compile storms). Conservative
defaults `batch_size=2`, `num_point=20000`.

**Disk** — the sharpest risk:

- Dataset lives **outside the repo** at `/workspace/data/graspnet/` (cannot be
  accidentally `git add`-ed) and is also listed in `.gitignore` defensively.
- Pre-flight check with a **hard abort**: refuse to start if free space <
  `ABORT_FLOOR_GB=80`; warn and require explicit confirmation if < `WARN_GB=150`.
- **Live disk monitor during training**: samples `df` ~every 60 s and, if free
  space drops below the floor, terminates the training process group
  (SIGTERM→SIGKILL). This is the single most important safety mechanism.
- Checkpoints / logs go to `/workspace/tools/runs/` (gitignored), keeping only
  the last N checkpoints.

**Authorization gates (report, then wait):** (1) before building the conda env,
(2) before the user downloads/places data, (3) before **every** training launch
(report the nvidia-smi table, chosen GPU, free disk, and settings).

## Directory layout & .gitignore

```
/workspace/tools/            (tracked repo)
  graspnet/                  <- cloned fork (gitignored: large, build artifacts)
  scripts/preflight.py       <- our wrapper (tracked)
  scripts/train_graspnet.sh  <- our wrapper (tracked)
  docs/setup_env.md          <- env/build doc (tracked)
  docs/dataset_download.md   <- download procedure doc (tracked)
  runs/                      <- logs/checkpoints (gitignored)
/workspace/data/graspnet/    <- dataset, outside the repo (cannot be git-added)
```

`.gitignore` additions (append): `graspnet/`, `runs/`, `data/`, `*.tar`.

## Training wrappers (3 modes, conservative defaults baked in)

- **`scripts/preflight.py`** (read-only safety check): prints the nvidia-smi
  table and validates the chosen GPU is idle, checks `df` free vs the floor and
  free RAM, then prints "SAFE TO LAUNCH" / "ABORT: <reason>" and exits non-zero
  on a problem.
- **`scripts/train_graspnet.sh`** (thin wrapper): requires `--gpu <N>` (no
  default; pins the GPU), exports thread caps / `CUDA_HOME` /
  `TORCH_CUDA_ARCH_LIST=8.6`, runs preflight first, starts the background disk
  monitor, then calls `train.py`. Defaults `--camera realsense
  --batch_size 2 --num_point 20000 --max_epoch 18`, `num_workers=2`.
- **Three modes (parameterized):** `MODE=scratch` (no checkpoint),
  `finetune-official` (`--checkpoint_path checkpoint-rs|kn.tar`),
  `finetune-custom` (user checkpoint + custom `--dataset_root`). Camera is also
  a parameter.
- The fork's hardcoded `num_workers=4` in `train.py` is changed to read from an
  argument/env. The fork is gitignored, so this edit is not part of our tracked
  code; it is documented in `docs/setup_env.md`.

## Build & verification (minimal subset first, proven step by step)

1. Env import: `torch.cuda.get_arch_list()` includes `sm_86`; `pointnet2._ext`
   and `knn_pytorch` import successfully.
2. Tiny forward pass (FPS + knn on a `(1, 1024, 3)` tensor) on the pinned GPU —
   proves kernels run on Ampere (catches "no kernel image"). No dataset needed.
3. Data-load smoke test: instantiate the dataset + one DataLoader batch on the
   minimal subset; assert tensor shapes.
4. One training step: `--max_epoch 1` with a small iteration count — forward +
   backward run, VRAM stays ~6–10 GB (fits one 3090), a checkpoint is written.
5. Checkpoint check: the `.tar` exists and is loadable via `torch.load`. Proves
   the full pipeline.
6. Only after steps 1–5 pass and `df` shows headroom, expand to the full
   realsense set (~35 GB).

## §4/§12 workflow mapping (split into four small PRs)

Each task: append ToDo.md → confirm with the user → `gh issue create` → branch
from `main` → work → PR. (The fork is gitignored, so no upstream code lands in
our PRs.)

1. `docs/graspnet-env-setup` — environment/build recipe (`docs/setup_env.md`).
2. `docs/graspnet-dataset` — download procedure, paths, disk budget.
3. `feature/graspnet-train-wrappers` — `preflight.py`, `train_graspnet.sh`,
   `.gitignore` additions (ruff-clean).
4. `docs/graspnet-validation` — minimal-subset validation results / runbook.

## Items to confirm before execution (proposed defaults; reject to change)

- GPU index: chosen at execution time after `nvidia-smi` (decision deferred).
- `ABORT_FLOOR_GB=80` and a minimal-phase data budget of `<= 60 GB` — acceptable?
- Dedicated conda env name `graspnet` (does not touch `base` / `peallab`).
- Permission to make the small `num_workers` edit in the fork's `train.py`.

## Execution gate

GraspNet setup, clone, install, build, download, and training will **not** start
until the user explicitly authorizes execution. Administrator approval of this
plan plus the user's go-ahead are both required before Task 1 begins.

## Sources

- `H-Freax/GraspNet_Pointnet2_PyTorch1.13.1` (PyTorch 1.13.1 / CUDA 11.x fork)
- `graspnet/graspnet-baseline` (README, `train.py`, `command_train.sh`)
- graspnet.net datasets page (sizes, form-gated download)
- PyTorch issue #73520 (THC removal) and the sm_86 compatibility forum threads
