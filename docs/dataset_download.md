# GraspNet-1Billion dataset download (manual)

This container does **not** download the dataset automatically: GraspNet-1Billion
is gated behind a registration form at <https://graspnet.net/datasets.html>.
Request access there, then download the files yourself and place them at the
path below. This document is the procedure; the training wrappers expect the
result.

## Target location (outside the git repo)

Place the dataset at:

```
/workspace/data/graspnet/
```

This is intentionally **outside** `/workspace/tools` so it can never be added to
git, and it is also listed in `.gitignore` defensively. The disk is shared with
the host, so mind the budget below.

## Disk budget (shared disk — be conservative)

The full dataset is ~152 GB; **do not download all of it here.** For
paper-faithful single-camera training, download only what is needed:

| Component | File | Size | Needed? |
|-----------|------|------|---------|
| Train images (split 1) | `train_1.zip` | ~20 GB | minimal subset |
| Train images (splits 2-4) | `train_2..4.zip` | ~46 GB | full single-camera train |
| 6-DoF grasp labels | `grasp_label.zip` | ~1.9 GB | required |
| Collision labels | `collision_label.zip` | ~0.4 GB | required |
| Object models | `models.zip` | ~4.3 GB | optional (collision/eval) |
| Test images | `test_*.zip` | ~59 GB | NOT needed (training only) |
| Rectangle labels | `rect_label.zip` | ~12.5 GB | NOT needed |
| DexNet models | `dex_models` | ~8.9 GB | NOT needed |

Recommended phases:
- **Pipeline validation**: `train_1.zip` + `grasp_label` + `collision_label`
  (~22 GB). Enough to validate the data path and run a short test.
- **Full single-camera (realsense) train**: `train_1..4.zip` + labels (~68 GB).

Before downloading, check free space (`scripts/preflight.py` also does this):
keep well above the `ABORT_FLOOR_GB=80` floor.

## Expected directory layout

After extracting, the root must look like this (the loader expects these names):

```
/workspace/data/graspnet/
├── scenes/
│   ├── scene_0000/
│   │   ├── realsense/   # rgb/ depth/ label/ ...
│   │   └── kinect/
│   └── ...
├── grasp_label/         # 0000_labels.npz ... 0087_labels.npz
├── collision_label/     # scene-wise collision masks
└── models/              # optional
```

## Tolerance labels (generated, not downloaded)

The baseline needs per-grasp "tolerance" labels. Generate them once after the
data is in place (run inside the `graspnet` conda env, from the fork's
`dataset/` dir):

```
cd /workspace/tools/graspnet/dataset
python generate_tolerance_label.py --dataset_root /workspace/data/graspnet \
    --camera realsense --num_workers 2
```

This writes `tolerance/` next to the labels. Keep `--num_workers` small (shared
box). Alternatively the authors provide a `tolerance.tar` download; if used,
extract it into the dataset root.

## After download — verify, then train

1. Confirm the layout above exists at `/workspace/data/graspnet`.
2. Run the pre-flight check: `python scripts/preflight.py --gpu 0`.
3. Launch a short test run with `scripts/train_graspnet.sh` (see that script).

No training starts until the data is present and the pre-flight passes.
