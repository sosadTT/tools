# Deployment & Test Plan — trained GraspNet checkpoint on robot/sim

How to take the weights trained in this container to a PC with the robot (or a
sim) and test grasping with the **DH Robotics PGC-140-50** parallel gripper.
This container only trains; all robot/sim testing happens on the target PC.

## 0. What we are deploying

- **Weights**: `runs/full_scratch_realsense/checkpoint.tar` (12 MB, epoch 18;
  `model_state_dict`). Trained scratch, RealSense, batch 4, 18 epochs.
- **Model**: `GraspNet(is_training=False)` from the fork; grasp output is a
  generic **parallel-jaw 6-DoF grasp** (position, rotation, width, depth, score)
  — gripper-agnostic, width ≤ `GRASP_MAX_WIDTH = 0.075 m` (75 mm).
- The PGC-140-50 (~50 mm stroke) is inside this range → no retraining; filter by
  width at deploy time.

## 1. Transfer from this container → target PC

Bundle (the dataset is NOT needed for inference):

| Item | Path | Note |
|------|------|------|
| Weights | `runs/full_scratch_realsense/checkpoint.tar` | the trained model |
| Fork code | `graspnet/` (models, pointnet2, knn, utils, `graspnet_baseline.py`) | inference code + the vendored fixes |
| Curve (report) | `runs/full_scratch_realsense/training_curve.png` | optional |

```bash
# from the host (e.g.):
tar czf graspnet_deploy.tar.gz -C /workspace/tools \
    graspnet runs/full_scratch_realsense/checkpoint.tar
# scp graspnet_deploy.tar.gz <user>@<robot_pc>:~/   then untar there
```

> The compiled `pointnet2`/`knn` extensions are GPU-arch-specific — do NOT copy
> the built `.so`; rebuild on the target (next step).

## 2. Target-PC environment (inference)

Inference needs more than training did: **open3d + graspnetAPI** (we skipped
these in the training env). Per `docs/setup_env.md`, on the target:

```bash
conda create -n graspnet python=3.8 -y
# torch matching the TARGET GPU (cu117/cu118 wheel with that GPU's arch)
conda run -n graspnet pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
conda run -n graspnet pip install numpy==1.23.5 scipy open3d graspnetAPI
# build extensions FOR THE TARGET GPU arch (set TORCH_CUDA_ARCH_LIST accordingly):
cd graspnet/pointnet2 && conda run -n graspnet python setup.py install
cd ../knn          && conda run -n graspnet python setup.py install
```

- If the target GPU is also Ampere (sm_86) → `TORCH_CUDA_ARCH_LIST="8.6"`; set it
  to the target's compute capability otherwise.
- Same import bootstrap as in this repo (`graspnet` imported as a package; see
  `graspnet/graspnet_baseline.py`).

## 3. Inference pipeline (use the fork's `GraspNetBaseLine`)

`graspnet/graspnet_baseline.py` already implements the full path:

```python
from graspnet.graspnet_baseline import GraspNetBaseLine   # import as package
net = GraspNetBaseLine(checkpoint_path="checkpoint.tar", num_point=20000,
                       collision_thresh=0.001, voxel_size=0.01)
gg = net.inference(o3d_pcd)   # o3d point cloud (camera frame) -> GraspGroup
```

Steps it performs: load checkpoint → sample 20,000 points → forward →
`pred_decode` → `GraspGroup` → `ModelFreeCollisionDetector` filtering.

Input prep (camera → point cloud): build the cloud from the RGB-D frame with
`CameraInfo` (the **target camera's intrinsics**) and
`create_point_cloud_from_depth_image`, plus a workspace mask — see
`graspnet/utils/data_utils.py` and the `get_and_process_data` method.

## 4. PGC-140-50 gripper handling (deploy-time, no retraining)

`GraspGroup` exposes `.scores`, `.widths` (meters), `.translations`,
`.rotation_matrices`. Apply:

```python
import numpy as np
MAX_STROKE = 0.050                 # PGC-140-50 stroke 50 mm
gg = gg[gg.widths <= MAX_STROKE - 0.005]   # keep grasps within stroke (5 mm margin)
gg = gg.sort_by_score()
best = gg[0]                       # top grasp: translation + rotation (camera frame)
```

Then: grasp pose (camera frame) → **robot base frame** via hand-eye calibration
→ motion-plan/IK → command the PGC-140 to the predicted width and close.
(Optionally double-check finger collision with the gripper's real geometry.)

## 5. Test paths

**A. Sim (recommended first)** — safe, repeatable:
- Load the robot + PGC-140 in the sim (Isaac Sim / PyBullet / etc.) with a
  RealSense-like RGB-D sensor.
- Render RGB-D → point cloud → `GraspNetBaseLine.inference` → width filter →
  execute top grasp → record success/fail over a set of objects.

**B. Real robot**:
- Hand-eye calibration first; define a safe workspace; e-stop ready.
- Same inference → filter → execute; start with isolated objects, then clutter.

## 6. Success criteria

1. **Sanity** (no robot): checkpoint loads, `inference()` returns a non-empty
   `GraspGroup` with plausible scores on a sample RGB-D scene; top grasps have
   width ≤ 50 mm after filtering.
2. **Sim/real grasp success rate** over N attempts (e.g., ≥ K/N on isolated
   objects), then in clutter.
3. (Optional, quantitative) graspnetAPI **AP** on the official test splits —
   requires downloading `test_seen/similar/novel` + `models/dex_models` and
   running the eval (not needed for a functional robot test).

## 7. Baseline comparison

Download the authors' official RealSense checkpoint (`checkpoint-rs.tar`, from
the graspnet-baseline README's Google Drive) on the target and run the same
inference on the same scenes. Compare grasp quality / success vs our scratch
checkpoint to judge whether our run is adequate or needs more epochs /
custom-data fine-tuning (`finetune-custom` mode in `scripts/train_graspnet.sh`).

## 8. Gotchas

- **Rebuild extensions for the target GPU** (don't copy `.so`).
- **Camera intrinsics** must match the target sensor (`CameraInfo`).
- **Coordinate frames**: model outputs are in the camera frame; calibrate to the
  robot base.
- **Width units are meters** (≤ 0.075 from the model; filter to ≤ 0.050).
- Inference deps (`open3d`, `graspnetAPI`) are required on the target but were
  intentionally absent in the training env.
