# GraspNet minimal-subset training validation

Validates the paper-faithful single-GPU pipeline end-to-end on a minimal subset
before full training.

## Setup

- Single GPU (GPU 2, shared with another user), paper-faithful `batch_size=4`,
  `num_workers=2`, 1 epoch.
- Minimal subset: `train_1` scenes 0000-0029, realsense; `--skip_eval` (no test
  split available).
- Command:
  ```
  scripts/train_graspnet.sh --gpu 2 --mode scratch --camera realsense \
    --batch_size 4 --num_workers 2 --max_epoch 1 \
    --scenes 0-29 --skip_eval --allow_busy
  ```

## Results — pipeline validated

- **preflight**: SAFE TO LAUNCH (GPU 2 busy → warned, shared use allowed; disk
  377 GB, RAM 458 GB).
- **model**: built on `cuda:0`, params 1,025,964, lr 0.001.
- **labels**: 88 grasp + tolerance loaded — confirms the `BASE_DIR` bugfix and
  the generated tolerance labels load correctly.
- **dataset**: `train=7680` samples (30 scenes x 256) — the `scene_ids` subset
  restriction works.
- **training**: forward + backward run with **0 errors**. All 7 loss components
  compute (including `grasp_tolerance_loss`, confirming tolerance labels are
  wired correctly). Loss decreases as training proceeds:

  | batch | overall_loss | grasp_tolerance_loss |
  |-------|--------------|----------------------|
  | 130   | 1.0485       | 0.1561               |
  | 900   | (decreasing) | 0.0253               |

  The decreasing tolerance loss shows the model is actually learning.

- **resource safety** (verified mid-run): GPU 2 at 19.2/24 GB (our ~15.4 GB +
  co-tenant ~3.8 GB, ~5.3 GB headroom, no OOM); disk 377 GB free with the live
  monitor active; RAM 409 GB free; CPU load ~7 / 128 cores. No container-crash
  risk and minimal impact on the GPU co-tenant.

## Checkpoint

The full 1-epoch run completed all **1920 batches** (exit code 0). Final
training `overall_loss = 0.726` (down from 1.05 at batch 130). The checkpoint
was saved and verified:

- `runs/scratch_realsense/checkpoint.tar` — **12 MB**, `torch.load` OK.
- Keys: `epoch`, `loss`, `model_state_dict` (162 tensors), `optimizer_state_dict`.
- `epoch = 1`. (`loss = 0.0` only because `--skip_eval` skips the eval-loss
  field; the training loss decreased normally as shown above.)
- After the run, GPU 2 returned to ~3.8 GB (our process released cleanly); disk
  held at 377 GB.

## Verdict

**Minimal-subset validation PASSED.** The patched pipeline — `BASE_DIR` fix,
`scene_ids` arg, `train.py` `--scenes/--skip_eval/--max_iters`, tolerance
generation + flat-import fix — runs end-to-end on real GraspNet data with
sensible, decreasing losses, within safe resource limits on a shared GPU.

## Next (full training)

Needs `train_3.zip` (scenes 0060-0089) for the full 100-scene train split,
restoring the full scene range, and the `test_seen` scenes (100-129) for
per-epoch evaluation; then run without `--scenes/--skip_eval/--max_iters`.
