# claude_test

Index of debug / exploratory / diagnostic scripts (see CLAUDE.md §3).
These are throwaway scripts, exempt from the 80-column and docstring rules
(§8). Anything promoted into `tests/` must conform fully.

| File | Purpose | What was learned |
|------|---------|------------------|
| `graspnet_ddp_smoke.py` | Synthetic multi-GPU smoke test for the GraspNet fork: runs `GraspNet(is_training=False)` forward+backward on random point clouds under single-GPU, DataParallel, and DDP, to decide the multi-GPU mode without the form-gated dataset. | All three PASS on 3x RTX 3090 (sm_86) with torch 1.13.1+cu117. The pointnet2/knn custom CUDA ops work under both DataParallel and DDP (no device-mismatch). Single-GPU peak ~1.87 GB at batch 1 / 20000 pts. DDP reported no unused params, so `find_unused_parameters=False` is fine. The fork's `__init__.py` pulls open3d/graspnetAPI, so the test registers `graspnet` as a namespace package to skip it. Recommendation: use DDP. |
