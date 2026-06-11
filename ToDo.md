# ToDo

> Cumulative command history for Claude Code sessions in this repository.
> Append-only: new tasks are added below; historical entries are never
> rewritten or reordered (see CLAUDE.md §4).

## 1. Initialize repository with Claude conventions

### Background
serena MCP and the CommonClaude rule set (`CLAUDE.md` + `.claude/` hooks)
were applied to `/workspace/tools`, but the working root was not a git
repository, so the §4/§12 branch-and-PR workflow could not run. This task
bootstraps the repository so that all subsequent work can follow the full
workflow.

This is a bootstrap exception: the genesis commit goes directly to `main`
because no branch can be cut before the repository exists. Every task after
this one follows the full §4/§12 flow (issue -> branch -> PR).

### Tasks
- [x] Configure git identity (sosadTT / sy000217@gmail.com)
- [x] Initialize repository on `main` (`git init -b main`)
- [x] Add `.gitignore` based on the §13.1 Python template, excluding
      `CommonClaude/`
- [x] Create this `ToDo.md`
- [x] Register GitHub issue #1 (closes #1)
- [x] Create the genesis commit on `main` (31abb9d)
- [x] Create a private GitHub repository and push

## 2. Share GraspNet training plan for administrator review

### Background
The real task is to set up TRAINING-ONLY of graspnet/graspnet-baseline in this
container, with strict guarantees that it never harms the host or other
containers (no resource isolation exists; the disk is shared and tight). Before
any resource-consuming action, the administrator must review and approve the
plan. This task only shares the plan via the repository (a PR is the review
surface). No GraspNet setup/build/download/training is performed here.

### Tasks
- [x] Create `docs/PLAN-graspnet.md` (English plan for review)
- [x] Create branch `docs/graspnet-plan-review`
- [x] Register GitHub issue for plan review (#2)
- [x] Open a PR so the administrator can review (#3)
- [x] Invite the administrator (coport-uni, write) as a collaborator
      (invitation pending acceptance; reviewer request can be added once
      accepted)

## 3. GraspNet environment setup + multi-GPU (DDP) smoke test

### Background
Administrator approved the plan (PR #3 merged) with two changes: allow full GPU
use (assuming no OOM) and report the paper's training GPU. The user triggered
execution but asked to proceed step by step: decide DDP vs DataParallel by
testing first, with only a short (~5-10 min) safe test.

Pre-flight (read-only) confirmed: 392 GB free disk, all 3x RTX 3090 idle, no
`graspnet` conda env yet. The GraspNet-1Billion dataset is form-gated and cannot
be auto-downloaded, but the DDP-vs-DataParallel decision does not need it: a
synthetic (random point cloud) multi-GPU smoke test exercises the pointnet2/knn
custom CUDA ops under each parallel mode. (See PLAN-graspnet.md / GitHub #4.)

### Tasks
- [x] Append `.gitignore` rules (graspnet/, runs/, data/, *.tar)
- [x] Clone fork `H-Freax/GraspNet_Pointnet2_PyTorch1.13.1` into `graspnet/`
- [x] Create dedicated conda env `graspnet` (Python 3.8, torch 1.13.1+cu117)
- [x] Build `pointnet2` and `knn` extensions (CUDA 11.8, sm_86, MAX_JOBS=4)
- [x] Verify: `sm_86` in arch list, extensions import, tiny single-GPU forward
- [x] Multi-GPU smoke test on synthetic data: DataParallel and DDP both PASS
- [x] Report results -> recommend DDP (both work; DDP is the robust choice)

## 4. Dataset download doc + single-GPU training wrappers

### Background
Verified from the CVPR 2020 paper PDF: the baseline was trained on ONE Nvidia
RTX 2080, batch_size 4, Adam lr 0.001 (->1e-4 @60ep, ->1e-5 @100ep). The
official repo's released config differs (batch 2, 18 epochs). The user chose the
paper-faithful setup: SINGLE-GPU, batch_size 4 (one RTX 3090 has ample headroom;
no DDP needed). Next step is to prepare the dataset download procedure and the
training wrappers so a run can start once the form-gated dataset is placed.

### Tasks
- [x] Write `docs/dataset_download.md` (form-gated download, minimal subset,
      target `/workspace/data/graspnet`, disk budget, tolerance-label step)
- [x] Write `docs/setup_env.md` (env recipe, build, import bootstrap, vendored
      train.py)
- [x] Write `scripts/preflight.py` (read-only GPU/disk/RAM safety check)
- [x] Write `scripts/train_graspnet.sh` (single-GPU, batch 4, paper LR
      schedule; modes scratch/finetune-official/finetune-custom; preflight +
      disk monitor)
- [x] Adapt a train entrypoint to the fork (vendored `graspnet/train.py`);
      `--dry_run` builds model+optimizer without data (verified, params ~1.03M)
- [x] Commit + PR (#7, stacks on #5)

## 5. Guarded operations: require approval even in auto mode

### Background
The user asked that two risky operations always warn and require explicit
approval, even in auto/bypass mode:
1. `pkill -f unicorn` (kills (g)unicorn workers),
2. changing nginx `worker_processes` from `auto` to a manual count (e.g., 2).
A documented rule alone is weak in headless mode, so we also add a PreToolUse
hook (hooks still run under bypass mode) that blocks these commands unless an
explicit ACK marker is present (added only after the user approves).

### Tasks
- [x] Add a CLAUDE.md section (§18) documenting the guarded operations + rule
- [x] Add `.claude/hooks/pre-guarded-ops.sh` (PreToolUse Bash guard)
- [x] Wire the hook into `.claude/settings.json`
- [x] Test the hook: blocks pkill (g)unicorn + numeric worker_processes;
      allows reads/reload/benign; allows with GUARDED_OPS_ACK=1
- [x] Commit + PR (#9)

## 6. Minimal-subset GraspNet training validation

### Background
User placed train_1.zip / grasp_label.zip / collision_label.zip in
`/workspace/data/graspnet/`. Validate the paper-faithful single-GPU pipeline
(batch 4) on a minimal subset: data load -> 1 step -> checkpoint, before full
training. Read-only investigation found: loader assumes scenes 0000-0099 and a
test split we lack (so restrict to train_1 scenes 0000-0029 + skip eval);
`load_grasp_labels` needs tolerance labels (generate); and the fork's
`graspnet_dataset.py` uses an undefined `BASE_DIR` (bugfix needed). All 3 GPUs
are currently shared by other users, so the GPU is chosen at run time behind a
preflight gate.

### Tasks
- [ ] Extract train_1 -> `scenes/`, grasp_label/, collision_label/
- [ ] Generate tolerance labels (num_workers=4) into `graspnet/dataset/tolerance/`
- [ ] Patch vendored: BASE_DIR fix + `scene_ids` arg + train.py
      `--scenes/--skip_eval/--max_iters`
- [ ] Enhance `scripts/preflight.py` (`--allow-busy`) + launcher passthrough
- [x] Preflight gate + run validation on GPU 2 (1 epoch, batch 4, scenes 0-29,
      skip_eval): full 1920 batches, exit 0, loss 1.05->0.726, no OOM
- [x] Verify checkpoint + report: checkpoint.tar 12 MB, torch.load OK (epoch 1,
      162 tensors); documented in docs/graspnet-validation.md; commit + PR

## 7. Docker migration kit + dataset integrity check

### Background
A GPU is likely occupied for a while, so the user wants to migrate to another
machine for testing. Two parts: (1) integrity-check the dataset zips that were
drag-and-dropped from the remote PC yesterday, and (2) prepare a Docker
migration kit. docker is not available inside the container and `/workspace` is
a host bind-mount (not captured by `docker commit`), so the kit covers both the
env image and the host-side `/workspace` artifacts. The ~122 GB dataset and the
drag-and-dropped zips are excluded from migration.

### Tasks
- [x] Integrity-check the 5 dataset zips (CRC test): train_1/2/4, grasp_label,
      collision_label all OK (no drag-and-drop corruption)
- [x] Export conda env (`migration/environment.yml`, `requirements-lock.txt`)
- [x] Write `migration/README.md` runbook (image + host artifacts + exclusions)
- [x] Write `HANDOFF.md` (project-context handoff for another machine/session)
- [x] Commit + PR (#13, stacks on #11)

## 8. Switch dataset acquisition to HuggingFace mirror + fetch train_3

### Background
The user found a public HF mirror of GraspNet-1Billion (`saic3d/graspnet`) and
asked whether downloads could be switched to it, ideally streaming. Survey
(HF 4 repos, Zenodo, OpenDataLab, Kaggle) found NO streaming-friendly
(parquet/webdataset) distribution anywhere — all archive-only — and the
training loader reads per-scene PNG/.mat from disk, so true streaming is not
possible. The real win is form-free, scriptable, selective, resumable
downloads. The mirror's zip sizes are byte-identical to our official copies;
verify once via sha256 before trusting it. Also fetch the missing train_3.zip
(scenes 0060-0089; peak +41 GB on 377 GB free, floor 80 GB — safe).

### Tasks
- [x] Install huggingface_hub into the `graspnet` env (0.36.2)
- [x] Verify mirror: sha256 of our official collision_label.zip AND
      grasp_label.zip exactly match the HF LFS oids (no download needed)
- [x] Write `scripts/download_dataset.sh` (disk gate tested, sha256 verify,
      extract, rm-zip; switched to curl because hf_hub client stalled on
      this host while direct HTTP runs at full speed)
- [x] Revise `docs/dataset_download.md` (HF primary, Drive fallback, no
      streaming note) + HANDOFF.md update
- [x] Download + extract train_3: sha256 OK (f86adcd0...), scenes 0060-0089
      extracted -> 60 scenes total, zip removed, disk 349G (floor 80G)
- [x] Commit + PR (#15, stacks on #13)

## 9. Complete full-training dataset (train_2/4 + test_seen)

### Background
60 scenes present (0000-0029, 0060-0089). To be ready for full GraspNet
training, extract the remaining held train splits and fetch the per-epoch eval
split, staying well above the 80 GB disk floor (347 GB free). train_2/4 zips are
already on disk (drag-and-dropped, CRC-OK) — extract only, do not delete.
test_seen comes from the HF mirror. test_novel/similar and models/dex/rect are
not needed for training and are skipped. Fixed download_dataset.sh so test_*.zip
extracts into scenes/ like train zips.

### Tasks
- [x] Fix download_dataset.sh: test_*.zip -> scenes/
- [x] Extract train_2 (0030-0059) + train_4 (0090-0099) -> 100 train scenes
- [x] Download + extract test_seen (0100-0129) via the script (sha256 OK
      bba24fce..., --rm-zip)
- [x] Verify 130 scenes (train 100 + test_seen 30); spot-check OK; disk 298G
- [x] Commit + PR (stacks on #15)

## 10. Training-spec verification script (read-only)

### Background
User wants to independently verify (and show their manager) the reported
training details. Add a read-only script that prints the source + measured
value for each. While gathering evidence, two reported numbers were corrected:
VRAM ~8G -> ~15G (training, not inference), and total time ~4.5d -> ~3d at
num_workers=2 / ~1.5-2d at 8 (measured 0.504 batch/s). Full scratch training
remains HELD pending manager approval.

### Tasks
- [x] Write `scripts/verify_training_specs.sh` (read-only; config, speed,
      data scale, GPU, VRAM evidence, external sources, time estimate)
- [x] `bash -n` + run once: 0.504 batch/s, 6400 batch/epoch, ~3.4d@nw2 confirmed
- [x] Commit + PR (stacks on #17)

## 11. Full scratch training (18 epoch, 100 scenes) in tmux

### Background
User authorized the run. Paper-faithful: single GPU, batch 4, 18 epochs, lr
0.001 decay @8/12/16 (official repo), num_workers 8, 100 train scenes +
per-epoch test_seen eval. Runs in a tmux session (persistent, user-attachable).
GPU gate: GPU 0 unsafe at ~15G VRAM; use GPU 1 or 2 (2 chosen, as in validation).
Logs to runs/full_scratch_realsense (log_train.txt + tensorboard); auto-resume
via checkpoint. Only tracked change: launcher lr_decay passthrough.

### Tasks
- [x] Add `--lr_decay_steps/--lr_decay_rates` passthrough to launcher (bash -n)
- [ ] Install tmux; preflight gate report; launch on GPU 2 in tmux
- [ ] Confirm started (log/GPU); report attach instructions + ETA
- [ ] Commit launcher change + PR (stacks on #19)
