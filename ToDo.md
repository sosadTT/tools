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
- [ ] Preflight gate + run validation on a user-chosen GPU
- [ ] Verify checkpoint + report; commit + PR (stacks on #9)

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
- [ ] Integrity-check the 5 dataset zips (CRC test) [background]
- [x] Export conda env (`migration/environment.yml`, `requirements-lock.txt`)
- [x] Write `migration/README.md` runbook (image + host artifacts + exclusions)
- [ ] Commit + PR (stacks on #11)
