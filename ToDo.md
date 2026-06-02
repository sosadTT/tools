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
- [ ] Create `docs/PLAN-graspnet.md` (English plan for review)
- [ ] Create branch `docs/graspnet-plan-review`
- [ ] Register GitHub issue for plan review
- [ ] Open a PR so the administrator can review
- [ ] Add the administrator as a collaborator and request review
