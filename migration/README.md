# Docker migration kit

Move this training setup to another machine for testing. The approach is a
host-side `docker commit`/`save` for the environment, plus a separate bundle for
the host bind-mount artifacts. The large dataset is **excluded**.

## What lives where (important)

`docker commit` captures only the container's own filesystem. `/workspace` is a
**host bind-mount**, so it is NOT in the image and must be handled separately.

| Item | Location | In image? | Migration |
|------|----------|-----------|-----------|
| conda env `graspnet` (+ built pointnet2/knn) | `/opt/conda/envs/graspnet` | ✅ yes | via image |
| CUDA 11.8, `/root/.local` (uv) | container fs | ✅ yes | via image |
| repo | `/workspace/tools` | ❌ no | `git clone` (on GitHub) |
| fork source + tolerance | `/workspace/tools/graspnet` (~6 GB) | ❌ no | tar bundle (below) |
| **dataset (drag-and-dropped zips + extracted)** | `/workspace/data/graspnet` (~122 GB) | ❌ no | **EXCLUDED — re-acquire on target** |

## Step A — Environment image (run on the HOST, not inside the container)

The container has no docker access; do this from the host shell.

```bash
# 1. Find the container (its hostname inside is the short id, e.g. 0af52f1b625a)
docker ps

# 2. Commit the running container to an image
docker commit <container_id> graspnet-train:migrate

# 3. Save the image to a portable tarball
docker save graspnet-train:migrate | gzip > graspnet-train-migrate.tar.gz

# 4. Transfer graspnet-train-migrate.tar.gz to the target machine, then:
docker load < graspnet-train-migrate.tar.gz
```

The image will be large (conda env ~3.7 GB + CUDA toolkit + base). It does NOT
contain `/workspace`.

## Step B — Host bind-mount artifacts (repo + fork + tolerance)

These are on `/workspace` and are NOT in the image. Two ways:

**B1. Bundle (keeps the compiled-free fork source + the 5.7 GB tolerance labels)**
Run this so the tarball lands OUTSIDE the dir being archived, and the 122 GB
dataset is never included (it is under `/workspace/data`, not `/workspace/tools`):
```bash
tar czf /workspace/data/graspnet-workspace.tar.gz \
    -C /workspace tools \
    --exclude='tools/.git' --exclude='tools/.ruff_cache' \
    --exclude='tools/CommonClaude'
# transfer graspnet-workspace.tar.gz, then on target:
#   mkdir -p /workspace && tar xzf graspnet-workspace.tar.gz -C /workspace
```

**B2. Reconstruct from sources (lighter transfer, needs build/regeneration)**
On the target (inside the loaded image), per `docs/setup_env.md`:
```bash
git clone <this repo> /workspace/tools && cd /workspace/tools
git checkout <branch>
git clone --depth 1 \
    https://github.com/H-Freax/GraspNet_Pointnet2_PyTorch1.13.1.git graspnet
# rebuild extensions only if NOT using the env image; the image already has them
# regenerate tolerance (~40 min) OR copy graspnet/dataset/tolerance/ from B1
```

If you used the **env image** (Step A), the compiled pointnet2/knn extensions
are already installed in the env, so you only need the fork SOURCE (for
`graspnet/train.py`, `dataset/`, `models/`) and `tolerance/` — both included in
the B1 bundle.

## Step C — Dataset (EXCLUDED)

The ~122 GB `/workspace/data/graspnet` (the drag-and-dropped `train_*.zip`,
`grasp_label.zip`, `collision_label.zip`, and extracted data) is **not
migrated**. Re-acquire it on the target per `docs/dataset_download.md` when
needed, into `/workspace/data/graspnet/`.

## Step D — Run on the target

```bash
docker run --gpus all --privileged \
    -v <target_host_workspace>:/workspace \
    -it graspnet-train:migrate bash
```
Then verify (no dataset needed):
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n graspnet python graspnet/train.py --dry_run
conda run -n graspnet python claude_test/graspnet_ddp_smoke.py
```

## Reference / fallback

- `migration/environment.yml` — conda env spec (export of `graspnet`).
- `migration/requirements-lock.txt` — `pip freeze` of the env.
These let you rebuild the env without the image if needed (then rebuild the
pointnet2/knn extensions per `docs/setup_env.md`).
