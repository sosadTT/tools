# GraspNet-1Billion dataset download

## Primary method: HuggingFace mirror (no form, scriptable)

The public HF repo **`saic3d/graspnet`** mirrors the official zips —
**verified byte-identical** to the official distribution (sha256 of our
official `collision_label.zip` and `grasp_label.zip` matches the HF LFS oids
exactly). No registration form, no gating.

Use the wrapper (disk-floor gate + sha256 verification built in):

```bash
# download only what you need; extract into the loader layout; drop the zip
scripts/download_dataset.sh --files train_3.zip --extract --rm-zip
scripts/download_dataset.sh --files grasp_label.zip,collision_label.zip --extract
```

Notes:
- Downloads are **resumable** (`huggingface_hub.hf_hub_download`).
- The script refuses to start if free disk minus the projected peak
  (download + extraction) would breach the 80 GB floor.
- Each file's sha256 is checked against the HF LFS oid after download.
- **Streaming training is not possible**: every known distribution of this
  dataset (HF repos, Zenodo, OpenDataLab, Kaggle) is archive-only (zip/tar; no
  parquet/webdataset), and the training loader reads per-scene PNG/.mat files
  from disk. Surveyed 2026-06; the HF zip mirror + selective download is the
  best available option.
- The mirror is third-party (not the original authors); the sha256 check above
  is why we trust it. If a file ever mismatches, stop and fall back to the
  official source below.

## Fallback: official graspnet.net (form-gated)

Request access at <https://graspnet.net/datasets.html>, download from the
Google Drive links, and place the files manually (the original flow).

## Target location (outside the git repo)

```
/workspace/data/graspnet/
```

Intentionally outside `/workspace/tools` so it can never be added to git (also
in `.gitignore`). The disk is shared with the host — mind the budget below.

## Disk budget (shared disk — be conservative)

| Component | File | Size | Needed? |
|-----------|------|------|---------|
| Train images | `train_1..4.zip` | ~68 GB | full 100-scene train |
| 6-DoF grasp labels | `grasp_label.zip` | ~2 GB | required |
| Collision labels | `collision_label.zip` | ~0.4 GB | required |
| Test (seen) | `test_seen.zip` | ~21 GB | only for per-epoch eval |
| Object models | `models.zip` | ~4.6 GB | optional |
| Rect labels / DexNet | `rect_labels`, `dex_models` | ~23 GB | NOT needed |

Use `--rm-zip` to avoid double occupancy after extraction.

## Expected directory layout

```
/workspace/data/graspnet/
├── scenes/              # scene_0000 ... (train zips extract here)
│   └── scene_XXXX/{realsense,kinect}/{rgb,depth,label,meta}/...
├── grasp_label/         # 000_labels.npz ... 087_labels.npz
└── collision_label/     # scene-wise collision masks
```

## Tolerance labels (generated locally, not downloaded)

Required by the loader; generated once from `grasp_label/` into
`graspnet/dataset/tolerance/` (the loader reads them relative to the fork's
`dataset/` dir — see `docs/setup_env.md`):

```bash
cd /workspace/tools/graspnet/dataset
conda run -n graspnet python generate_tolerance_label.py \
    --dataset_root /workspace/data/graspnet --num_workers 32
```

## After download — verify, then train

1. Confirm the layout above exists at `/workspace/data/graspnet`.
2. Run the pre-flight check: `python scripts/preflight.py --gpu <N>`.
3. Launch via `scripts/train_graspnet.sh` (see that script).
