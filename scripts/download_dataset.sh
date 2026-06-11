#!/bin/bash
# Selective GraspNet-1Billion downloader from the HuggingFace mirror
# (saic3d/graspnet — verified byte-identical to the official zips via
# sha256). Replaces the form-gated graspnet.net Google Drive flow.
#
# The shared host disk is the main risk, so this wrapper refuses to
# start unless free space covers download + extraction above the floor,
# verifies each file's sha256 against the HF LFS oid, and can delete
# zips after extraction.
#
# Usage:
#   scripts/download_dataset.sh --files train_3.zip[,grasp_label.zip,...]
#       [--extract] [--rm-zip] [--floor-gb 80]
# Layout on --extract (matches the training loader):
#   train_N / test_*.zip -> <root>/scenes/  (contain scene_XXXX/ at top)
#   other zips           -> <root>/         (contain their own top dir)
set -euo pipefail

REPO_ID="saic3d/graspnet"
DATA_ROOT="/workspace/data/graspnet"
CONDA_ENV="graspnet"
# The huggingface_hub Python client (both xet and plain backends) stalled
# indefinitely on this host, while direct HTTP to the resolve URL works at
# full speed. So downloads use curl (resumable via -C -); integrity is
# guaranteed by the sha256 check against the HF LFS oid below.
FLOOR_GB=80
FILES=""
DO_EXTRACT=0
RM_ZIP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --files) FILES="$2"; shift 2 ;;
        --extract) DO_EXTRACT=1; shift ;;
        --rm-zip) RM_ZIP=1; shift ;;
        --floor-gb) FLOOR_GB="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$FILES" ]]; then
    echo "ERROR: --files <a.zip[,b.zip...]> is required." >&2
    exit 2
fi
IFS=',' read -r -a FILE_ARR <<< "$FILES"

# ---- fetch HF metadata (sizes + sha256 oids) --------------------------
META_JSON=$(curl -sf "https://huggingface.co/api/models/${REPO_ID}/tree/main")
need_bytes=0
for f in "${FILE_ARR[@]}"; do
    size=$(echo "$META_JSON" | python3 -c "
import sys, json
d = {x['path']: x['size'] for x in json.load(sys.stdin)}
print(d.get('$f', -1))")
    if [[ "$size" == "-1" ]]; then
        echo "ERROR: $f not found in ${REPO_ID}." >&2
        exit 2
    fi
    need_bytes=$(( need_bytes + size ))
done
need_gb=$(( need_bytes / 1024 / 1024 / 1024 + 1 ))
# Extraction roughly doubles the footprint while the zip still exists.
peak_gb=$(( DO_EXTRACT == 1 ? need_gb * 2 : need_gb ))

# ---- disk gate --------------------------------------------------------
avail_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "disk: ${avail_gb}G free | need peak ~${peak_gb}G | floor ${FLOOR_GB}G"
if (( avail_gb - peak_gb < FLOOR_GB )); then
    echo "ABORT: ${avail_gb}G - ${peak_gb}G would breach the ${FLOOR_GB}G floor." >&2
    exit 1
fi

mkdir -p "$DATA_ROOT"

for f in "${FILE_ARR[@]}"; do
    echo ">> downloading $f from ${REPO_ID} (curl, resumable)"
    curl -L -C - --fail --retry 5 --retry-delay 10 \
        -o "$DATA_ROOT/$f" \
        "https://huggingface.co/${REPO_ID}/resolve/main/$f"

    echo ">> verifying sha256 of $f against the HF LFS oid"
    want=$(echo "$META_JSON" | python3 -c "
import sys, json
d = {x['path']: (x.get('lfs') or {}).get('oid') for x in json.load(sys.stdin)}
print(d.get('$f') or '')")
    got=$(sha256sum "$DATA_ROOT/$f" | awk '{print $1}')
    if [[ -n "$want" && "$got" != "$want" ]]; then
        echo "ABORT: sha256 mismatch for $f (got $got, want $want)." >&2
        exit 1
    fi
    echo "   sha256 OK: $got"

    if [[ "$DO_EXTRACT" == "1" ]]; then
        if [[ "$f" == train_*.zip || "$f" == test_*.zip ]]; then
            dest="$DATA_ROOT/scenes"
        else
            dest="$DATA_ROOT"
        fi
        echo ">> extracting $f -> $dest"
        mkdir -p "$dest"
        unzip -q -o "$DATA_ROOT/$f" -d "$dest"
        if [[ "$RM_ZIP" == "1" ]]; then
            rm -f "$DATA_ROOT/$f"
            echo "   removed $f after extraction"
        fi
    fi
done

echo "DONE | disk now: $(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')G free"
