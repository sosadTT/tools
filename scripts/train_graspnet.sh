#!/bin/bash
# Conservative single-GPU launcher for GraspNet baseline training.
#
# Paper-faithful defaults (CVPR 2020): single GPU, batch_size 4, Adam
# lr 0.001 with decay at epochs 60 and 100. One RTX 3090 has ample
# headroom. The container has no resource isolation and shares its disk
# with the host, so this wrapper:
#   1) runs scripts/preflight.py and aborts unless it passes,
#   2) caps CPU threads and DataLoader workers,
#   3) starts a background disk monitor that kills training before the
#      shared disk can fill.
#
# Usage:
#   scripts/train_graspnet.sh --gpu 0 --mode scratch \
#       --dataset_root /workspace/data/graspnet
# Modes:
#   scratch           start from random init
#   finetune-official resume from the official checkpoint-rs/kn.tar
#   finetune-custom   resume from a user checkpoint (--checkpoint)
set -euo pipefail

# ---- defaults (override via flags) ------------------------------------
GPU=""
MODE="scratch"
CAMERA="realsense"
DATASET_ROOT="/workspace/data/graspnet"
CHECKPOINT=""
BATCH_SIZE=4
NUM_POINT=20000
MAX_EPOCH=100
LEARNING_RATE=0.001
NUM_WORKERS=2
LOG_DIR=""
CONDA_ENV="graspnet"
ABORT_FLOOR_GB=80

REPO_ROOT="/workspace/tools"
FORK_DIR="${REPO_ROOT}/graspnet"
TRAIN_PY="${FORK_DIR}/train.py"

# ---- parse flags ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --camera) CAMERA="$2"; shift 2 ;;
        --dataset_root) DATASET_ROOT="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --num_point) NUM_POINT="$2"; shift 2 ;;
        --max_epoch) MAX_EPOCH="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$GPU" ]]; then
    echo "ERROR: --gpu is required (single index, e.g. --gpu 0)." >&2
    exit 2
fi
# Paper-faithful default is one GPU; refuse multi-GPU here on purpose.
if [[ "$GPU" == *,* ]]; then
    echo "ERROR: this launcher is single-GPU; pass one index." >&2
    exit 2
fi

if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="${REPO_ROOT}/runs/${MODE}_${CAMERA}"
fi

# ---- pre-flight safety gate ------------------------------------------
echo ">> pre-flight check"
python "${REPO_ROOT}/scripts/preflight.py" \
    --gpu "$GPU" --abort-floor-gb "$ABORT_FLOOR_GB"

# ---- conservative resource caps --------------------------------------
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# ---- background disk monitor (kills training if disk gets low) -------
mkdir -p "$LOG_DIR"
MONITOR_PID=""
start_disk_monitor() {
    local target_pid="$1"
    (
        while kill -0 "$target_pid" 2>/dev/null; do
            avail=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
            if [[ "$avail" -lt "$ABORT_FLOOR_GB" ]]; then
                echo "DISK MONITOR: free ${avail}G < ${ABORT_FLOOR_GB}G," \
                     "terminating training $target_pid" >&2
                kill -TERM "$target_pid" 2>/dev/null || true
                sleep 10
                kill -KILL "$target_pid" 2>/dev/null || true
                break
            fi
            sleep 60
        done
    ) &
    MONITOR_PID=$!
}

# ---- assemble checkpoint flag per mode -------------------------------
CKPT_FLAG=()
case "$MODE" in
    scratch) ;;
    finetune-official|finetune-custom)
        if [[ -z "$CHECKPOINT" ]]; then
            echo "ERROR: $MODE needs --checkpoint <path.tar>." >&2
            exit 2
        fi
        CKPT_FLAG=(--checkpoint_path "$CHECKPOINT") ;;
    *) echo "Unknown --mode: $MODE" >&2; exit 2 ;;
esac

if [[ ! -f "$TRAIN_PY" ]]; then
    echo "ERROR: train entrypoint missing: $TRAIN_PY" >&2
    echo "See docs/setup_env.md for the train.py integration step." >&2
    exit 2
fi

echo ">> launching: mode=$MODE camera=$CAMERA gpu=$GPU batch=$BATCH_SIZE"
echo "   dataset_root=$DATASET_ROOT log_dir=$LOG_DIR"

conda run -n "$CONDA_ENV" --no-capture-output python "$TRAIN_PY" \
    --camera "$CAMERA" \
    --dataset_root "$DATASET_ROOT" \
    --log_dir "$LOG_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_point "$NUM_POINT" \
    --max_epoch "$MAX_EPOCH" \
    --learning_rate "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    "${CKPT_FLAG[@]}" &
TRAIN_PID=$!

start_disk_monitor "$TRAIN_PID"
trap 'kill "$TRAIN_PID" "$MONITOR_PID" 2>/dev/null || true' INT TERM
wait "$TRAIN_PID"
TRAIN_RC=$?
kill "$MONITOR_PID" 2>/dev/null || true
echo ">> training exited with code $TRAIN_RC"
exit "$TRAIN_RC"
