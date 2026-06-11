#!/bin/bash
# Read-only verification of the reported GraspNet training specs.
#
# Run this to independently confirm (and capture for a report) where each
# reported number comes from and what was actually measured. It changes
# nothing and starts no training -- only grep/ls/stat/nvidia-smi/df/awk.
#
# Usage: scripts/verify_training_specs.sh
set -euo pipefail

REPO_ROOT="/workspace/tools"
DATA_ROOT="/workspace/data/graspnet"
VAL_LOG="$REPO_ROOT/runs/scratch_realsense/log_train.txt"
VAL_CKPT="$REPO_ROOT/runs/scratch_realsense/checkpoint.tar"
BATCH=4
IMGS_PER_SCENE=256
VAL_BATCHES=1920   # validation run: 30 scenes x 256 / 4

line() { printf '%s\n' "------------------------------------------------------------"; }

echo "############ GraspNet training-spec verification (read-only) ############"

line
echo "[1] ACTUAL applied config (recorded by the validation run)"
echo "    source: $VAL_LOG"
if [[ -f "$VAL_LOG" ]]; then
    grep -aoE "Namespace\(.*\)" "$VAL_LOG" | head -1 | tr ',' '\n' \
        | grep -E "batch_size|learning_rate|num_point|num_view|num_workers|max_epoch|camera|scenes|skip_eval|lr_decay" \
        | sed 's/^[ ]*/    /'
else
    echo "    (validation log not found -- run the 1-epoch validation first)"
fi

line
echo "[2] MEASURED speed (validation: $VAL_BATCHES batches, skip_eval)"
if [[ -f "$VAL_LOG" && -f "$VAL_CKPT" ]]; then
    start_s=$(grep -aoE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" "$VAL_LOG" | head -1)
    end_s=$(stat -c '%y' "$VAL_CKPT" | cut -d'.' -f1)
    s1=$(date -d "$start_s" +%s 2>/dev/null || echo 0)
    s2=$(date -d "$end_s" +%s 2>/dev/null || echo 0)
    el=$(( s2 - s1 ))
    echo "    epoch start : $start_s"
    echo "    ckpt mtime  : $end_s"
    if (( el > 0 )); then
        awk -v b="$VAL_BATCHES" -v e="$el" 'BEGIN{
            printf "    elapsed     : %d s (%.1f min)\n", e, e/60
            printf "    speed       : %.3f batch/s\n", b/e }'
    fi
else
    echo "    (need both validation log and checkpoint to measure)"
fi

line
echo "[3] DATA scale"
n_train=$(ls -d "$DATA_ROOT"/scenes/scene_00[0-9][0-9] 2>/dev/null | wc -l)
n_all=$(ls -d "$DATA_ROOT"/scenes/scene_* 2>/dev/null | wc -l)
echo "    train scenes (0000-0099): $n_train"
echo "    total scenes            : $n_all"
awk -v s="$n_train" -v i="$IMGS_PER_SCENE" -v b="$BATCH" 'BEGIN{
    printf "    samples = %d x %d = %d\n", s, i, s*i
    printf "    batch/epoch = %d / %d = %d\n", s*i, b, s*i/b }'
echo "    grasp_label npz : $(ls "$DATA_ROOT"/grasp_label/*.npz 2>/dev/null | wc -l)"
echo "    collision scenes: $(ls -d "$DATA_ROOT"/collision_label/scene_* 2>/dev/null | wc -l)"
echo "    tolerance npy   : $(ls "$REPO_ROOT"/graspnet/dataset/tolerance/*.npy 2>/dev/null | wc -l)"

line
echo "[4] GPU status (live)"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/    GPU /'
echo "    -- other users' compute processes --"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo "    free disk: $(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')G (floor 80G)"

line
echo "[5] VRAM evidence"
echo "    During the validation run GPU2 total was ~19.2G with a co-tenant"
echo "    using ~3.8G  ->  our process ~= 15.4G (training, not inference)."
echo "    Verify live during real training:  watch -n2 nvidia-smi"
echo "    (smoke-test inference was only ~1.9G; training is much larger.)"

line
echo "[6] External sources for each reported number"
echo "    batch_size = 4        -> PAPER (repo default is 2):"
echo "      https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf"
echo "    max_epoch=18, lr_decay @8,12,16 -> OFFICIAL repo train.py defaults:"
echo "      https://github.com/graspnet/graspnet-baseline/blob/main/train.py"
echo "    lr=0.001, num_point=20000, num_view=300 -> both agree."

line
echo "[7] Total-time estimate (full training, 100 train scenes)"
if [[ -f "$VAL_LOG" && -f "$VAL_CKPT" && "${el:-0}" -gt 0 ]]; then
    awk -v s="$n_train" -v i="$IMGS_PER_SCENE" -v b="$BATCH" -v vb="$VAL_BATCHES" -v e="$el" 'BEGIN{
        rate = vb/e
        bpe  = s*i/b
        train_h = bpe/rate/3600
        eval_h  = vb/rate/3600
        per = train_h + eval_h
        printf "    measured rate     : %.3f batch/s (num_workers=2)\n", rate
        printf "    train/epoch       : %d batch -> %.1f h\n", bpe, train_h
        printf "    eval(test_seen)/ep: %d batch -> %.1f h\n", vb, eval_h
        printf "    ~per epoch        : %.1f h\n", per
        printf "    x18 epochs        : %.1f h  (~%.1f days) at nw=2\n", per*18, per*18/24
        printf "    nw=8 (est. ~2-3x) : ~%.1f-%.1f days\n", per*18/24/3, per*18/24/2 }'
else
    echo "    (need validation timing to estimate)"
fi

line
echo "NOTE: this script started NO training and modified nothing."
echo "############################# end ######################################"
