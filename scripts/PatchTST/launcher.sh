#!/bin/bash

# Shared GPU launcher for DeltaTSF shell scripts.
# Source this file from run_best.sh or scripts/PatchTST/run_all.sh.

START_GPU=${START_GPU:-0}
END_GPU=${END_GPU:-7}
MAX_PARALLEL=${MAX_PARALLEL:-$((END_GPU - START_GPU + 1))}
GPU_ID=$START_GPU
CURRENT_JOBS=0

cleanup_jobs() {
    echo -e "\n[Interrupted] Stopping launched jobs..."
    jobs -p | xargs -r kill 2>/dev/null
    exit 1
}

next_gpu() {
    if [ "$GPU_ID" -ge "$END_GPU" ]; then
        GPU_ID=$START_GPU
    else
        GPU_ID=$((GPU_ID + 1))
    fi
}

launch_weighted_task() {
    local script_path=$1
    local weight=$2

    if [ ! -f "$script_path" ]; then
        echo "[Skip] Missing script: $script_path"
        return 0
    fi

    echo "[Launch] GPU $GPU_ID | $script_path | w_diff=$weight"
    CUDA_VISIBLE_DEVICES=$GPU_ID bash "$script_path" "$weight" &

    CURRENT_JOBS=$((CURRENT_JOBS + 1))
    next_gpu

    if [ "$CURRENT_JOBS" -ge "$MAX_PARALLEL" ]; then
        wait
        CURRENT_JOBS=0
        echo "[Batch complete]"
    fi
}

wait_for_all_tasks() {
    wait
    CURRENT_JOBS=0
}
