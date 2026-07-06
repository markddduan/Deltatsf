#!/bin/bash

# Stop child GPU jobs cleanly when the launcher is interrupted.
trap 'cleanup' SIGINT SIGTERM

cleanup() {
    echo -e "\n\n[Interrupted] Stopping all launched GPU jobs..."
    pids=$(jobs -p)
    if [ -n "$pids" ]; then
        kill $pids 2>/dev/null
    fi
    echo "All jobs stopped. Exiting."
    exit 1
}

# Main DeltaTSF + PatchTST reproduction tasks.
tasks=(
    "scripts/PatchTST/etth1.sh|5"
    "scripts/PatchTST/etth2.sh|5"
    "scripts/PatchTST/ettm1.sh|0.001"
    "scripts/PatchTST/ettm2.sh|0.05"
    "scripts/PatchTST/weather.sh|0.05"
)

gpu_id=${START_GPU:-0}

echo "Launching DeltaTSF tasks. Press Ctrl+C to stop all jobs."
echo "------------------------------------------------"

for task in "${tasks[@]}"; do
    IFS="|" read -r script_path weight <<< "$task"
    data_name=$(basename "$script_path" .sh)

    echo "[Launch] $data_name -> GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id bash "$script_path" "$weight" &

    gpu_id=$((gpu_id + 1))
    if [ $gpu_id -gt ${END_GPU:-7} ]; then
        gpu_id=${START_GPU:-0}
    fi
done

wait

echo "------------------------------------------------"
echo "All DeltaTSF tasks finished."
