#!/bin/bash

set -e

source scripts/PatchTST/launcher.sh
trap cleanup_jobs SIGINT SIGTERM

tasks=(
    "scripts/PatchTST/main/etth1.sh|5"
    "scripts/PatchTST/main/etth2.sh|5"
    "scripts/PatchTST/main/ettm1.sh|0.001"
    "scripts/PatchTST/main/ettm2.sh|0.05"
    "scripts/PatchTST/main/weather.sh|0.05"
    "scripts/PatchTST/main/electricity.sh|0.01"
    "scripts/PatchTST/main/traffic.sh|0.01"
)

echo "Launching DeltaTSF best-setting experiments on GPUs ${START_GPU}-${END_GPU}."
echo "Set START_GPU, END_GPU, or MAX_PARALLEL to override scheduling."
echo "------------------------------------------------"

for task in "${tasks[@]}"; do
    IFS="|" read -r script_path weight <<< "$task"
    launch_weighted_task "$script_path" "$weight"
done

wait_for_all_tasks

echo "------------------------------------------------"
echo "All DeltaTSF best-setting experiments finished."
