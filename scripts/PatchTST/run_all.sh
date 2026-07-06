#!/bin/bash

set -e

source scripts/PatchTST/launcher.sh
trap cleanup_jobs SIGINT SIGTERM

scripts=(
    "scripts/PatchTST/rr/ettm2.sh"
    "scripts/PatchTST/rr/ettm1.sh"
    "scripts/PatchTST/rr/etth1.sh"
    "scripts/PatchTST/rr/etth2.sh"
    "scripts/PatchTST/rr/weather.sh"
)

w_diff_params="1e-3 0.01 0.1 1.0 5.0"

echo "Launching DeltaTSF grid search on GPUs ${START_GPU}-${END_GPU}."
echo "Set START_GPU, END_GPU, MAX_PARALLEL, or edit w_diff_params as needed."
echo "------------------------------------------------"

for script_path in "${scripts[@]}"; do
    for weight in $w_diff_params; do
        launch_weighted_task "$script_path" "$weight"
    done
done

wait_for_all_tasks

echo "------------------------------------------------"
echo "All DeltaTSF grid-search experiments finished."
