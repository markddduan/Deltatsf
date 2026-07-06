#!/bin/bash

script_dir="./scripts/difftst_best"
summary_file="final_best_metrics.txt"
log_root="./logs/Best_Final"

target_scripts=(
    # "etth1.sh"
    # "etth2.sh"
    # "ettm1.sh"
    # "ettm2.sh"
    "weather.sh"
    # "ECL.sh"
    # "traffic.sh"
)

gpu_start=4
gpu_end=7
max_parallel=4

echo "Final Best-Weight Results Summary - $(date)" > $summary_file
echo "====================================================" >> $summary_file

gpu_idx=$gpu_start
current_jobs=0

echo ">>>>>>> Starting Multi-GPU Task Scheduling <<<<<<<"

for script_name in "${target_scripts[@]}"
do
    script_path="${script_dir}/${script_name}"
    
    if [ ! -f "$script_path" ]; then
        echo "Warning: $script_path not found, skipping..."
        continue
    fi

    echo "[GPU $gpu_idx] Launching $script_name..."

    CUDA_VISIBLE_DEVICES=$gpu_idx bash "$script_path" &
    
    ((current_jobs++))
    
    if [ $gpu_idx -eq $gpu_end ]; then
        gpu_idx=$gpu_start
    else
        ((gpu_idx++))
    fi

    if [ $current_jobs -ge $max_parallel ]; then
        wait
        current_jobs=0
    fi
done

wait
echo ">>>>>>> All experiments finished! Starting results collection... <<<<<<<"

for ds_dir in $(ls "$log_root"); do
    if [ -d "${log_root}/${ds_dir}" ]; then
        echo -e "\nDataset: $ds_dir" >> $summary_file
        echo "-------------------------------" >> $summary_file
        
        for len in 96 192 336 720; do
            log_path="${log_root}/${ds_dir}/${len}.log"
            if [ -f "$log_path" ]; then
                metrics=$(grep -E "mse|mae" "$log_path" | tail -n 1)
                if [ ! -z "$metrics" ]; then
                    echo "Length ${len}: $metrics" >> $summary_file
                else
                    echo "Length ${len}: No metrics found." >> $summary_file
                fi
            fi
        done
    fi
done

echo ">>>>>>> Collection Complete! Results saved in: $summary_file <<<<<<<"