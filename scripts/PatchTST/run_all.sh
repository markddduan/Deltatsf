#!/bin/bash
# 当脚本退出时（包括 Ctrl+C），自动杀死所有子进程
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
# ================= 1. 配置区域 =================
scripts=(
    "scripts/PatchTST/ettm2_rr.sh"
    "scripts/PatchTST/ettm1_rr.sh"
    # "scripts/PatchTST/electricity.sh"
    # "scripts/PatchTST/weather_rr.sh"
    # "scripts/PatchTST/etth1_rr.sh"
    # # "scripts/PatchTST/etth2_rr.sh"
    # "scripts/PatchTST/exchange.sh"
    
)

# 网格搜索范围
w_diff_params="1e-3 0.01 0.1 1.0 5.0"
#w_diff_params="6.0 7.0 8.0 9.0 10.0 20.0"
# --- 关键修改：定义使用的 GPU 范围 ---
gpu_start=0
gpu_end=7
max_parallel=8 # 实际使用的 GPU 数量 (1,2,3,4,5,6,7)

# ================= 2. 运行逻辑 =================
gpu_idx=$gpu_start  # 从 1 号卡开始
current_jobs=0

echo ">>>>>>> Starting Grid Search on GPUs $gpu_start to $gpu_end <<<<<<<"

for script in "${scripts[@]}"
do
    for wd in $w_diff_params
    do
        if [ ! -f "$script" ]; then
            echo "Warning: $script not found, skipping..."
            continue
        fi

        echo "[GPU $gpu_idx] Running $script with w_diff1=$wd"

        # 使用当前的 gpu_idx 锁定显卡
        CUDA_VISIBLE_DEVICES=$gpu_idx bash "$script" "$wd" &
        
        ((current_jobs++))
        
        # --- 关键修改：索引在 1-7 之间轮转 ---
        # 如果 gpu_idx 到了 7，就跳回 1
        if [ $gpu_idx -eq $gpu_end ]; then
            gpu_idx=$gpu_start
        else
            ((gpu_idx++))
        fi

        # 当后台任务达到 7 个（即所有指定显卡都占满）时进行等待
        if [ $current_jobs -ge $max_parallel ]; then
            wait
            current_jobs=0
            echo ">>>>>>> Batch on GPUs $gpu_start-$gpu_end completed. <<<<<<<"
        fi
    done
done

wait
echo ">>>>>>> All grid search experiments are finished! <<<<<<<"