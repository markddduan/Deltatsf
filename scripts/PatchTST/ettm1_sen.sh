#!/bin/bash

# ================= 0. 优雅退出机制 (Ctrl+C 触发) =================
# 自动杀死所有后台挂起的 Python 子进程
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# ================= 1. 基础参数配置 =================
seq_len=336      
model_name=PatchTST
root_path_name=/media/D1/temp22/dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
random_seed=2021
enc_in=14        

# ================= 2. 🎯 精准定位缺失的 8 个实验 =================
# 格式为 "w_diff pred_len"
missing_tasks=(
    "0.1 336"
    "0.1 720"
    "1.0 192"
    "1.0 336"
    "1.0 720"
    "5.0 192"
    "5.0 336"
    "5.0 720"
)

# ================= 3. GPU 调度配置 =================
gpu_start=1
gpu_end=7
gpu_idx=$gpu_start
current_jobs=0
max_parallel=7

echo ">>>> 开始执行缺失的 8 个实验，自动分配至 GPU 1-7... <<<<"

# ================= 4. 循环派发任务 =================
for task in "${missing_tasks[@]}"
do
    # 解析当前的 w_diff 和 pred_len
    read -r w_diff pred_len <<< "$task"
    
    # 动态创建对应的日志文件夹
    save_dir="./logs/${model_id_name}/${w_diff}"
    if [ ! -d "$save_dir" ]; then
        mkdir -p "$save_dir"
    fi

    echo ">>>> 派发任务 | w_diff: ${w_diff} | pred: ${pred_len} | 目标显卡: GPU ${gpu_idx} <<<<"

    # 提交到后台运行
    CUDA_VISIBLE_DEVICES=$gpu_idx python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id "${model_id_name}_${seq_len}_${pred_len}_wd${w_diff}" \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in $enc_in \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2 \
      --fc_dropout 0.2 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 50 \
      --patience 5 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --w_diff1 "$w_diff" \
      --itr 1 --batch_size 128 --learning_rate 0.0001 \
      --use_gpu True \
      --devices "0" \
      --rand_replace 1 \
      > "${save_dir}/${pred_len}.log" &
      
    ((current_jobs++))
    
    # 显卡 1-7 轮换逻辑
    if [ $gpu_idx -eq $gpu_end ]; then
        gpu_idx=$gpu_start
    else
        ((gpu_idx++))
    fi

    # 如果达到最大并行数，等待这一批跑完
    if [ $current_jobs -ge $max_parallel ]; then
        wait
        current_jobs=0
    fi
done

# 等待最后剩下在跑的任务全部执行完毕
wait
echo ">>>> 所有缺失的实验已全部补齐！ <<<<"