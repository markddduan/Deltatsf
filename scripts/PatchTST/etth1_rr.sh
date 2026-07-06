# ================= 1. 接收外部 w_diff 参数 =================
w_diff=$1
if [ -z "$w_diff" ]; then
    w_diff=5.0  # 默认值
fi

# ================= 2. 基础参数配置 =================
seq_len=720      # 统一改为 720
model_name=PatchTST
root_path_name=/media/D1/temp22/dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2021
enc_in=14        # 原为 7，翻倍至 14

# ================= 3. 动态创建分层日志路径 =================
# 格式：./logs/ETTh1/参数值/长度.log
save_dir="./logs/${model_id_name}/${w_diff}"
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

# ================= 4. 循环运行 4 个预测长度 =================
for pred_len in 96 192 336 720
do
    echo ">>>> Running ${model_id_name} | w_diff: ${w_diff} | pred_len: ${pred_len} <<<<"

    # 注意：在 8 卡并行的总控脚本下，这里固定为 --devices 0
    # 因为总控脚本会通过 CUDA_VISIBLE_DEVICES 限制每个进程只能看到一张物理显卡
    python -u run_longExp.py \
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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 100 \
      --w_diff1 "$w_diff" \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --use_gpu True \
      --devices "0" \
      --rand_replace 1 \
      > "${save_dir}/${pred_len}.log" 
done