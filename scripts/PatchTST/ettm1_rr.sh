# ================= 1. 接收总控脚本传入的参数 =================
w_diff=$1
if [ -z "$w_diff" ]; then
    w_diff=0.001  # 默认值
fi

# ================= 2. 基础参数配置 =================
seq_len=720      # 已改为 720
model_name=PatchTST
root_path_name=/media/D1/temp22/dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
random_seed=2021
enc_in=14        # 已由 7 翻倍至 14

# ================= 3. 动态创建分层日志路径 =================
# 路径格式：./logs/ETTm1/0.01/
save_dir="./logs_720/${model_id_name}/${w_diff}"
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

# ================= 4. 循环运行 4 个预测长度 =================
for pred_len in 96 192 336 720
do
    echo ">>>> Running ${model_id_name} | w_diff: ${w_diff} | pred_len: ${pred_len} <<<<"

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
      > "${save_dir}/${pred_len}.log" 
done


  
   