w_diff=$1
subset_ratio=0.1

if [ -z "$w_diff" ]; then
    w_diff=0.01
fi

seq_len=720
model_name=PatchTST
root_path_name=/media/D1/temp22/dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
random_seed=2021
enc_in=1724

save_dir="./logs/${model_id_name}/${w_diff}"
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

for pred_len in 96 192 336 720
do
    echo ">>>> Running ${model_id_name} | w_diff: ${w_diff} | ratio: ${subset_ratio} | pred_len: ${pred_len} <<<<"

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id "${model_id_name}_${seq_len}_${pred_len}_wd${w_diff}_r${subset_ratio}" \
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
      --train_epochs 100 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.2 \
      --w_diff1 "$w_diff" \
      --subset_ratio "$subset_ratio" \
      --itr 1 --batch_size 2 --accumulation_steps 12 --learning_rate 0.0001 \
      --use_gpu True \
      --devices "0" \
      > "${save_dir}/${pred_len}.log"
done
