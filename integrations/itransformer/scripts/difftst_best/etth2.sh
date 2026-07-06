#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

model_name=iTransformer
root_path_name=/media/D1/temp22/dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
seq_len=720
enc_in=14

save_dir="./logs/Best_Final/${model_id_name}"
mkdir -p "$save_dir"

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_96_96_best \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $enc_in \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --w_diff1 0.5 \
  --itr 1 > "${save_dir}/96.log" 2>&1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_96_192_best \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $enc_in \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --w_diff1 0.01 \
  --itr 1 > "${save_dir}/192.log" 2>&1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_96_336_best \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $enc_in \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --w_diff1 0.05 \
  --itr 1 > "${save_dir}/336.log" 2>&1

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_96_720_best \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in $enc_in \
  --dec_in $enc_in \
  --c_out $enc_in \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --w_diff1 0.1 \
  --itr 1 > "${save_dir}/720.log" 2>&1