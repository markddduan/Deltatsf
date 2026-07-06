w_diff=$1
if [ -z "$w_diff" ]; then
    w_diff=5.0
fi

seq_len=720
model_name=PatchTST
root_path_name=/media/D1/temp22/dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
random_seed=2021
enc_in=14

save_dir="./logs/${model_id_name}/${w_diff}"
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

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
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3 \
      --fc_dropout 0.3 \
      --head_dropout 0 \
      --patch_len 16 \
      --stride 8 \
      --des 'Exp' \
      --train_epochs 50 \
      --w_diff1 "$w_diff" \
      --itr 1 \
      --batch_size 128 \
      --learning_rate 0.0001 \
      --use_gpu True \
      --rand_replace 1 \
      --devices "0" > "${save_dir}/${pred_len}.log"
done
