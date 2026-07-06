export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /media/D1/temp22/dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --w_diff1 1.0 \
  --itr 1 \
  --top_k 5 


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /media/D1/temp22/dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --w_diff1 1.0 \
  --itr 1 \
  --top_k 5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /media/D1/temp22/dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --w_diff1 1.0 \
  --itr 1 \
  --top_k 5


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /media/D1/temp22/dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --w_diff1 1.0 \
  --itr 1 \
  --top_k 5
