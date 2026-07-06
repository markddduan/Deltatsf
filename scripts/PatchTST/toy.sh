# 确保数据文件名正确
data_file_name=nonlinear_trend_v2.csv

python -u run_longExp.py \
  --random_seed 2021 \
  --is_training 1 \
  --root_path /media/D1/temp22/dataset/ \
  --data_path $data_file_name \
  --model_id 'Baseline_nonlinear' \
  --model PatchTST \
  --data custom \
  --features S \
  --target OT \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 1 \
  --c_out 1 \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 16 \
  --d_ff 128 \
  --dropout 0.3 \
  --fc_dropout 0.3 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp_Baseline' \
  --train_epochs 20 \
  --patience 5 \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.001 >logs/LongForecasting/'Baseline_Nonlinear.log' 2>&1