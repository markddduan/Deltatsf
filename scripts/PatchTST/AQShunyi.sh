if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=720
model_name=PatchTST

root_path_name=/media/D1/temp22/dataset/
data_path_name=AQShunyi.csv
model_id_name=AQShunyi
data_name=custom

random_seed=2021
for pred_len in 96 192 336 720
#for pred_len in 192 
do
    #export CUDA_VISIBLE_DEVICES=4,5,6,7
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 22 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 30\
      --patience 5\
      --gpu 3 \
      --lradj 'TST'\
      --pct_start 0.2\
      --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done
#11
'''
--use_multi_gpu \
      --devices 0,1,2,3 \