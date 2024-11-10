#!/bin/bash

# Set model and training parameters
model_name=TimeLLM
train_epochs=10
learning_rate=0.01
llama_layers=2

master_port=29500
num_process=8
batch_size=8
d_model=16
d_ff=32
OMP_NUM_THREADS=64
comment='TimeLLM-ECL'
export OMP_NUM_THREADS=$OMP_NUM_THREADS

# Set the target feature you want to forecast
target_feature='OT'  # Replace with the actual feature name

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512_96 \
  --model $model_name \
  --data ECL \
  --features MS \
  --seq_len 512 \
  --label_len 8 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 3 \
  --dec_in 1 \
  --c_out 1 \
  --target $target_feature \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --percent 100 \
  --model_comment $comment \
  --content "Sinusoid" \
