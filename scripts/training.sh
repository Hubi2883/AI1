#!/bin/bash
# This is the end of the update
# Set model and training parameters
model_name=TimeLLM
train_epochs=5
learning_rate=0.1
llama_layers=5
master_port=29500
num_process=8
batch_size=3
d_model=16
d_ff=32
OMP_NUM_THREADS=64
comment='TimeLLM-ECL'
export OMP_NUM_THREADS=$OMP_NUM_THREADS
export RUN_MODE="TRAINING"

export MASTER_ADDR="localhost"
export MASTER_PORT=29500
export WORLD_SIZE=8


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
  --features S \
  --seq_len 512 \
  --label_len 32 \
  --pred_len 1000 \
  --e_layers 4 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 3 \
  --enc_in 3 \
  --dec_in 1 \
  --stride 4 \
  --c_out 1 \
  --d_ff $d_ff \
  --d_model $d_model \
  --target $target_feature \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --percent 100 \
  --model_comment $comment \
  --content "Sinusoid" \
  --patch_len 128\