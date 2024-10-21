#!/bin/bash

# ============================================================
# Bash Script for Training Time-LLM on Sinusoidal Dataset
# ============================================================


# ------------------------------------------------------------
# Define Variables
# ------------------------------------------------------------

model_name=TimeLLM
comment='TimeLLM-Sinusoidal'

# LLM Configuration
llama_layers=4               # Reduced layers for simpler data
llm_dim=4096                # Dimension of LLM embeddings

# Training Configuration
learning_rate=0.001          # Optimizer learning rate

# Hardware Configuration
master_port=29500            # Master port (ensure it's free; avoid leading zeros)
num_process=2                # Number of processes (GPUs)

# Batch and Model Parameters
batch_size=12                # Per-GPU batch size
d_model=32
d_ff=64

# ------------------------------------------------------------
# Define Prediction Lengths and Corresponding Parameters
# ------------------------------------------------------------

# Arrays for varying prediction lengths, model dimensions, and training epochs
pred_lens=(10 20 30 40)      # Example prediction lengths for sine wave forecasting
d_ffs=(64 64 128 128)        # Corresponding d_ff values
train_epochs_arr=(20 20 20 30) # Corresponding training epochs

# ------------------------------------------------------------
# Loop Through Different Configurations and Launch Training
# ------------------------------------------------------------

for i in "${!pred_lens[@]}"; do
    pred_len=${pred_lens[$i]}
    d_ff=${d_ffs[$i]}
    epochs=${train_epochs_arr[$i]}
    
    echo "------------------------------------------------------------"
    echo "Starting Training Run: pred_len=$pred_len, d_ff=$d_ff, epochs=$epochs"
    echo "------------------------------------------------------------"
    
    accelerate launch --mixed_precision bf16 \
      --num_processes $num_process \
      --main_process_port $master_port \
      run_main.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/sinusoidal/ \
      --data_path sinusoidal.csv \
      --model_id sinusoidal_512_${pred_len} \
      --model $model_name \
      --data sinusoidal \
      --features S \
      --seq_len 512 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --llm_layers $llama_layers \
      --train_epochs $epochs \
      --model_comment "$comment" \
      --target value
done
