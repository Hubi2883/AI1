#!/bin/bash
# Unified script for training and inference

data_path="/ceph/home/student.aau.dk/xx06av/AI1/dataset/electricity/electricity.csv"
output_plot="forecasting_results.png"
output_csv="inference_results.csv"

checkpoint_path="/ceph/home/student.aau.dk/xx06av/AI1/checkpoints/long_term_forecast_ECL_512_96_TimeLLM_ECL_ftMS_sl512_ll32_pl1000_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-ECL/checkpoint.pth"

# Set common parameters
model_name=TimeLLM
master_port=29500
num_process=8
comment='TimeLLM-ECL'
features='MS'
target_feature='OT'
content="The output is a sinusoid"
OMP_NUM_THREADS=64

# Model training
train_epochs=3
learning_rate=0.01

# Important Parameters
llama_layers=3
batch_size=4
d_model=16
d_ff=32
stride=4

# Lengths
seq_len=512
pred_len=1000
patch_len=128

# Layer sizes
enc_in=3
dec_in=1
c_out=1
factor=3
e_layers=2
d_layers=1



csv_name="try_71"
RUN_MODE="I"  # I or T for inference or training

# Export environment variables
export OMP_NUM_THREADS=$OMP_NUM_THREADS
export RUN_MODE=$RUN_MODE

# Select mode
if [ "$RUN_MODE" == "T" ]; then
  echo "Running in TRAINING mode..."
  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path $data_path \
    --model_id ECL_512_96 \
    --model $model_name \
    --data ECL \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --stride $stride \
    --c_out $c_out \
    --d_ff $d_ff \
    --d_model $d_model \
    --label_len 32 \
    --target $target_feature \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --percent 100 \
    --model_comment $comment \
    --content "$content"

elif [ "$RUN_MODE" == "I" ]; then
  echo "Running in INFERENCE mode..."
  python run_inference.py \
    --model $model_name \
    --checkpoint_path "$checkpoint_path" \
    --data_path "$data_path" \
    --data ECL \
    --features $features \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --c_out $c_out \
    --d_model $d_model \
    --n_heads 8 \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --d_ff $d_ff \
    --factor $factor \
    --embed timeF \
    --freq h \
    --target $target_feature \
    --num_workers 10 \
    --label_len 32 \
    --percent 100 \
    --llm_dim 4096 \
    --llm_layers $llama_layers \
    --llm_model LLAMA \
    --prompt_domain 1 \
    --content "$content" \
    --stride $stride \
    --dropout 0 \
    --task_name long_term_forecast \
    --batch_size $batch_size \
    --save_plot_path "$output_plot" \
    --seasonal_patterns Monthly \
    --device cuda \
    --csv_name "$csv_name"
else
  echo "Invalid RUN_MODE selected. Please set RUN_MODE to either 'TRAINING' or 'INFERENCE'."
  exit 1
fi
