#!/bin/bash
# Unified script for training and inference

# Set paths
data_path="/ceph/home/student.aau.dk/tk55it/AI1/dataset/sinusoidal/synthetic_anomaly_classification_dataset.csv"
checkpoint_path="/ceph/home/student.aau.dk/tk55it/AI1/checkpoints/classification_ECL_512_96_TimeLLM_ECL_ftMS_sl2_ll512_pl32_dm1000_nh16_el8_dl4_df2_fc32_eb3_timeF_test-TimeLLM-ECL/checkpoint.pth"

# Output files
output_plot="classification_plot.png"
output_csv="inference_results_classification.csv"

# Set common parameters
model_name="TimeLLM"
master_port=29500
num_processes=8
comment="TimeLLM-ECL"
features="MS"
target_feature="target"
content="The output is a sinusoid"
OMP_NUM_THREADS=64
task_name="classification"

# Model training parameters
train_epochs=5
learning_rate=0.0001

# Important Parameters
llama_layers=4
batch_size=4
d_model=16
d_ff=32
stride=4
n_heads=8  # Added n_heads parameter

# Lengths
seq_len=512
pred_len=1000
patch_len=128
label_len=32  # Added label_len variable for consistency

# Layer sizes
enc_in=1
dec_in=1
c_out=2  # For classification with 2 classes
num_classes=2  # Number of classes for classification
factor=3
e_layers=2
d_layers=1

csv_name="try_71"
RUN_MODE="I"  # Set to "T" for training or "I" for inference

# Export environment variables
export OMP_NUM_THREADS=$OMP_NUM_THREADS
export RUN_MODE=$RUN_MODE

# Select mode
if [ "$RUN_MODE" == "T" ]; then
  echo "Running in TRAINING mode..."
  accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_processes --main_process_port $master_port run_main.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path "$data_path" \
    --model_id ECL_512_96 \
    --model "$model_name" \
    --data ECL \
    --features "$features" \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --num_classes $num_classes \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor $factor \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --stride $stride \
    --c_out $c_out \
    --d_ff $d_ff \
    --d_model $d_model \
    --task_name "$task_name" \
    --target "$target_feature" \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --percent 100 \
    --model_comment "$comment" \
    --content "$content" \
    --n_heads $n_heads \
    --dropout 0.1 \
    --embed timeF \
    --freq h \
    --llm_dim 4096 \
    --llm_model LLAMA \
    --prompt_domain 1 \
    --seasonal_patterns Monthly
elif [ "$RUN_MODE" == "I" ]; then
  echo "Running in INFERENCE mode..."
  python run_inference.py \
    --model "$model_name" \
    --checkpoint_path "$checkpoint_path" \
    --data_path "$data_path" \
    --data ECL \
    --features "$features" \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --patch_len $patch_len \
    --enc_in $enc_in \
    --dec_in $dec_in \
    --num_classes $num_classes \
    --d_model $d_model \
    --n_heads $n_heads \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --d_ff $d_ff \
    --factor $factor \
    --embed timeF \
    --freq h \
    --target "$target_feature" \
    --num_workers 10 \
    --percent 100 \
    --llm_dim 4096 \
    --llm_layers $llama_layers \
    --llm_model LLAMA \
    --prompt_domain 1 \
    --content "$content" \
    --stride $stride \
    --dropout 0.1 \
    --task_name "$task_name" \
    --batch_size $batch_size \
    --save_results_path "$output_csv" \
    --save_plot_path "$output_plot" \
    --seasonal_patterns Monthly \
    --device cuda \
    --csv_name "$csv_name"
else
  echo "Invalid RUN_MODE selected. Please set RUN_MODE to either 'T' (training) or 'I' (inference)."
  exit 1
fi
