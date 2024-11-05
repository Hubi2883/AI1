#!/bin/bash

# Exit immediately if a command exits with a non-zero status
#set -e

# Define variables for paths to make the script easier to modify
CHECKPOINT_PATH="/ceph/home/student.aau.dk/wb68dm/AI1/checkpoints/long_term_forecast_ECL_512_96_TimeLLM_ECL_ftM_sl512_ll8_pl336_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-ECL/checkpoint.pth"
DATA_PATH="/ceph/home/student.aau.dk/wb68dm/AI1/dataset/electricity/synthetic2.csv"
OUTPUT_PLOT="anomaly_detection_plots11.png"

# Run the inference script with all necessary parameters
python run_inference.py \
  --model TimeLLM \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --data ECL \
  --features M \
  --seq_len 512 \
  --label_len 8 \
  --pred_len 336 \
  --batch_size 8 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --d_model 16 \
  --n_heads 8 \
  --e_layers 2 \
  --d_layers 1 \
  --d_ff 32 \
  --factor 3 \
  --embed timeF \
  --freq h \
  --target OT \
  --num_workers 10 \
  --percent 100 \
  --llm_dim 4096 \
  --llm_layers 2 \
  --llm_model LLAMA \
  --prompt_domain 1 \
  --content "This should be just straight line equal to 100!" \
  --patch_len 16 \
  --stride 8 \
  --dropout 0.1 \
  --task_name long_term_forecast \
  --save_plot_path "$OUTPUT_PLOT"
