#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables for paths to make the script easier to modify
CHECKPOINT_PATH="/ceph/home/student.aau.dk/xx06av/AI1/checkpoints/long_term_forecast_ECL_512_96_TimeLLM_ECL_ftMS_sl512_ll8_pl96_dm16_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-ECL/checkpoint.pth"
OUTPUT_PLOT="forecasting_results.png"  # Base name; script will append '_OT.png'
OUTPUT_CSV="inference_results.csv"
DATA_PATH="/ceph/home/student.aau.dk/xx06av/AI1/dataset/electricity/electricity.csv"
CSV_NAME="try_3_pl128_dm32"
# Set pred_len to match the training configuration (e.g., 96)
PRED_LEN=96
# Set batch size (recommend setting to 1 or a manageable number)
BATCH_SIZE=1
num_process=8
master_port=29500

# Execute the inference script with all necessary parameters
python run_inference.py \
  --model TimeLLM \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --data ECL \
  --features MS \
  --seq_len 512 \
  --label_len 64 \
  --pred_len "$PRED_LEN" \
  --patch_len 128 \
  --enc_in 3 \
  --dec_in 1 \
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
  --llm_layers 8 \
  --llm_model LLAMA \
  --prompt_domain 1 \
  --content "The output is a sinusoid" \
  --stride 4 \
  --dropout 0 \
  --task_name long_term_forecast \
  --batch_size "$BATCH_SIZE" \
  --save_plot_path "$OUTPUT_PLOT" \
  --seasonal_patterns Monthly \
  --device cuda \
  --csv_name "$CSV_NAME" 


# Optional: Print completion message
echo "Inference completed successfully. Results saved to $OUTPUT_PLOT and $CSV_NAME."
