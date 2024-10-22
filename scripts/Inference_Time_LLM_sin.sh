#!/bin/bash

# Define the required arguments
TASK_NAME="long_term_forecast"
IS_TRAINING=0  # Set to 0 since it's for inference, not training
MODEL_ID="test"
MODEL_COMMENT="TimeLLM_Inference"
MODEL_NAME="TimeLLM"  # Specify the TimeLLM model
DATA="sinusoidal"  # Dataset you are using

# Paths
ROOT_PATH="./dataset"  # Path to dataset root
DATA_PATH="sinusoidal/sinusoidal.csv"  # Dataset CSV file path
OUTPUT_WEIGHTS_PATH="/ceph/home/student.aau.dk/xx06av/AI1/output_weights/model_weights_epoch_19_predlen_10.pth"  # Adjust this path to your saved weights

# Model parameters
SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96
EVAL_BATCH_SIZE=8
LLM_MODEL="LLAMA"  # Specify LLM model (LLAMA, GPT2, etc.)
LLM_DIM=4096  # Dimension of LLM
TARGET="value"  # Make sure this is the correct target column in your dataset

# Run the inference Python script
python inference.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --model_id $MODEL_ID \
  --model_comment $MODEL_COMMENT \
  --model $MODEL_NAME \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --eval_batch_size $EVAL_BATCH_SIZE \
  --llm_model $LLM_MODEL \
  --llm_dim $LLM_DIM \
  --target $TARGET
