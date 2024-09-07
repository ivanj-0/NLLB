#!/bin/bash

# Define variables
CONDA_ENV="itv2"
MODEL_PATH="nllb-200-distilled-600M/"
RUN="V1_600M"
BASE_SAVE_PATH="/mnt/disks/disk1/${RUN}/"
LOG_FILE="${BASE_SAVE_PATH}${RUN}.log"
TRAIN="train_data(V1).xlsx"
DEV="validation_data.xlsx"
TEST="test_data.xlsx"
WARMUP_STEPS=1000
TRAINING_STEPS=200_001
EARLY_STOPPING_STEPS=2000
TOP_N_MODELS=40

# Create the base save path directory if it doesn't exist
mkdir -p $BASE_SAVE_PATH

# Create an empty log file if it doesn't exist
touch $LOG_FILE

# Print the tail command to follow the log file
echo "tail -f ${LOG_FILE}"

# Run the Python script in the background with nohup
nohup python main.py \
  --train $TRAIN \
  --dev $DEV \
  --test $TEST \
  --warmup_steps $WARMUP_STEPS \
  --training_steps $TRAINING_STEPS \
  --model_path $MODEL_PATH \
  --base_save_path $BASE_SAVE_PATH \
  --early_stopping_steps $EARLY_STOPPING_STEPS \
  --top_n_models $TOP_N_MODELS \
  > $LOG_FILE 2>&1 &
