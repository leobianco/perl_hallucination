#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E4B"
SFT_MODEL_PATH="${USER}/"

# SSFO Generation Parameters (https://arxiv.org/abs/2508.17225)
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
MAX_NEW_TOKENS=150
USE_GROUND_TRUTH_CHOSEN=False
MAX_SAMPLES=""

# Parameters derived from above
TASK_NAME="$1"
DATASET_REPO_ID="${USER}/${TASK_NAME}_sft"
OUTPUT_DATASET_REPO_ID="${USER}/${TASK_NAME}_ssfo_preference"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name: $TASK_NAME"
    echo "Valid choices: npov, bosch, ragtruth"
    exit 1
fi

CMD_ARGS=(
  --task_name "$TASK_NAME"
  --seed "$SEED"
  --dataset_repo_id "$DATASET_REPO_ID"
  --model_repo_id "${MODEL_REPO_ID}"
  --sft_model_path "${SFT_MODEL_PATH}"
  --output_dataset_repo_id "$OUTPUT_DATASET_REPO_ID"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --top_k "$TOP_K"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --use_ground_truth_chosen "$USE_GROUND_TRUTH_CHOSEN"
  --push_to_hub True
)

if [ -n "$MAX_SAMPLES" ]; then
  CMD_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python3 -m src.ssfo_data_generation "${CMD_ARGS[@]}"
