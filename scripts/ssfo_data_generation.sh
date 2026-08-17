#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E4B-it"
SFT_MODEL_PATH="${USER}/"

# SSFO Generation Parameters (matching evaluator.sh)
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=0
MAX_NEW_TOKENS=250
BATCH_SIZE=16
MAX_SAMPLES=""  # Set number of samples here directly (e.g. 500), or leave empty for full data
USE_GROUND_TRUTH_CHOSEN=False

# Parameters derived from above
TASK_NAME="$1"
if [ -n "$2" ] && [ "$2" != "--max_samples" ]; then
  MAX_SAMPLES="$2"
elif [ "$2" == "--max_samples" ] && [ -n "$3" ]; then
  MAX_SAMPLES="$3"
fi

DATASET_REPO_ID="${DATASET_REPO_ID:-${USER}/${TASK_NAME}_perl}"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $NF}')
TIMESTAMP=$(date '+%y%m%d%H%M')

SAMPLES_TAG=""
if [ -n "$MAX_SAMPLES" ]; then
  SAMPLES_TAG="_n${MAX_SAMPLES}"
fi
OUTPUT_DATASET_REPO_ID="${USER}/${TASK_NAME}_ssfo_preference_${MODEL_NAME}${SAMPLES_TAG}_${TIMESTAMP}"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name: $TASK_NAME"
    echo "Usage: ./scripts/ssfo_data_generation.sh <task_name> [max_samples]"
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
  --batch_size "$BATCH_SIZE"
  --use_ground_truth_chosen "$USE_GROUND_TRUTH_CHOSEN"
  --push_to_hub True
)

if [ -n "$MAX_SAMPLES" ]; then
  CMD_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python3 -m src.ssfo_data_generation "${CMD_ARGS[@]}"
