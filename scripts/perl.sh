#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E2B-it"
SFT_MODEL_PATH="${USER}/"
REWARD_MODEL_PATH="${USER}/"

# Training Parameters
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-5
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1
BETA=1e-4
MAX_COMPLETION_LENGTH=256
NUM_GENERATIONS=4
NUM_ITERATIONS=1
STEPS_PER_GENERATION=16
PER_DEVICE_BATCH_SIZE=8
AUTO_FIND_BATCH_SIZE=False
TEMPERATURE=0.1
SAVE_STRATEGY="epoch"
DO_EVAL=True
EVAL_STRATEGY="steps"
EVAL_STEPS=10
EVAL_ON_START=True

# Infrastructure Parameters
DEEPSPEED_CONFIG="scripts/deepspeed_config.yaml"
SHUTDOWN=false

# Parameters derived from above
TASK_NAME="$1"
TIMESTAMP=$(date '+%y%m%d%H%M')
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
RUN_IDENTIFIER="${USER}/${TASK_NAME}_PERL_${MODEL_NAME}_S${SEED}_epo${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_beta${BETA}_${TIMESTAMP}"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK_NAME"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  src/perl.py \
  --task_name "$TASK_NAME" \
  --seed "$SEED" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK_NAME}/perl/${RUN_IDENTIFIER}" \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "${USER}/${TASK_NAME}_perl" \
  --model_repo_id "${MODEL_REPO_ID}" \
  --do_train True \
  --do_eval "$DO_EVAL" \
  --eval_strategy "$EVAL_STRATEGY" \
  --eval_steps "$EVAL_STEPS" \
  --eval_on_start "$EVAL_ON_START" \
  --save_strategy "$SAVE_STRATEGY" \
  --load_best_model_at_end True \
  --metric_for_best_model "rewards/reward_fn/mean" \
  --greater_is_better True \
  --save_total_limit 1 \
  --save_only_model True \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --weight_decay 0.0 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --eval_accumulation_steps 1 \
  --reward_model_path "${REWARD_MODEL_PATH}" \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --beta "$BETA" \
  --num_generations "$NUM_GENERATIONS" \
  --num_iterations "$NUM_ITERATIONS" \
  --steps_per_generation "$STEPS_PER_GENERATION" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --auto_find_batch_size "$AUTO_FIND_BATCH_SIZE" \
  --temperature "$TEMPERATURE"

if [ "$SHUTDOWN" = true ]; then
  echo "Shutting down the VM..."
  sudo shutdown -h now
fi
