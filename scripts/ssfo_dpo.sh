#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E4B"
SFT_MODEL_PATH="${USER}/"

# Training Parameters
# Training Parameters
BATCH_SIZE=8
AUTO_FIND_BATCH_SIZE=True
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=5e-6
BETA=0.1
WEIGHT_DECAY=0.0
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.0
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1
MAX_LENGTH=512
MAX_PROMPT_LENGTH=384
EVAL_STRATEGY="epoch"
SAVE_STRATEGY="epoch"

# Infrastructure Parameters
DEEPSPEED_CONFIG="scripts/deepspeed_config.yaml"

# Parameters derived from above
TASK_NAME="$1"
DATASET_REPO_ID="${USER}/${TASK_NAME}_ssfo_preference"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')
RUN_IDENTIFIER="${USER}/${TASK_NAME}_SSFO_DPO_${MODEL_NAME}_S${SEED}_epo${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_beta${BETA}_${TIMESTAMP}"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name: $TASK_NAME"
    echo "Valid choices: npov, bosch, ragtruth"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  src/dpo.py \
  --task_name "$TASK_NAME" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK_NAME}/ssfo_dpo/${RUN_IDENTIFIER}" \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --seed "$SEED" \
  --dataset_repo_id "${DATASET_REPO_ID}" \
  --model_repo_id "${MODEL_REPO_ID}" \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --do_train True \
  --bf16 True \
  --do_eval True \
  --eval_strategy "$EVAL_STRATEGY" \
  --save_strategy "$SAVE_STRATEGY" \
  --load_best_model_at_end True \
  --metric_for_best_model "loss" \
  --greater_is_better False \
  --save_total_limit 1 \
  --beta "$BETA" \
  --max_length "$MAX_LENGTH" \
  --max_prompt_length "$MAX_PROMPT_LENGTH" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --weight_decay "$WEIGHT_DECAY" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --auto_find_batch_size "$AUTO_FIND_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --eval_accumulation_steps 1 \
  --peft_type "LORA" \
  --task_type "CAUSAL_LM" \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT"
