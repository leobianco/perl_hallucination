#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E2B-it"

# Training Parameters
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
AUTO_FIND_BATCH_SIZE=True
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=3e-3
WEIGHT_DECAY=0.0
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.0
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1
NUM_FEWSHOT=0
EVAL_STRATEGY="epoch"
SAVE_STRATEGY="epoch"

# Infrastructure Parameters
DEEPSPEED_CONFIG="scripts/deepspeed_config.yaml"

# Parameters derived from above
TASK_NAME="$1"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')
FORMATTED_LR=$(python3 -c "import sys; lr=float('${LEARNING_RATE}'); print(f'{lr:.1e}')" 2>/dev/null || echo "$LEARNING_RATE")
FORMATTED_EPOCHS=$(python3 -c "import sys; e=float('${NUM_TRAIN_EPOCHS}'); print(f'{e:.2g}')" 2>/dev/null || echo "$NUM_TRAIN_EPOCHS")
RUN_IDENTIFIER="${USER}/${TASK_NAME}_SFT_${MODEL_NAME}_S${SEED}_epo${FORMATTED_EPOCHS}_lr${FORMATTED_LR}_r${LORA_RANK}_${TIMESTAMP}"

if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK_NAME"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  src/writer_sft.py \
  --task_name "$TASK_NAME" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK_NAME}/writer_sft/${RUN_IDENTIFIER}" \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --seed "$SEED" \
  --dataset_repo_id "${USER}/${TASK_NAME}_sft" \
  --model_repo_id "${MODEL_REPO_ID}" \
  --do_train True \
  --bf16 True \
  --do_eval True \
  --eval_on_start True \
  --eval_strategy "$EVAL_STRATEGY" \
  --save_strategy "$SAVE_STRATEGY" \
  --load_best_model_at_end True \
  --metric_for_best_model "loss" \
  --greater_is_better False \
  --save_total_limit 1 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_fewshot "$NUM_FEWSHOT" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --auto_find_batch_size "$AUTO_FIND_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --peft_type "LORA" \
  --task_type "CAUSAL_LM" \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT"
