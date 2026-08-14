#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=12345
MODEL_REPO_ID="google/gemma-3-1b-it"

# Dataset Parameters
ORGANIC=true
SYN_HALL_LLM=false
SYN_HALL_STRUCT=false
NUM_ORGANIC_HALLUS_TO_KEEP=0
NUM_STRUCT_HALLUS_TO_KEEP=0

# Training Parameters
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
AUTO_FIND_BATCH_SIZE=True
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=1e-3
WEIGHT_DECAY=0.0
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.0
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1

# Infrastructure Parameters
PRECISION="BF16"
EVAL_STEPS=50
DEEPSPEED_CONFIG="scripts/deepspeed_config.yaml"

# Parameters derived from above
TASK_NAME="$1"
DATASET_REPO_ID="${USER}/${TASK_NAME}_rm"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')
FORMATTED_LR=$(python3 -c "import sys; lr=float('${LEARNING_RATE}'); print(f'{lr:.1e}')" 2>/dev/null || echo "$LEARNING_RATE")
FORMATTED_EPOCHS=$(python3 -c "import sys; e=float('${NUM_TRAIN_EPOCHS}'); print(f'{e:.2g}')" 2>/dev/null || echo "$NUM_TRAIN_EPOCHS")
RUN_IDENTIFIER="${USER}/${TASK_NAME}_RM_${MODEL_NAME}_S${SEED}_LLM_${SYN_HALL_LLM}_STRUCT_${SYN_HALL_STRUCT}_epo${FORMATTED_EPOCHS}_lr${FORMATTED_LR}_r${LORA_RANK}_${TIMESTAMP}"

# Checks
if [ "$PRECISION" = "FP16" ]; then
  FP16="True"
  BF16="False"
elif [ "$PRECISION" = "BF16" ]; then
  FP16="False"
  BF16="True"
else
  echo "Invalid PRECISION value. Must be 'FP16' or 'BF16'."
  exit 1
fi

if [ "$TASK_NAME" != "ragtruth" ] && [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ]; then
    echo "Invalid task name"
    exit 1
fi

if [ "$ORGANIC" = true ]; then
  DATASET_REPO_ID=$DATASET_REPO_ID"_organic"
  echo "Using dataset ${DATASET_REPO_ID}"
fi

if [ "$SYN_HALL_LLM" = true ]; then
  DATASET_REPO_ID=$DATASET_REPO_ID"_synthetic_llm"
  echo "Using dataset ${DATASET_REPO_ID}"
fi

if [ "$SYN_HALL_STRUCT" = true ]; then
  DATASET_REPO_ID=$DATASET_REPO_ID"_synthetic_struct"
  echo "Using dataset ${DATASET_REPO_ID}"
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  src/reward_model.py \
  --task_name "$TASK_NAME" \
  --seed "$SEED" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK_NAME}/reward_model/${RUN_IDENTIFIER}" \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "${DATASET_REPO_ID}" \
  --num_organic_hallus_to_keep $NUM_ORGANIC_HALLUS_TO_KEEP \
  --num_struct_hallus_to_keep $NUM_STRUCT_HALLUS_TO_KEEP \
  --model_repo_id "${MODEL_REPO_ID}" \
  --do_train True \
  --fp16 "$FP16" \
  --bf16 "$BF16" \
  --save_strategy "steps" \
  --save_steps "$EVAL_STEPS" \
  --load_best_model_at_end True \
  --metric_for_best_model "roc_auc" \
  --greater_is_better True \
  --save_total_limit 1 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --weight_decay "$WEIGHT_DECAY" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --auto_find_batch_size "$AUTO_FIND_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --do_eval True \
  --eval_on_start True \
  --eval_strategy "steps" \
  --eval_steps "$EVAL_STEPS" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --task_type "SEQ_CLS" \
  --peft_type "LORA" \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" 
