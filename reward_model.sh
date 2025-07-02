#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=12345
MODEL_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# Dataset Parameters
ORGANIC=true
SYN_HALL_LLM=false
SYN_HALL_STRUCT=false
NUM_ORGANIC_HALLUS_TO_KEEP=0
NUM_STRUCT_HALLUS_TO_KEEP=0

# Training Parameters
BATCH_SIZE=1
NUM_TRAIN_EPOCHS=2
LEARNING_RATE=5e-5
WEIGHT_DECAY=5e-4
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1

# Infrastructure Parameters
PRECISION="BF16"
EVAL_STEPS=5
DEEPSPEED_CONFIG="./deepspeed_config.yaml"

# Parameters derived from above
TASK="$1"
DATASET_REPO_ID="${USER}/${TASK}_rm"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')
RUN_IDENTIFIER="leobianco/${TASK}_RM_${MODEL_NAME}_S${SEED}_LLM_${SYN_HALL_LLM}_STRUCT_${SYN_HALL_STRUCT}_epo${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_r${LORA_RANK}_${TIMESTAMP}"

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

if [ "$TASK" != "ragtruth" ] && [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ]; then
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
  reward_model.py \
  -- \
  --task "$TASK" \
  --seed "$SEED" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK}/reward_model/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "${DATASET_REPO_ID}" \
  --num_organic_hallus_to_keep $NUM_ORGANIC_HALLUS_TO_KEEP \
  --num_struct_hallus_to_keep $NUM_STRUCT_HALLUS_TO_KEEP \
  --model_repo_id "${MODEL_REPO_ID}" \
  --do_train True \
  --fp16 "$FP16" \
  --bf16 "$BF16" \
  --save_strategy "epoch" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --weight_decay "$WEIGHT_DECAY" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --do_eval True \
  --eval_on_start True \
  --eval_strategy "steps" \
  --eval_steps "$EVAL_STEPS" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --eval_accumulation_steps 1 \
  --task_type "SEQ_CLS" \
  --peft_type "LORA" \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" 