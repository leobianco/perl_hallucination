#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-2-2b-it"

# Training Parameters
BATCH_SIZE=4
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=3e-3
WEIGHT_DECAY=0.0
LORA_RANK=8
NUM_FEWSHOT=0
SAVE_STEPS=200

# Infrastructure Parameters
DEEPSPEED_CONFIG="./deepspeed_config.yaml"

# Parameters derived from above
TASK="$1"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')
RUN_IDENTIFIER="${USER}/${TASK}_SFT_${MODEL_NAME}_S${SEED}_epo${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_r${LORA_RANK}_${TIMESTAMP}"

if [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ] && [ "$TASK" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  writer_sft.py \
  -- \
  --task "$TASK" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK}/writer_sft/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --seed "$SEED" \
  --dataset_repo_id "${USER}/${TASK}_sft" \
  --model_repo_id "${MODEL_REPO_ID}" \
  --do_train True \
  --bf16 True \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --num_fewshot "$NUM_FEWSHOT" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --do_eval True \
  --eval_on_start True \
  --eval_strategy "epoch" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --eval_accumulation_steps 1 \
  --task_type "CAUSAL_LM" \
  --r "$LORA_RANK"
