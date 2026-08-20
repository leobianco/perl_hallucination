#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E4B"
SFT_MODEL_PATH="${USER}/"

# Training Parameters
BATCH_SIZE=8
EVAL_BATCH_SIZE=16
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
EVAL_STRATEGY="epoch"
SAVE_STRATEGY="epoch"

# Infrastructure Parameters
DEEPSPEED_CONFIG="scripts/deepspeed_config.yaml"

# Parameters derived from above
TASK_NAME="$1"
DATASET_REPO_ID="${DATASET_REPO_ID:-${USER}/${TASK_NAME}_scope_preference}"
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $NF}')
TIMESTAMP=$(date '+%y%m%d%H%M')
FORMATTED_LR=$(python3 -c "import sys; lr=float('${LEARNING_RATE}'); print(f'{lr:.1e}')" 2>/dev/null || echo "$LEARNING_RATE")
FORMATTED_BETA=$(python3 -c "import sys; b=float('${BETA}'); print(f'{b:.2g}' if b>=0.001 else f'{b:.1e}')" 2>/dev/null || echo "$BETA")
FORMATTED_EPOCHS=$(python3 -c "import sys; e=float('${NUM_TRAIN_EPOCHS}'); print(f'{e:.2g}')" 2>/dev/null || echo "$NUM_TRAIN_EPOCHS")
RUN_IDENTIFIER="${USER}/${TASK_NAME}_SCOPE_DPO_${MODEL_NAME}_S${SEED}_epo${FORMATTED_EPOCHS}_lr${FORMATTED_LR}_beta${FORMATTED_BETA}_${TIMESTAMP}"

# Resumption configuration (can be passed via environment variable or second argument)
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
if [ -n "$2" ] && [[ "$2" != --* ]]; then
    RESUME_FROM_CHECKPOINT="$2"
fi

OUTPUT_DIR="./checkpoints/${TASK_NAME}/scope_dpo/${RUN_IDENTIFIER}"
if [ -n "$RESUME_FROM_CHECKPOINT" ] && [ "$RESUME_FROM_CHECKPOINT" != "auto" ] && [ "$RESUME_FROM_CHECKPOINT" != "True" ] && [ "$RESUME_FROM_CHECKPOINT" != "true" ]; then
    if [[ "$RESUME_FROM_CHECKPOINT" == *"checkpoint-"* ]] && [ -d "$RESUME_FROM_CHECKPOINT" ]; then
        OUTPUT_DIR=$(dirname "$RESUME_FROM_CHECKPOINT")
        RUN_IDENTIFIER=$(basename "$OUTPUT_DIR")
    elif [ -d "$RESUME_FROM_CHECKPOINT" ]; then
        OUTPUT_DIR="$RESUME_FROM_CHECKPOINT"
        RUN_IDENTIFIER=$(basename "$OUTPUT_DIR")
    elif [[ "$RESUME_FROM_CHECKPOINT" == *"/"* ]]; then
        CLEANED_HF_REPO=$(echo "$RESUME_FROM_CHECKPOINT" | sed 's|https://huggingface.co/||g' | sed 's|http://huggingface.co/||g' | sed 's|hf://||g' | sed 's|hf.co/||g' | awk -F'[@:]' '{print $1}')
        RUN_IDENTIFIER="$CLEANED_HF_REPO"
        OUTPUT_DIR="./checkpoints/${TASK_NAME}/scope_dpo/${RUN_IDENTIFIER}"
    fi
fi

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

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
  --output_dir "$OUTPUT_DIR" \
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
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --weight_decay "$WEIGHT_DECAY" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --auto_find_batch_size "$AUTO_FIND_BATCH_SIZE" \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --peft_type "LORA" \
  --task_type "CAUSAL_LM" \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  "${EXTRA_ARGS[@]}"
