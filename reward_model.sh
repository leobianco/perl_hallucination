#!/bin/bash

# If calling from hyperparameter search script, variables are imported.
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here.
  TASK="$1"
  SYN_HALL_LLM=false
  SYN_HALL_STRUCT=true
  DATASET_REPO_ID="leobianco/${TASK}_rm_processed"
  DEEPSPEED_CONFIG="./deepspeed_config.yaml"
  MODEL_REPO_ID="google/gemma-2-2b-it"
  SEED=12345
  PRECISION="BF16"  # BF16 for Gemma, Mistral, Qwen
  NUM_TRAIN_EPOCHS=2
  BATCH_SIZE=2
  LEARNING_RATE=5e-5
  LR_SCHEDULER_TYPE="cosine"
  WARMUP_RATIO=0.15
  LORA_RANK=8
  LORA_ALPHA=8
  LORA_DROPOUT=0.1
  WEIGHT_DECAY=5e-4
  MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
  RUN_IDENTIFIER="leobianco/${TASK}_RM_model_${MODEL_NAME}_seed_${SEED}_SYN_LLM_${SYN_HALL_LLM}_SYN_STRUCT_${SYN_HALL_STRUCT}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
fi

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
  --logging_steps 5 \
  --output_dir "./checkpoints/${TASK}/reward_model/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "${DATASET_REPO_ID}" \
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
  --eval_steps 20 \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --eval_accumulation_steps 1 \
  --task_type "SEQ_CLS" \
  --peft_type "LORA" \
  --r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" 