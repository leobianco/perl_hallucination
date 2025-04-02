#!/bin/bash

# If calling from hyperparameter search script, variables are imported.
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here.
  TASK="$1"
  SYN_HALL_LLM=true
  DATASET_REPO_ID="leobianco/${TASK}_rm_processed"
  DEEPSPEED_CONFIG="./deepspeed_config.yaml"
  SEED=130104
  NUM_TRAIN_EPOCHS=3
  LEARNING_RATE=1e-3
  LORA_RANK=8
  RUN_IDENTIFIER="leobianco/${TASK}_RM_seed_${SEED}_SYN_HALL_LLM_${SYN_HALL_LLM}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
fi

if [ "$TASK" != "halomi" ] && [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ]; then
    echo "Invalid task name"
    exit 1
fi

if [ "$SYN_HALL_LLM" = true ]; then
  DATASET_REPO_ID=$DATASET_REPO_ID"_synthetic_llm"
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
  --dataset_repo_id "leobianco/${TASK}_rm_processed" \
  --model_repo_id "google/gemma-2-2b-it" \
  --do_train True \
  --bf16 True \
  --save_strategy "epoch" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --do_eval True \
  --eval_on_start True \
  --eval_strategy "steps" \
  --eval_steps 20 \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --task_type "SEQ_CLS" \
  --r "$LORA_RANK"
