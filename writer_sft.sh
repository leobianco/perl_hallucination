#!/bin/bash

# If calling from hyperparameter search script, import variables from there
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here
  DATASET="$1"
  USER="leobianco"
  DEEPSPEED_CONFIG="./deepspeed_config.yaml"
  EXPERIMENT_TYPE="${DATASET}_SFT"
  SEED=130401
  NUM_TRAIN_EPOCHS=3
  LEARNING_RATE=5e-5
  LORA_RANK=8
  RUN_IDENTIFIER="${USER}/${EXPERIMENT_TYPE}_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
fi

if [ "$DATASET" != "halomi" ] && [ "$DATASET" != "npov" ]; then
    echo "Invalid dataset name"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  writer_sft.py \
  -- \
  --dataset "$DATASET" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 10 \
  --output_dir "./checkpoints/${DATASET}/writer_sft/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --seed "$SEED" \
  --dataset_repo_id "leobianco/writer_sft_${DATASET}_processed" \
  --model_repo_id "google/gemma-2-2b-it" \
  --do_train True \
  --bf16 True \
  --save_strategy "epoch" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay 0.0 \
  --max_seq_length 512 \
  --dataset_text_field "prompt" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --do_eval False \
  --eval_strategy "no" \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --task_type "CAUSAL_LM" \
  --r "$LORA_RANK"
