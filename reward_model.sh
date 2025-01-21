#!/bin/bash

# If calling from hyperparameter search script, variables are imported.
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here.
  DEEPSPEED_CONFIG="./deepspeed_config.yaml"
  SEED=130104
  NUM_TRAIN_EPOCHS=3
  LEARNING_RATE=1e-3
  LORA_RANK=8
  RUN_IDENTIFIER="leobianco/HALOMI_RM_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
fi

accelerate launch \
--config_file=${DEEPSPEED_CONFIG} \
reward_model.py \
-- \
--seed $SEED \
--report_to "wandb" \
--run_name $RUN_IDENTIFIER \
--logging_steps 5 \
--output_dir "./checkpoints/halomi/reward_model/${RUN_IDENTIFIER}" \
--overwrite_output_dir True \
--push_to_hub True \
--hub_model_id $RUN_IDENTIFIER \
--dataset_name "leobianco/rm_halomi_processed" \
--model_identifier "google/gemma-2-2b-it" \
--do_train True \
--save_strategy "no" \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--learning_rate $LEARNING_RATE \
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
--r $LORA_RANK \
