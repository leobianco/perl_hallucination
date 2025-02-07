#!/bin/bash

# If calling from hyperparameter search script,	variables are imported.
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here.
  SEED=130104
  DEEPSPEED_CONFIG="./deepspeed_config.yaml"
  TOTAL_EPISODES=25000
  RESPONSE_LENGTH=128
  NUM_SAMPLE_GENERATIONS=20
  LEARNING_RATE=5e-6
  LORA_RANK=16
  KL_COEFF=3e-2
  RLOO_K=2
  NUM_PPO_EPOCHS=4
  NUM_MINIBATCHES=16
  PER_DEVICE_TRAIN_BATCH_SIZE=2
  LOCAL_ROLLOUT_FORWARD_BATCH_SIZE=8
  TEMPERATURE=7e-1
  RUN_IDENTIFIER="leobianco/HALOMI_PERL_seed_${SEED}_episodes_${TOTAL_EPISODES}_lora_${LORA_RANK}_lr_${LEARNING_RATE}_klcoeff_${KL_COEFF}_rlook_${RLOO_K}_ppoepochs_${NUM_PPO_EPOCHS}"
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  perl.py \
  -- \
  --seed "$SEED" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 10 \
  --output_dir "./checkpoints/halomi/perl/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "leobianco/perl_halomi_processed" \
  --model_repo_id "google/gemma-2-2b-it" \
  --stop_token "eos" \
  --do_train True \
  --save_strategy "steps" \
  --save_steps 150 \
  --total_episodes "$TOTAL_EPISODES" \
  --learning_rate "$LEARNING_RATE" \
  --response_length "$RESPONSE_LENGTH" \
  --weight_decay 0.0 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --reward_model_path "leobianco/HALOMI_RM_seed_130104_epochs_3_lr_1e-3_lora_8" \
  --sft_model_path "google/gemma-2-2b-it" \
  --r "$LORA_RANK" \
  --kl_coef "$KL_COEFF" \
  --rloo_k "$RLOO_K" \
  --num_ppo_epochs "$NUM_PPO_EPOCHS" \
  --num_mini_batches "$NUM_MINIBATCHES" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --local_rollout_forward_batch_size "$LOCAL_ROLLOUT_FORWARD_BATCH_SIZE" \
  --missing_eos_penalty 1.0 \
  --temperature "$TEMPERATURE" \
  --num_sample_generations "$NUM_SAMPLE_GENERATIONS"
