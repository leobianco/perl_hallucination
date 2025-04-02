#!/bin/bash

# If calling from hyperparameter search script,	variables are imported.
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here.
  TASK="$1"
  SEED=130104
  BASE_MODEL_PATH="google/gemma-2-2b-it"
  REWARD_MODEL_PATH="leobianco/bosch_RM_seed_130104_SYN_HALL_LLM_true_epochs_3_lr_1e-3_lora_8"
  SFT_MODEL_PATH="leobianco/bosch_SFT_seed_130104_epochs_0.01_lr_3e-3_lora_8_fewshot_0"
  DEEPSPEED_CONFIG="./deepspeed_config.yaml"
  TOTAL_EPISODES=20000
  RESPONSE_LENGTH=150
  NUM_SAMPLE_GENERATIONS=10
  LEARNING_RATE=2e-5
  KL_COEFF=1e-4
  RLOO_K=2
  NUM_PPO_EPOCHS=1
  NUM_MINIBATCHES=16
  PER_DEVICE_TRAIN_BATCH_SIZE=1
  LOCAL_ROLLOUT_FORWARD_BATCH_SIZE=8
  TEMPERATURE=7e-1
  SAVE_STEPS=1000
  RUN_IDENTIFIER="leobianco/${TASK}_PERL_seed_${SEED}_episodes_${TOTAL_EPISODES}_lr_${LEARNING_RATE}_klcoeff_${KL_COEFF}_temp_${TEMPERATURE}"
  SHUTDOWN=false
fi

if [ "$TASK" != "halomi" ] && [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ]; then
    echo "Invalid task name"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  perl.py \
  -- \
  --task "$TASK" \
  --seed "$SEED" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 10 \
  --output_dir "./checkpoints/${TASK}/perl/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "leobianco/${TASK}_perl_processed" \
  --model_repo_id "${BASE_MODEL_PATH}" \
  --stop_token "eos" \
  --do_train True \
  --save_strategy "steps" \
  --save_steps "$SAVE_STEPS" \
  --save_only_model True \
  --total_episodes "$TOTAL_EPISODES" \
  --learning_rate "$LEARNING_RATE" \
  --response_length "$RESPONSE_LENGTH" \
  --weight_decay 0.0 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 1 \
  --eval_accumulation_steps 1 \
  --reward_model_path "${REWARD_MODEL_PATH}" \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --kl_coef "$KL_COEFF" \
  --rloo_k "$RLOO_K" \
  --num_ppo_epochs "$NUM_PPO_EPOCHS" \
  --num_mini_batches "$NUM_MINIBATCHES" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --local_rollout_forward_batch_size "$LOCAL_ROLLOUT_FORWARD_BATCH_SIZE" \
  --missing_eos_penalty 1.0 \
  --temperature "$TEMPERATURE" \
  --num_sample_generations "$NUM_SAMPLE_GENERATIONS"

if [ "$SHUTDOWN" = true ]; then
  echo "Shutting down the VM..."
  sudo shutdown -h now
fi

