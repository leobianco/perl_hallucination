#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-2-2b-it"
REWARD_MODEL_PATH="${USER}/bosch_RM_seed_130104_SYN_HALL_LLM_true_epochs_3_lr_1e-3_lora_8"
SFT_MODEL_PATH="${USER}/bosch_SFT_seed_130104_epochs_0.01_lr_3e-3_lora_8_fewshot_0"

# Training Parameters
TOTAL_EPISODES=20000
LEARNING_RATE=2e-5
KL_COEFF=1e-4
RESPONSE_LENGTH=150
RLOO_K=2
NUM_PPO_EPOCHS=1
NUM_MINIBATCHES=16
PER_DEVICE_BATCH_SIZE=4
LOCAL_ROLLOUT_FORWARD_BATCH_SIZE=8
TEMPERATURE=7e-1
SAVE_STEPS=1000
NUM_SAMPLE_GENERATIONS=10

# Infrastructure Parameters
DEEPSPEED_CONFIG="./deepspeed_config.yaml"

# Parameters derived from above
TASK="$1"
TIMESTAMP=$(date '+%y%m%d%H%M')
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
RUN_IDENTIFIER="${USER}/${TASK}_PERL_${MODEL_NAME}_S_${SEED}_episodes_${TOTAL_EPISODES}_lr_${LEARNING_RATE}_kl_${KL_COEFF}_${TIMESTAMP}"
SHUTDOWN=false

# Checks
if [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ] && [ "$TASK" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK"
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
  --logging_steps 1 \
  --output_dir "./checkpoints/${TASK}/perl/${RUN_IDENTIFIER}" \
  --overwrite_output_dir True \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "${USER}/${TASK}_perl" \
  --model_repo_id "${MODEL_REPO_ID}" \
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
  --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --eval_accumulation_steps 1 \
  --reward_model_path "${REWARD_MODEL_PATH}" \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --kl_coef "$KL_COEFF" \
  --rloo_k "$RLOO_K" \
  --num_ppo_epochs "$NUM_PPO_EPOCHS" \
  --num_mini_batches "$NUM_MINIBATCHES" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --local_rollout_forward_batch_size "$LOCAL_ROLLOUT_FORWARD_BATCH_SIZE" \
  --missing_eos_penalty 1.0 \
  --temperature "$TEMPERATURE" \
  --num_sample_generations "$NUM_SAMPLE_GENERATIONS"

if [ "$SHUTDOWN" = true ]; then
  echo "Shutting down the VM..."
  sudo shutdown -h now
fi

