#!/bin/bash

# ==============================================================================
# SCOPE Baseline Pipeline Script
# Paper: "SCOPE: A Self-supervised Framework for Improving Faithfulness in
#        Conditional Text Generation" (https://arxiv.org/abs/2502.13674)
#
# Stages:
#   1. SFT Training (Half Data, D1): Train initial SFT checkpoint on 50% of data.
#   2. Synthetic Data Generation (D2): Generate dispreferred completions via
#      noisy decoding by mixing logits between SFT and base pre-trained models (Algorithm 1).
#   3. DPO Preference Tuning: Optimize policy to prefer ground truth completions
#      over unfaithful synthetic completions using Direct Preference Optimization.
# ==============================================================================

# Core Parameters
USER="leobianco"
SEED=130104
MODEL_REPO_ID="google/gemma-4-E4B"
TASK_NAME="$1"
STAGE="${2:-all}"  # "sft", "generate_data", "dpo", or "all"

# SFT Training Parameters (Stage 1)
SFT_BATCH_SIZE=4
SFT_NUM_EPOCHS=1
SFT_LR=3e-3
SFT_LORA_R=8
SFT_LORA_ALPHA=16
SFT_DATA_FRACTION=0.5

# SCOPE Synthetic Data Generation Parameters (Stage 2 - Algorithm 1)
ALPHA=0.5
SPLIT_RATIO=0.5
SAMPLING_MODE="bernoulli"
GEN_TEMP=0.7
GEN_TOP_P=0.9
GEN_TOP_K=50
MAX_NEW_TOKENS=150

# DPO Training Parameters (Stage 3)
DPO_BETA=0.1
DPO_BATCH_SIZE=2
DPO_NUM_EPOCHS=1
DPO_LR=5e-6
DPO_LORA_R=8
DPO_LORA_ALPHA=16
MAX_LENGTH=512
MAX_PROMPT_LENGTH=384

# Infrastructure Parameters
DEEPSPEED_CONFIG="scripts/deepspeed_config.yaml"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name: $TASK_NAME"
    echo "Usage: ./scripts/scope_baseline.sh <task_name> [stage: all|sft|generate_data|dpo]"
    echo "Valid task choices: npov, bosch, ragtruth"
    exit 1
fi

MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')

# Identifiers
SFT_RUN_IDENTIFIER="${USER}/${TASK_NAME}_SFT_half_${MODEL_NAME}_S${SEED}_${TIMESTAMP}"
SFT_MODEL_PATH="${USER}/${TASK_NAME}_SFT_half_${MODEL_NAME}_S${SEED}_${TIMESTAMP}"
PREF_DATASET_REPO="${USER}/${TASK_NAME}_scope_preference"
DPO_RUN_IDENTIFIER="${USER}/${TASK_NAME}_SCOPE_DPO_${MODEL_NAME}_S${SEED}_${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Stage 1: SFT on Half Data (D1)
# ------------------------------------------------------------------------------
if [ "$STAGE" == "all" ] || [ "$STAGE" == "sft" ]; then
  echo "======================================================================"
  echo "Stage 1: Training initial SFT model on half data (fraction: $SFT_DATA_FRACTION)..."
  echo "======================================================================"

  accelerate launch \
    --config_file="${DEEPSPEED_CONFIG}" \
    src/writer_sft.py \
    -- \
    --task_name "$TASK_NAME" \
    --report_to "wandb" \
    --run_name "$SFT_RUN_IDENTIFIER" \
    --logging_steps 1 \
    --output_dir "./checkpoints/${TASK_NAME}/writer_sft_half/${SFT_RUN_IDENTIFIER}" \
    --push_to_hub True \
    --hub_model_id "$SFT_RUN_IDENTIFIER" \
    --seed "$SEED" \
    --dataset_repo_id "${USER}/${TASK_NAME}_sft" \
    --model_repo_id "${MODEL_REPO_ID}" \
    --sft_data_fraction "$SFT_DATA_FRACTION" \
    --do_train True \
    --bf16 True \
    --num_train_epochs "$SFT_NUM_EPOCHS" \
    --learning_rate "$SFT_LR" \
    --per_device_train_batch_size "$SFT_BATCH_SIZE" \
    --peft_type "LORA" \
    --task_type "CAUSAL_LM" \
    --lora_r "$SFT_LORA_R" \
    --lora_alpha "$SFT_LORA_ALPHA"
fi

# ------------------------------------------------------------------------------
# Stage 2: Synthetic Data Generation on D2 (Algorithm 1)
# ------------------------------------------------------------------------------
if [ "$STAGE" == "all" ] || [ "$STAGE" == "generate_data" ]; then
  echo "======================================================================"
  echo "Stage 2: Generating synthetic dispreferred data via noisy decoding (alpha=$ALPHA)..."
  echo "======================================================================"

  python3 -m src.scope_data_generation \
    --task_name "$TASK_NAME" \
    --seed "$SEED" \
    --dataset_repo_id "${USER}/${TASK_NAME}_sft" \
    --model_repo_id "${MODEL_REPO_ID}" \
    --sft_model_path "${SFT_MODEL_PATH}" \
    --output_dataset_repo_id "$PREF_DATASET_REPO" \
    --alpha "$ALPHA" \
    --split_ratio "$SPLIT_RATIO" \
    --sampling_mode "$SAMPLING_MODE" \
    --temperature "$GEN_TEMP" \
    --top_p "$GEN_TOP_P" \
    --top_k "$GEN_TOP_K" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --push_to_hub True
fi

# ------------------------------------------------------------------------------
# Stage 3: DPO Preference Tuning
# ------------------------------------------------------------------------------
if [ "$STAGE" == "all" ] || [ "$STAGE" == "dpo" ]; then
  echo "======================================================================"
  echo "Stage 3: Preference tuning with DPO (beta=$DPO_BETA)..."
  echo "======================================================================"

  accelerate launch \
    --config_file="${DEEPSPEED_CONFIG}" \
    src/dpo.py \
    -- \
    --task_name "$TASK_NAME" \
    --report_to "wandb" \
    --run_name "$DPO_RUN_IDENTIFIER" \
    --logging_steps 1 \
    --output_dir "./checkpoints/${TASK_NAME}/scope_dpo/${DPO_RUN_IDENTIFIER}" \
    --push_to_hub True \
    --hub_model_id "$DPO_RUN_IDENTIFIER" \
    --seed "$SEED" \
    --dataset_repo_id "$PREF_DATASET_REPO" \
    --model_repo_id "${MODEL_REPO_ID}" \
    --sft_model_path "${SFT_MODEL_PATH}" \
    --do_train True \
    --bf16 True \
    --beta "$DPO_BETA" \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --num_train_epochs "$DPO_NUM_EPOCHS" \
    --learning_rate "$DPO_LR" \
    --per_device_train_batch_size "$DPO_BATCH_SIZE" \
    --peft_type "LORA" \
    --task_type "CAUSAL_LM" \
    --lora_r "$DPO_LORA_R" \
    --lora_alpha "$DPO_LORA_ALPHA"
fi

echo "SCOPE baseline completed successfully!"
