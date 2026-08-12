#!/bin/bash

# ==============================================================================
# SSFO Baseline Pipeline Script
# Paper: "SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented
#        Generation" (https://arxiv.org/abs/2508.17225)
#
# Stages:
#   1. SFT Training: Train supervised fine-tuned writer model on SFT dataset.
#   2. Synthetic Preference Data Generation: Contrast SFT model generations
#      conditioned on (query + context) as preferred, and conditioned only on
#      (query) as dispreferred (contextual hallucination).
#   3. DPO Preference Tuning: Optimize policy using Direct Preference Optimization.
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

# SSFO Generation Parameters (Stage 2)
GEN_TEMP=0.7
GEN_TOP_P=0.9
GEN_TOP_K=50
MAX_NEW_TOKENS=150
USE_GROUND_TRUTH_CHOSEN=False

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
    echo "Usage: ./scripts/ssfo_baseline.sh <task_name> [stage: all|sft|generate_data|dpo]"
    echo "Valid task choices: npov, bosch, ragtruth"
    exit 1
fi

MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $1}')
TIMESTAMP=$(date '+%y%m%d%H%M')

# Identifiers
SFT_RUN_IDENTIFIER="${USER}/${TASK_NAME}_SFT_${MODEL_NAME}_S${SEED}_${TIMESTAMP}"
SFT_MODEL_PATH="${USER}/${TASK_NAME}_SFT_${MODEL_NAME}_S${SEED}_${TIMESTAMP}"
PREF_DATASET_REPO="${USER}/${TASK_NAME}_ssfo_preference"
DPO_RUN_IDENTIFIER="${USER}/${TASK_NAME}_SSFO_DPO_${MODEL_NAME}_S${SEED}_${TIMESTAMP}"

# ------------------------------------------------------------------------------
# Stage 1: SFT Training
# ------------------------------------------------------------------------------
if [ "$STAGE" == "all" ] || [ "$STAGE" == "sft" ]; then
  echo "======================================================================"
  echo "Stage 1: Training initial SFT model..."
  echo "======================================================================"

  accelerate launch \
    --config_file="${DEEPSPEED_CONFIG}" \
    src/writer_sft.py \
    -- \
    --task_name "$TASK_NAME" \
    --report_to "wandb" \
    --run_name "$SFT_RUN_IDENTIFIER" \
    --logging_steps 1 \
    --output_dir "./checkpoints/${TASK_NAME}/writer_sft/${SFT_RUN_IDENTIFIER}" \
    --push_to_hub True \
    --hub_model_id "$SFT_RUN_IDENTIFIER" \
    --seed "$SEED" \
    --dataset_repo_id "${USER}/${TASK_NAME}_sft" \
    --model_repo_id "${MODEL_REPO_ID}" \
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
# Stage 2: SSFO Preference Data Generation
# ------------------------------------------------------------------------------
if [ "$STAGE" == "all" ] || [ "$STAGE" == "generate_data" ]; then
  echo "======================================================================"
  echo "Stage 2: Generating synthetic preference data via context contrasting..."
  echo "======================================================================"

  python3 -m src.ssfo_data_generation \
    --task_name "$TASK_NAME" \
    --seed "$SEED" \
    --dataset_repo_id "${USER}/${TASK_NAME}_sft" \
    --model_repo_id "${MODEL_REPO_ID}" \
    --sft_model_path "${SFT_MODEL_PATH}" \
    --output_dataset_repo_id "$PREF_DATASET_REPO" \
    --temperature "$GEN_TEMP" \
    --top_p "$GEN_TOP_P" \
    --top_k "$GEN_TOP_K" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --use_ground_truth_chosen "$USE_GROUND_TRUTH_CHOSEN" \
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
    --output_dir "./checkpoints/${TASK_NAME}/ssfo_dpo/${DPO_RUN_IDENTIFIER}" \
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

echo "SSFO baseline completed successfully!"
