#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=130104
# Overridable from the environment so that swapping the base model does
# not require editing this script:  MODEL_REPO_ID=Qwen/Qwen3-4B-Instruct-2507 ./perl.sh
MODEL_REPO_ID="${MODEL_REPO_ID:-google/gemma-4-E4B-it}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${USER}/}"

# LoRA Parameters
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.0

# Training Parameters
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-5
LR_SCHEDULER_TYPE="cosine"
WARMUP_RATIO=0.1
BETA=1e-4
MAX_COMPLETION_LENGTH=512

# Token budget the reward model uses to score (prompt + completion).
#
# MUST be identical to REWARD_MAX_LENGTH in scripts/reward_model.sh, which is
# where the reward model was trained. If they disagree, the model is trained
# on one view of the text and queried on another.
#
# MUST also leave room for MAX_COMPLETION_LENGTH on top of the prompt.
# Truncation is applied from the LEFT so the completion always survives, but
# if the budget is too small the reward model stops seeing enough context to
# judge groundedness.
REWARD_MAX_LENGTH="${REWARD_MAX_LENGTH:-2048}"

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
NUM_ITERATIONS=1
STEPS_PER_GENERATION=16
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE=16
AUTO_FIND_BATCH_SIZE=False
TEMPERATURE="${TEMPERATURE:-0.7}"
NUM_FEWSHOT=0
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
REWARD_PENALTY_ALPHA="${REWARD_PENALTY_ALPHA:-1.0}"
SAVE_STRATEGY="steps"
SAVE_STEPS=25
DO_EVAL=True
EVAL_STRATEGY="steps"
EVAL_STEPS=25
EVAL_ON_START=True

# Infrastructure Parameters
# Override for a large policy, e.g.
#   DEEPSPEED_CONFIG=scripts/deepspeed_config_zero3.yaml ./perl.sh
# The orchestrator picks this automatically from the model size (see
# src/orchestrator/accel.py); a hand-run script has to be told.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/deepspeed_config.yaml}"
SHUTDOWN=false

# Parameters derived from above
TASK_NAME="$1"
TIMESTAMP=$(date '+%y%m%d%H%M')
# $NF, not $1: awk -F'/' '{print $1}' returns the *vendor* namespace, so every
# run identifier said "google" instead of naming the model. Harmless while one
# model was ever used; with two it makes runs of different models collide.
MODEL_NAME=$(echo "$MODEL_REPO_ID" | awk -F'/' '{print $NF}')
FORMATTED_LR=$(python3 -c "import sys; lr=float('${LEARNING_RATE}'); print(f'{lr:.1e}')" 2>/dev/null || echo "$LEARNING_RATE")
FORMATTED_BETA=$(python3 -c "import sys; b=float('${BETA}'); print(f'{b:.2g}' if b>=0.001 else f'{b:.1e}')" 2>/dev/null || echo "$BETA")
FORMATTED_EPOCHS=$(python3 -c "import sys; e=float('${NUM_TRAIN_EPOCHS}'); print(f'{e:.2g}')" 2>/dev/null || echo "$NUM_TRAIN_EPOCHS")
ALPHA_SUFFIX=""
if [ "$REWARD_PENALTY_ALPHA" != "1.0" ] && [ "$REWARD_PENALTY_ALPHA" != "1" ]; then
    ALPHA_SUFFIX="_a${REWARD_PENALTY_ALPHA}"
fi
RUN_IDENTIFIER="${USER}/${TASK_NAME}_PERL_${MODEL_NAME}_S${SEED}_epo${FORMATTED_EPOCHS}_lr${FORMATTED_LR}_beta${FORMATTED_BETA}_r${LORA_RANK}${ALPHA_SUFFIX}_${TIMESTAMP}"

# Resumption configuration (can be passed via environment variable or second argument)
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
if [ -n "$2" ] && [[ "$2" != --* ]]; then
    RESUME_FROM_CHECKPOINT="$2"
fi

OUTPUT_DIR="./checkpoints/${TASK_NAME}/perl/${RUN_IDENTIFIER}"
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
        OUTPUT_DIR="./checkpoints/${TASK_NAME}/perl/${RUN_IDENTIFIER}"
    fi
fi

EXTRA_ARGS=()
if [ -n "$RESUME_FROM_CHECKPOINT" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [ -n "$SFT_MODEL_PATH" ] && [ "$SFT_MODEL_PATH" != "none" ] && [ "$SFT_MODEL_PATH" != "None" ] && [ "$SFT_MODEL_PATH" != "${USER}/" ]; then
    EXTRA_ARGS+=(--sft_model_path "$SFT_MODEL_PATH")
fi

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ] && [ "$TASK_NAME" != "ragtruth-qa" ] && [ "$TASK_NAME" != "ragtruth-summarization" ]; then
    echo "Invalid task name" 
    echo "$TASK_NAME"
    exit 1
fi

accelerate launch \
  --config_file="${DEEPSPEED_CONFIG}" \
  src/perl.py \
  --task_name "$TASK_NAME" \
  --seed "$SEED" \
  --report_to "wandb" \
  --run_name "$RUN_IDENTIFIER" \
  --logging_steps 1 \
  --log_completions True \
  --output_dir "$OUTPUT_DIR" \
  --push_to_hub True \
  --hub_model_id "$RUN_IDENTIFIER" \
  --dataset_repo_id "${USER}/${TASK_NAME}_perl" \
  --model_repo_id "${MODEL_REPO_ID}" \
  --do_train True \
  --do_eval "$DO_EVAL" \
  --eval_strategy "$EVAL_STRATEGY" \
  --eval_steps "$EVAL_STEPS" \
  --eval_on_start "$EVAL_ON_START" \
  --save_strategy "$SAVE_STRATEGY" \
  --save_steps "$SAVE_STEPS" \
  --load_best_model_at_end True \
  --metric_for_best_model "rewards/reward_fn/mean" \
  --greater_is_better True \
  --save_total_limit 2 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
  --warmup_ratio "$WARMUP_RATIO" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --reward_max_length "$REWARD_MAX_LENGTH" \
  --weight_decay 0.0 \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --reward_penalty_alpha "$REWARD_PENALTY_ALPHA" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --reward_model_path "${REWARD_MODEL_PATH}" \
  --beta "$BETA" \
  --num_generations "$NUM_GENERATIONS" \
  --num_iterations "$NUM_ITERATIONS" \
  --steps_per_generation "$STEPS_PER_GENERATION" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --auto_find_batch_size "$AUTO_FIND_BATCH_SIZE" \
  --temperature "$TEMPERATURE" \
  --peft_type "LORA" \
  --task_type "CAUSAL_LM" \
  --lora_r "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --num_fewshot "$NUM_FEWSHOT" \
  "${EXTRA_ARGS[@]}"

if [ "$SHUTDOWN" = true ]; then
  echo "Shutting down the VM..."
  sudo shutdown -h now
fi
