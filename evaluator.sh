#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=12345
RUN_IDENTIFIER="$USER/npov_PERL_seed_130104_episodes_20000_lr_2e-5_klcoeff_1e-4_temp_1e-1"
BASE_MODEL="google/gemma-2-2b-it"
EVALUATOR_MODEL="gemini-2.0-flash"
USE_GEMINI="True"
GEMINI_API_KEY="AIzaSyCIPXhApp0pcu7TruZ8EyuW086VJ1wzrhk"
THRESHOLD=0.9998
EVALUATOR_NUM_FEWSHOT=2
MAX_TOKENS=768
TEMPERATURE=7e-1
TOP_P=1

# Dataset Parameters
DATASET_PROMPTS="${USER}/npov_perl"
DATASET_PROMPTS_SPLIT="train"
DATASET_LABELS="${USER}/npov_autorater"
DATASET_LABELS_SPLIT="test"

# Parameters derived from above
TASK="$1"

# Checks
if [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ] && [ "$TASK" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK"
    exit 1
fi

if [ "$2" == "generate" ]; then
    echo "Running in generation mode..."
    python3 evaluator.py \
        --task "$TASK" \
        --user "$USER" \
        --seed "$SEED" \
        --dataset_prompts "$DATASET_PROMPTS" \
        --dataset_prompts_split "$DATASET_PROMPTS_SPLIT" \
        --writer_model_base ${BASE_MODEL} \
        --writer_model_lora "${RUN_IDENTIFIER}" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P"
elif [ "$2" == "score" ]; then
    DATASET_WITH_COMPLETIONS="${USER}/eval_${RUN_IDENTIFIER#*/}_completions"
    echo "Running in scoring mode..."
    python3 evaluator.py \
        --task "$TASK" \
        --user "$USER" \
        --seed "$SEED" \
        --dataset_labels "$DATASET_LABELS" \
        --dataset_labels_split "$DATASET_LABELS_SPLIT" \
        --writer_model_lora "${RUN_IDENTIFIER}" \
        --evaluator_model ${EVALUATOR_MODEL} \
        --use_gemini $USE_GEMINI \
        --gemini_api_key ${GEMINI_API_KEY} \
        --evaluator_num_fewshot $EVALUATOR_NUM_FEWSHOT \
        --evaluate_evaluator False \
        --threshold $THRESHOLD \
        --dataset_with_completions "$DATASET_WITH_COMPLETIONS"
elif [ "$2" == "autoratereval" ]; then
    echo "Running in autoratereval mode..."
    python3 evaluator.py \
        --task "$TASK" \
        --user "$USER" \
        --seed "$SEED" \
        --dataset_labels "$DATASET_LABELS" \
        --dataset_labels_split "$DATASET_LABELS_SPLIT" \
        --writer_model_lora "${RUN_IDENTIFIER}" \
        --evaluator_model ${EVALUATOR_MODEL} \
        --use_gemini $USE_GEMINI \
        --gemini_api_key ${GEMINI_API_KEY} \
        --evaluator_num_fewshot $EVALUATOR_NUM_FEWSHOT \
        --evaluate_evaluator True \
        --threshold $THRESHOLD
else
    echo "Invalid mode. Use 'generate', 'score', or 'autoratereval'"
    exit 1
fi
