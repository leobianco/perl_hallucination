#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=12345
RUN_IDENTIFIER="leobianco/npov_SFT_google_S130104_epo25_lr1e-4_r8_2602261253"
BASE_MODEL="google/gemma-2-2b-it"
EVALUATOR_MODEL="gemini-2.0-flash"
USE_GEMINI="True"
THRESHOLD=0.99999
EVALUATOR_NUM_FEWSHOT=3
WRITER_NUM_FEWSHOT=0
MAX_TOKENS=300
TEMPERATURE=0.1
TOP_P=0.9
TOP_K=40

# Parameters derived from above
TASK_NAME="$1"

# Dataset Parameters
DATASET_PROMPTS="${USER}/${TASK_NAME}_final_test_set"
DATASET_PROMPTS_SPLIT="test"
DATASET_LABELS="${USER}/${TASK_NAME}_autorater"
DATASET_LABELS_SPLIT="test"

# Google Sheets integration
GSHEETS_NAME="The Great Final Push (Connected)"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK_NAME"
    exit 1
fi

if [ -z "${GEMINI_API_KEY}" ]; then
    echo "GEMINI_API_KEY environment variable is not set. Please export it before running this script."
    exit 1
fi

if [ "$2" == "generate" ]; then
    echo "Running in generation mode..."
    python3 -m src.evaluator \
        --task_name "$TASK_NAME" \
        --user "$USER" \
        --seed "$SEED" \
        --dataset_labels "$DATASET_LABELS" \
        --dataset_labels_split "$DATASET_LABELS_SPLIT" \
        --dataset_prompts "$DATASET_PROMPTS" \
        --dataset_prompts_split "$DATASET_PROMPTS_SPLIT" \
        --writer_model_base ${BASE_MODEL} \
        --writer_model_lora "${RUN_IDENTIFIER}" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --writer_num_fewshot $WRITER_NUM_FEWSHOT \
        --gsheets_name "$GSHEETS_NAME"
elif [ "$2" == "score" ]; then
    DATASET_WITH_COMPLETIONS="${USER}/eval_${RUN_IDENTIFIER#*/}_gens_T${TEMPERATURE}_wfs${WRITER_NUM_FEWSHOT}"
    echo "Running in scoring mode..."
    python3 -m src.evaluator \
        --task_name "$TASK_NAME" \
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
        --dataset_with_completions "$DATASET_WITH_COMPLETIONS" \
        --gsheets_name "$GSHEETS_NAME"
elif [ "$2" == "autoratereval" ]; then
    echo "Running in autoratereval mode..."
    python3 -m src.evaluator \
        --task_name "$TASK_NAME" \
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
