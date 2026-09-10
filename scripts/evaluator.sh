#!/bin/bash

# Core Parameters
USER="leobianco"
SEED=12345
RUN_IDENTIFIER="google/gemma-4-E2B-it"
BASE_MODEL="google/gemma-4-E4B-it"
EVALUATOR_MODEL="gemini-2.5-flash"
USE_GEMINI="True"
RUN_AUTORATER="True"
THRESHOLD=0.1025
EVALUATOR_NUM_FEWSHOT=2
WRITER_NUM_FEWSHOT=0
MAX_TOKENS=250
MAX_EVAL_SAMPLES=1000
EVAL_BATCH_SIZE=32
MAX_WORKERS=32
TEMPERATURE=0.0
TOP_P=1.0
TOP_K=0
COMPUTE_BERTSCORE="True"
BERTSCORE_MODEL="sentence-transformers/all-MiniLM-L6-v2"
COMPUTE_PERPLEXITY="True"
FLUENCY_MODEL="${BASE_MODEL}"
LOG_TO_WANDB="True"
WANDB_PROJECT="new_perl_eval"
OVERWRITE_SCORES="${OVERWRITE_SCORES:-False}"
SCORES_CHECKPOINT_PATH="${SCORES_CHECKPOINT_PATH:-}"

# Parameters derived from above
TASK_NAME="$1"

# Dataset Parameters
DATASET_PROMPTS="${USER}/${TASK_NAME}_final_test_set"
DATASET_PROMPTS_SPLIT="test"
DATASET_LABELS="${USER}/${TASK_NAME}_autorater"
DATASET_LABELS_SPLIT="test"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK_NAME"
    exit 1
fi

export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
export GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI:-true}"

if [ "$RUN_AUTORATER" == "True" ] && [ -z "${GEMINI_API_KEY}" ] && [ -z "${GOOGLE_CLOUD_PROJECT}" ] && [ "${GOOGLE_GENAI_USE_VERTEXAI}" != "true" ]; then
    echo "Please set either GEMINI_API_KEY (for AI Studio) or GOOGLE_CLOUD_PROJECT / GOOGLE_GENAI_USE_VERTEXAI (for Vertex AI) before running this script."
    exit 1
fi

if [ "$2" == "generate" ]; then
    echo "Running in generation mode (max_eval_samples=$MAX_EVAL_SAMPLES)..."
    python3 -m src.evaluator \
        --mode "generate" \
        --task_name "$TASK_NAME" \
        --user "$USER" \
        --seed "$SEED" \
        --max_eval_samples "$MAX_EVAL_SAMPLES" \
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
        ${DATASET_WITH_COMPLETIONS:+--dataset_with_completions "$DATASET_WITH_COMPLETIONS"}
elif [ "$2" == "score" ]; then
    echo "Running in scoring mode (run_autorater=$RUN_AUTORATER, max_eval_samples=$MAX_EVAL_SAMPLES)..."
    python3 -m src.evaluator \
        --mode "score" \
        --task_name "$TASK_NAME" \
        --user "$USER" \
        --seed "$SEED" \
        --max_eval_samples "$MAX_EVAL_SAMPLES" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --max_workers "$MAX_WORKERS" \
        --dataset_labels "$DATASET_LABELS" \
        --dataset_labels_split "$DATASET_LABELS_SPLIT" \
        --writer_model_base ${BASE_MODEL} \
        --writer_model_lora "${RUN_IDENTIFIER}" \
        --run_autorater $RUN_AUTORATER \
        --evaluator_model ${EVALUATOR_MODEL} \
        --use_gemini $USE_GEMINI \
        --gemini_api_key ${GEMINI_API_KEY} \
        --evaluator_num_fewshot $EVALUATOR_NUM_FEWSHOT \
        --evaluate_evaluator False \
        --threshold $THRESHOLD \
        --temperature "$TEMPERATURE" \
        --writer_num_fewshot $WRITER_NUM_FEWSHOT \
        ${DATASET_WITH_COMPLETIONS:+--dataset_with_completions "$DATASET_WITH_COMPLETIONS"} \
        ${OVERWRITE_SCORES:+--overwrite_scores "$OVERWRITE_SCORES"} \
        ${SCORES_CHECKPOINT_PATH:+--scores_checkpoint_path "$SCORES_CHECKPOINT_PATH"} \
        --compute_bertscore "$COMPUTE_BERTSCORE" \
        --bertscore_model "$BERTSCORE_MODEL" \
        --compute_perplexity "$COMPUTE_PERPLEXITY" \
        --fluency_model "$FLUENCY_MODEL" \
        --log_to_wandb "${LOG_TO_WANDB:-False}" \
        --wandb_project "${WANDB_PROJECT:-new_perl_eval}"
elif [ "$2" == "autoratereval" ]; then
    echo "Running in autoratereval mode (max_eval_samples=$MAX_EVAL_SAMPLES)..."
    python3 -m src.evaluator \
        --mode "autoratereval" \
        --task_name "$TASK_NAME" \
        --user "$USER" \
        --seed "$SEED" \
        --max_eval_samples "$MAX_EVAL_SAMPLES" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --max_workers "$MAX_WORKERS" \
        --dataset_labels "$DATASET_LABELS" \
        --dataset_labels_split "$DATASET_LABELS_SPLIT" \
        --writer_model_lora "${RUN_IDENTIFIER}" \
        --evaluator_model ${EVALUATOR_MODEL} \
        --use_gemini $USE_GEMINI \
        --gemini_api_key ${GEMINI_API_KEY} \
        --evaluator_num_fewshot $EVALUATOR_NUM_FEWSHOT \
        --evaluate_evaluator True \
        --threshold $THRESHOLD \
        ${OVERWRITE_SCORES:+--overwrite_scores "$OVERWRITE_SCORES"} \
        ${SCORES_CHECKPOINT_PATH:+--scores_checkpoint_path "$SCORES_CHECKPOINT_PATH"}
else
    echo "Invalid mode. Use 'generate', 'score', or 'autoratereval'"
    exit 1
fi
