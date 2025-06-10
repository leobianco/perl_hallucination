#!/bin/bash

# If calling from hyperparameter search script, import hyperparameters
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run (copy and paste identifier)
  TASK="$1"
  SEED=12345
  USER="leobianco"
  MAX_TOKENS=768
  DATASET_PROMPTS="${USER}/bosch_perl_processed"
  DATASET_PROMPTS_SPLIT="test"
  DATASET_LABELS="${USER}/ragtruth_autorater_data"
  DATASET_LABELS_SPLIT="test"
  RUN_IDENTIFIER="$USER/bosch_PERL_seed_130104_episodes_20000_lr_2e-5_klcoeff_1e-4_temp_7e-1"
fi

if [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ] && [ "$TASK" != "ragtruth" ]; then
    echo "Invalid task name" 
    echo "$TASK"
    exit 1
fi

BASE_MODEL="google/gemma-2-2b-it"
EVALUATOR_MODEL="gemini-2.0-flash"
USE_GEMINI="True"
GEMINI_API_KEY="AIzaSyCIPXhApp0pcu7TruZ8EyuW086VJ1wzrhk"
EVALUATOR_NUM_FEWSHOT=2
THRESHOLD=0.9998
TEMPERATURE=7e-1
TOP_P=1

# Check if we're in generation, scoring, or autoratereval mode
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
    echo "Running in scoring mode..."
    # Get the path to the dataset with completions
    DATASET_WITH_COMPLETIONS="${USER}/eval_${RUN_IDENTIFIER#*/}_completions"

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
