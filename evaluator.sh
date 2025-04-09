#!/bin/bash

# If calling from hyperparameter search script, import hyperparameters
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run (copy and paste identifier)
  TASK="$1"
  SEED=12345
  USER="leobianco"
  DATASET_LABELS="leobianco/ragtruth_autorater_data"
  DATASET_LABELS_SPLIT="test"
  DATASET_PROMPTS="leobianco/bosch_perl_processed"
  DATASET_PROMPTS_SPLIT="test"
  RUN_IDENTIFIER="$USER/bosch_PERL_seed_130104_episodes_20000_lr_2e-5_klcoeff_1e-4_temp_7e-1"
fi

if [ "$TASK" != "ragtruth" ] && [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ]; then
    echo "Invalid task name" 
    echo "$TASK"
    exit 1
fi

BASE_MODEL="google/gemma-2-2b-it"
EVALUATOR_MODEL="gemini-2.0-flash"
USE_GEMINI="True"
GEMINI_API_KEY="AIzaSyCIPXhApp0pcu7TruZ8EyuW086VJ1wzrhk"
EVALUATOR_NUM_FEWSHOT=2
WRITER_NUM_FEWSHOT=0
EVAL_EVALUATOR="True"
THRESHOLD=0.9998
TEMPERATURE=7e-1

python3 evaluator.py \
  --task "$TASK" \
  --user "$USER" \
  --seed "$SEED" \
  --dataset_labels "$DATASET_LABELS" \
  --dataset_labels_split "$DATASET_LABELS_SPLIT" \
  --dataset_prompts "$DATASET_PROMPTS" \
  --dataset_prompts_split "$DATASET_PROMPTS_SPLIT" \
  --writer_model_base ${BASE_MODEL} \
  --writer_model_lora "${RUN_IDENTIFIER}" \
  --max_tokens 768 \
  --evaluator_model ${EVALUATOR_MODEL} \
  --use_gemini $USE_GEMINI \
  --gemini_api_key ${GEMINI_API_KEY} \
  --evaluator_num_fewshot $EVALUATOR_NUM_FEWSHOT \
  --writer_num_fewshot $WRITER_NUM_FEWSHOT \
  --evaluate_evaluator $EVAL_EVALUATOR \
  --threshold $THRESHOLD \
  --temperature $TEMPERATURE
