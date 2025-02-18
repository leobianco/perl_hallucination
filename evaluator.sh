#!/bin/bash

# If calling from hyperparameter search script, import hyperparameters
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run (copy and paste identifier)
  TASK="$1"
  USER="leobianco"
  SEED=130104
  DATASET_LABELS="leobianco/npov_rm_processed"
  DATASET_PROMPTS="leobianco/npov_rm_processed"
  RUN_IDENTIFIER="google/gemma-2-2b-it"
fi

if [ "$TASK" != "halomi" ] && [ "$TASK" != "npov" ]; then
    echo "Invalid dataset name"
    exit 1
fi

BASE_MODEL="google/gemma-2-2b-it"
EVALUATOR_MODEL="google/gemma-2-27b-it"
EVALUATOR_NUM_FEWSHOT=4
WRITER_NUM_FEWSHOT=1
EVAL_EVALUATOR="False"
THRESHOLD=0.1
TEMPERATURE=1e-1

python3 evaluator.py \
  --task "$TASK" \
  --user "$USER" \
  --seed "$SEED" \
  --dataset_labels "$DATASET_LABELS" \
  --dataset_prompts "$DATASET_PROMPTS" \
  --writer_model_base ${BASE_MODEL} \
  --writer_model_lora "${RUN_IDENTIFIER}" \
  --max_tokens 768 \
  --evaluator_model ${EVALUATOR_MODEL} \
  --evaluator_num_fewshot $EVALUATOR_NUM_FEWSHOT \
  --writer_num_fewshot $WRITER_NUM_FEWSHOT \
  --evaluate_evaluator $EVAL_EVALUATOR \
  --threshold $THRESHOLD \
  --temperature $TEMPERATURE
