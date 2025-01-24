#!/bin/bash

# If calling from hyperparameter search script, import hyperparameters
if [ $SHLVL -gt 2 ]
then
  :
else
  # If single-run (copy and paste identifier)
  SEED=130104
  RUN_IDENTIFIER="leobianco/HALOMI_SFT_seed_130401_epochs_3_lr_5e-5_lora_8"
fi

BASE_MODEL="google/gemma-2-2b-it"
EVALUATOR_MODEL="google/gemma-2-27b-it"
NUM_FEWSHOT=4
EVAL_EVALUATOR="False"
THRESHOLD=0.15
TEMPERATURE=1

echo "Calling the evaluator script..."

python3 evaluator.py \
  --seed $SEED \
  --writer_model_base ${BASE_MODEL} \
  --writer_model_lora ${RUN_IDENTIFIER} \
  --max_tokens 256 \
  --evaluator_model ${EVALUATOR_MODEL} \
  --num_fewshot_examples $NUM_FEWSHOT \
  --evaluate_evaluator $EVAL_EVALUATOR \
  --threshold $THRESHOLD \
  --temperature $TEMPERATURE \
