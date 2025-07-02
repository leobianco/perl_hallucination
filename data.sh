#!/usr/bin/env bash

# Parameters
USER="leobianco"
SEED=12345
SYNTH_LLM="False"
SYNTH_STRUCT="False"
NUM_SYNTH_HALLUS=50
GEMINI_API_KEY="AIzaSyCIPXhApp0pcu7TruZ8EyuW086VJ1wzrhk"
SYNTH_LLM_TEMPERATURE=0.7
SYNTH_LLM_NUM_FEWSHOT=5

# Parameters derived from above
TASK_NAME="$1"

# Checks
if [ "$TASK_NAME" != "ragtruth" ] && [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ]; then
    echo "Invalid dataset name"
    exit 1
fi

# Run script
python data.py \
    --task_name "$TASK_NAME" \
    --seed "$SEED" \
    --synthetic_hallus_llm $SYNTH_LLM \
    --synthetic_hallus_struct $SYNTH_STRUCT \
    --num_synth_hallus $NUM_SYNTH_HALLUS \
    --gemini_api_key "${GEMINI_API_KEY}" \
    --synth_llm_temperature $SYNTH_LLM_TEMPERATURE \
    --synth_llm_num_fewshot $SYNTH_LLM_NUM_FEWSHOT
