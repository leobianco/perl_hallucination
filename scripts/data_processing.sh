#!/usr/bin/env bash

# Parameters
TASK_NAME="$1"
USER="leobianco"
SEED=12345
HF_REPO="${USER}/${TASK_NAME}"
SYNTH_STRUCT="True"
SYNTH_LLM="True"
NUM_SYNTH_HALLUS=100
GEMINI_API_KEY="AIzaSyCIPXhApp0pcu7TruZ8EyuW086VJ1wzrhk"
SYNTH_LLM_TEMPERATURE=0.7
SYNTH_LLM_NUM_FEWSHOT=5

# Parameters derived from above

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid dataset name"
    exit 1
fi

# Run script
python data_processing.py \
    --task_name "$TASK_NAME" \
    --seed "$SEED" \
    --hf_repo "$HF_REPO" \
    --synthetic_hallus_struct $SYNTH_STRUCT \
    --synthetic_hallus_llm $SYNTH_LLM \
    --num_synth_hallus $NUM_SYNTH_HALLUS \
    --gemini_api_key "${GEMINI_API_KEY}" \
    --synth_llm_temperature $SYNTH_LLM_TEMPERATURE \
    --synth_llm_num_fewshot $SYNTH_LLM_NUM_FEWSHOT
