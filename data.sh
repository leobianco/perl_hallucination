#!/usr/bin/env bash

USER="leobianco"
TASK="$1"
SEED=12345
SYNTH_LLM="False"
SYNTH_STRUCT="True"
NUM_SYNTH_HALLUS=50
GEMINI_API_KEY="AIzaSyCIPXhApp0pcu7TruZ8EyuW086VJ1wzrhk"
SYNTH_LLM_TEMPERATURE=0.7
SYNTH_LLM_NUM_FEWSHOT=5

if [ "$TASK" != "ragtruth" ] && [ "$TASK" != "npov" ] && [ "$TASK" != "bosch" ]; then
    echo "Invalid dataset name"
    exit 1
fi

python data.py \
    --task "$TASK" \
    --seed "$SEED" \
    --synthetic_hallus_llm $SYNTH_LLM \
    --synthetic_hallus_struct $SYNTH_STRUCT \
    --num_synth_hallus $NUM_SYNTH_HALLUS \
    --gemini_api_key "${GEMINI_API_KEY}" \
    --synth_llm_temperature $SYNTH_LLM_TEMPERATURE \
    --synth_llm_num_fewshot $SYNTH_LLM_NUM_FEWSHOT
