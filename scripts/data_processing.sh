#!/usr/bin/env bash

# Parameters
TASK_NAME="$1"
USER="leobianco"
SEED=12345
HF_REPO="${USER}/${TASK_NAME}"
SYNTH_STRUCT="True"
SYNTH_LLM="False"
NUM_SYNTH_HALLUS=70
SYNTH_LLM_TEMPERATURE=0.7
SYNTH_LLM_NUM_FEWSHOT=5
SYNTH_STRUCT_TOP_K="${SYNTH_STRUCT_TOP_K:-3}"
SYNTH_STRUCT_HALLU_THRESHOLD="${SYNTH_STRUCT_HALLU_THRESHOLD:-0.20}"
SYNTH_STRUCT_IRRELEVANT_THRESHOLD="${SYNTH_STRUCT_IRRELEVANT_THRESHOLD:-0.15}"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ]; then
    echo "Invalid dataset name"
    exit 1
fi

if [ -z "${GEMINI_API_KEY}" ]; then
    echo "GEMINI_API_KEY environment variable is not set. Please export it before running this script."
    exit 1
fi

# Run script
python3 -m src.data_processing \
    --task_name "$TASK_NAME" \
    --seed "$SEED" \
    --hf_repo "$HF_REPO" \
    --synthetic_hallus_struct $SYNTH_STRUCT \
    --synthetic_hallus_llm $SYNTH_LLM \
    --num_synth_hallus $NUM_SYNTH_HALLUS \
    --gemini_api_key "${GEMINI_API_KEY}" \
    --synth_llm_temperature $SYNTH_LLM_TEMPERATURE \
    --synth_llm_num_fewshot $SYNTH_LLM_NUM_FEWSHOT \
    --synth_struct_top_k "$SYNTH_STRUCT_TOP_K" \
    --synth_struct_hallu_threshold "$SYNTH_STRUCT_HALLU_THRESHOLD" \
    --synth_struct_irrelevant_threshold "$SYNTH_STRUCT_IRRELEVANT_THRESHOLD"
