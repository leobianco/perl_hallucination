#!/usr/bin/env bash

# Parameters
TASK_NAME="$1"
USER="leobianco"
SEED=12345
HF_REPO="${USER}/${TASK_NAME}"
SYNTH_STRUCT="True"
SYNTH_LLM="False"
if [[ "$TASK_NAME" == ragtruth* ]]; then
    # -1 means "no extra cap on the hallucinated synthesis arm"; the arm is
    # already bounded by SYNTH_PERTURB_FRACTION and the per-source caps below.
    NUM_SYNTH_HALLUS="${NUM_SYNTH_HALLUS:--1}"
else
    NUM_SYNTH_HALLUS="${NUM_SYNTH_HALLUS:-70}"
fi
SYNTH_LLM_TEMPERATURE=0.7
SYNTH_LLM_NUM_FEWSHOT=5
SYNTH_STRUCT_TOP_K=3
SYNTH_STRUCT_HALLU_THRESHOLD=0.40
SYNTH_STRUCT_IRRELEVANT_THRESHOLD=0.15

# Leakage controls (RAGTruth). Every split below is made over source_id, never
# over rows: one RAGTruth source yields up to six responses that all share the
# same prompt, so a row-level split would put the same prompt on both sides.
# The SFT / PE-RL / reward-model blocks are mutually disjoint.
DROP_BAD_QUALITY="${DROP_BAD_QUALITY:-True}"
SPLIT_FRAC_SFT="${SPLIT_FRAC_SFT:-0.40}"
SPLIT_FRAC_PERL="${SPLIT_FRAC_PERL:-0.25}"   # reward model gets the remainder
SFT_VAL_FRACTION="${SFT_VAL_FRACTION:-0.15}"
PERL_NUM_TEST_PROMPTS="${PERL_NUM_TEST_PROMPTS:-50}"
SYNTH_STRUCT_MAX_RESPONSES_PER_SOURCE="${SYNTH_STRUCT_MAX_RESPONSES_PER_SOURCE:-2}"
SYNTH_STRUCT_MAX_HALLU_PER_ENTRY="${SYNTH_STRUCT_MAX_HALLU_PER_ENTRY:-1}"
SYNTH_PERTURB_FRACTION="${SYNTH_PERTURB_FRACTION:-0.5}"

# Checks
if [ "$TASK_NAME" != "npov" ] && [ "$TASK_NAME" != "bosch" ] && [ "$TASK_NAME" != "ragtruth" ] && [ "$TASK_NAME" != "ragtruth-qa" ] && [ "$TASK_NAME" != "ragtruth-summarization" ]; then
    echo "Invalid dataset name: $TASK_NAME"
    echo "Valid choices: npov, bosch, ragtruth, ragtruth-qa, ragtruth-summarization"
    exit 1
fi

if [ "$SYNTH_LLM" == "True" ] && [ -z "${GEMINI_API_KEY}" ]; then
    echo "GEMINI_API_KEY environment variable is not set. Please export it before running this script with SYNTH_LLM=True."
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
    --synth_struct_irrelevant_threshold "$SYNTH_STRUCT_IRRELEVANT_THRESHOLD" \
    --drop_bad_quality "$DROP_BAD_QUALITY" \
    --split_frac_sft "$SPLIT_FRAC_SFT" \
    --split_frac_perl "$SPLIT_FRAC_PERL" \
    --sft_val_fraction "$SFT_VAL_FRACTION" \
    --perl_num_test_prompts "$PERL_NUM_TEST_PROMPTS" \
    --synth_struct_max_responses_per_source "$SYNTH_STRUCT_MAX_RESPONSES_PER_SOURCE" \
    --synth_struct_max_hallu_per_entry "$SYNTH_STRUCT_MAX_HALLU_PER_ENTRY" \
    --synth_perturb_fraction "$SYNTH_PERTURB_FRACTION"
