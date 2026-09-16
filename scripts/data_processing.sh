#!/usr/bin/env bash

# Parameters
TASK_NAME="$1"
USER="leobianco"
SEED=12345
HF_REPO="${USER}/${TASK_NAME}"
# Synthetic hallucination generation.
#
# SYNTH_STRUCT builds an ADDITIONAL reward-model dataset (`*_rm_synthetic_struct`)
# by structurally perturbing examples. It does not replace the organic one
# (`*_rm_organic`); which dataset a run trains on is chosen by reward_model.sh.
# Keeping both is the point: organic-RM vs synthetic-RM is a treatment
# comparison, not a fallback.
#
# Worth knowing for RAGTruth specifically, since it is the one corpus here
# that does NOT need fabricated positives:
#
#   * RAGTruth is already a hallucination corpus. The QA train block alone
#     carries ~1,460 human-annotated hallucinated responses, so the synthetic
#     arm is not here to manufacture volume. Turn SYNTH_STRUCT off if you only
#     want the organic condition; nothing downstream breaks.
#
#   * The obvious objection to structural perturbation is that erasing context
#     to create a hallucination makes hallucinated examples systematically
#     shorter, handing the reward model a length shortcut that will not exist
#     at PE-RL time. That does not apply here, by construction:
#       - schema 1 erases SUPPORTING context (high ROUGE) -> hallucinated arm
#       - schema 2 erases IRRELEVANT context (low ROUGE)  -> faithful arm
#     Both shorten the context, so length does not separate the classes. The
#     faithful arm also keeps the untouched original.
#
#   * The two arms are disjoint by source_id (see SYNTH_PERTURB_FRACTION), so
#     no passage appears as both a perturbed and a clean example.
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

# The official RAGTruth test split is carved, again by source_id, into a
# 'dev' pool and a 'final' pool.
#
#   dev   -> reward-model eval arm, autorater threshold, PE-RL eval prompts,
#            i.e. everything that influences which checkpoint or which sweep
#            trial gets picked.
#   final -> the reported evaluation set, and nothing else.
#
# Being absent from training is not the same as being held out. Every stage
# above selects a model by looking at its score, and the orchestrator ranks
# sweep trials on their BEST step, so scoring the final result on the same
# rows reports the maximum of many noisy estimates instead of generalisation.
#
# Sizing. RAGTruth ships exactly 150 held-out test SOURCE passages per subtask
# (900 test responses = 150 sources x 6 generator LLMs), and every consumer
# dedupes to one prompt per source, so 150 unique evaluation prompts is a hard
# ceiling -- no setting of this variable can produce more. 1/3 spends 50
# sources on tuning and leaves 100 for the reported number.
TEST_DEV_FRACTION="${TEST_DEV_FRACTION:-0.333333}"
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
    --test_dev_fraction "$TEST_DEV_FRACTION" \
    --synth_struct_max_responses_per_source "$SYNTH_STRUCT_MAX_RESPONSES_PER_SOURCE" \
    --synth_struct_max_hallu_per_entry "$SYNTH_STRUCT_MAX_HALLU_PER_ENTRY" \
    --synth_perturb_fraction "$SYNTH_PERTURB_FRACTION"
