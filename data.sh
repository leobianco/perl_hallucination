#!/usr/bin/env bash

TASK="$1"
USER="leobianco"
SEED=12345
TOKENIZER_MODEL="google/gemma-2-2b-it"
MAX_SEQ_LENGTH=1152
RAW_REPO_ID="${USER}/${TASK}_raw"
PROCESSED_REPO_ID="${USER}/${TASK}_processed"
RM_PROCESSED_REPO_ID="${USER}/${TASK}_rm_processed"
RM_VALIDATION_SIZE=0.5
WRITER_SFT_PROCESSED_REPO_ID="${USER}/${TASK}_writer_sft_processed"
PERL_RAW_REPO_ID="okezieowen/english_to_spanish"
PERL_PROCESSED_REPO_ID="${USER}/${TASK}_perl_processed"
PERL_TRAIN_SIZE=25000
PERL_VALIDATION_SIZE=500
AUGMENTED_REPO_ID="${USER}/${TASK}_augmented_validation"

if [ "$TASK" != "halomi" ] && [ "$TASK" != "npov" ] && [ "$TASK" != "owkin" ]; then
    echo "Invalid dataset name"
    exit 1
fi

python data.py \
    --task "$TASK" \
    --seed "$SEED" \
    --tokenizer_model "$TOKENIZER_MODEL" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --raw_repo_id "$RAW_REPO_ID" \
    --processed_repo_id "$PROCESSED_REPO_ID" \
    --rm_processed_repo_id "$RM_PROCESSED_REPO_ID" \
    --rm_validation_size "$RM_VALIDATION_SIZE" \
    --writer_sft_processed_repo_id "$WRITER_SFT_PROCESSED_REPO_ID" \
    --perl_raw_repo_id "$PERL_RAW_REPO_ID" \
    --perl_processed_repo_id "$PERL_PROCESSED_REPO_ID" \
    --perl_train_size "$PERL_TRAIN_SIZE" \
    --perl_validation_size "$PERL_VALIDATION_SIZE" \
    --augmented_repo_id "$AUGMENTED_REPO_ID"
