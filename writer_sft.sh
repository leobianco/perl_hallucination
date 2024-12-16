SEED=130401
NUM_TRAIN_EPOCHS=10
LEARNING_RATE=1e-4
LORA_RANK=4
RUN_IDENTIFIER="leobianco/HALOMI_SFT_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
accelerate launch \
--config_file=/home/leobianco/.cache/huggingface/accelerate/zero3.yaml \
writer_sft.py \
-- \
--report_to "wandb" \
--run_name $RUN_IDENTIFIER \
--logging_steps 10 \
--output_dir "./checkpoints/halomi/writer_sft/${RUN_IDENTIFIER}" \
--overwrite_output_dir True \
--push_to_hub True \
--hub_model_id $RUN_IDENTIFIER \
--seed $SEED \
--dataset_name "leobianco/writer_sft_halomi_processed" \
--model_identifier "google/gemma-2-2b-it" \
--do_train True \
--save_strategy "no" \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--learning_rate $LEARNING_RATE \
--weight_decay 0.0 \
--max_seq_length 512 \
--dataset_text_field "prompt" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--do_eval False \
--eval_strategy "no" \
--per_device_eval_batch_size 1 \
--eval_accumulation_steps 1 \
--task_type "CAUSAL_LM" \
--r $LORA_RANK \
