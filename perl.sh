#!/bin/bash

# If calling from hyperparameter search script,	variables are imported.
if [ $SHLVL -gt 2 ]; then
  :
else
  # If single-run, set variables here.
  SEED=130104
  NUM_TRAIN_EPOCHS=5
  LEARNING_RATE=1e-4
  LORA_RANK=4
  KL_COEFF=5e-2
  RLOO_K=2
  NUM_PPO_EPOCHS=2
  NUM_MINI_BATCHES=2
  TOTAL_EPISODES=5000
  RUN_IDENTIFIER="leobianco/HALOMI_PERL_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lora_${LORA_RANK}_lr_${LEARNING_RATE}_klcoeff_${KL_COEFF}_rlook_${RLOO_K}_ppoepochs_{$NUM_PPO_EPOCHS}_minibatches_${NUM_MINI_BATCHES}_episodes_${TOTAL_EPISODES}"
fi

accelerate launch \
--config_file=/home/leobianco/.cache/huggingface/accelerate/zero3.yaml \
perl.py \
-- \
--seed $SEED \
--report_to "wandb" \
--run_name $RUN_IDENTIFIER \
--logging_steps 10 \
--output_dir "./checkpoints/halomi/perl/${RUN_IDENTIFIER}" \
--overwrite_output_dir True \
--push_to_hub True \
--hub_model_id $RUN_IDENTIFIER \
--dataset_name "leobianco/perl_halomi_processed" \
--model_identifier "google/gemma-2-2b-it" \
--do_train True \
--save_strategy "no" \
--num_train_epochs $NUM_TRAIN_EPOCHS \
--learning_rate $LEARNING_RATE \
--weight_decay 0.0 \
--max_seq_len 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--do_eval True \
--eval_strategy "steps" \
--eval_steps 50 \
--per_device_eval_batch_size 1 \
--eval_accumulation_steps 1 \
--reward_model_path "leobianco/halomi_reward_model" \
--sft_model_path "leobianco/halomi_writer_sft" \
--r $LORA_RANK \
--kl_coef $KL_COEFF \
--rloo_k $RLOO_K \
--num_ppo_epochs $NUM_PPO_EPOCHS \
--num_mini_batches $NUM_MINI_BATCHES \
--total_episodes $TOTAL_EPISODES \
--missing_eos_penalty 1.0 \
