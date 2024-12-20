#!/bin/bash

# Hyperparameters common to all
declare -x -r USER="leobianco"
declare -x -r DEEPSPEED_CONFIG="./deepspeed_config.yaml"
declare -x -r EXPERIMENT_TYPE="HALOMI_RM"
declare -x -i SEED=130104
declare -x -a NEPOCHS=(1 3 5)
declare -x -a LEARNING_RATES=(5e-5 5e-4)
declare -x -a LORA_RANKS=(8 16)

# PERL specific hyperparameters
declare -x -a KL_COEFF=(5e-2)
declare -x -a RLOO_K=(2)
declare -x -a NUM_PPO_EPOCHS=(2)
declare -x -a NUM_MINI_BATCHES=(2)
declare -x -a TOTAL_EPISODES=(5000)

# Loop through grid and execute scripts
for NUM_TRAIN_EPOCHS in "${NEPOCHS[@]}"; do
  for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for LORA_RANK in "${LORA_RANKS[@]}"; do

      export RUN_IDENTIFIER="${USER}/${EXPERIMENT_TYPE}_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
      echo "Run identifier: ${RUN_IDENTIFIER}"

      if [ "${EXPERIMENT_TYPE}" = "HALOMI_RM" ]; then
        /bin/bash ./reward_model.sh
      elif [ "${EXPERIMENT_TYPE}" = "HALOMI_SFT" ]; then
        /bin/bash ./writer_sft.sh
        /bin/bash ./evaluator.sh
      elif [ "${EXPERIMENT_TYPE}" = "HALOMI_PERL" ]; then
	# Append PERL-specific hyperparameters to run identifier.
	export RUN_IDENTIFIER="${RUN_IDENTIFIER}_klcoeff_${KL_COEFF}_rlook_${RLOO_K}_ppoepochs_${NUM_PPO_EPOCHS}_minibatches_${NUM_MINI_BATCHES}_episodes_${TOTAL_EPISODES}"
        /bin/bash ./perl.sh
        /bin/bash ./evaluator.sh
      fi
    done
  done
done

# Uncomment to shutdown instance after finished
# sudo shutdown -h now
