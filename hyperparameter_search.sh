#!/bin/bash

# Hyperparameters common to all
declare -x -r USER="leobianco"
declare -x -r DEEPSPEED_CONFIG="./deepspeed_config.yaml"
declare -x -r EXPERIMENT_TYPE="HALOMI_RM"
declare -x -i SEED=130104
declare -x -a NEPOCHS=(10)
declare -x -a LEARNING_RATES=(1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3)
declare -x -a LORA_RANKS=(4 8 16 32)

# PERL specific hyperparameters
declare -x -a KL_COEFFS=(5e-2)
declare -x -a RLOO_K_GRID=(2)
declare -x -a NUM_PPO_EPOCHS_GRID=(2)
declare -x -a NUM_MINIBATCHES_GRID=(2)
declare -x -a TOTAL_EPISODES_GRID=(5000)

# Loop through grid and execute scripts
for NUM_TRAIN_EPOCHS in "${NEPOCHS[@]}"; do
  for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    for LORA_RANK in "${LORA_RANKS[@]}"; do

      export NUM_TRAIN_EPOCHS LEARNING_RATE LORA_RANK
      export RUN_IDENTIFIER="${USER}/${EXPERIMENT_TYPE}_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"

      if [ "${EXPERIMENT_TYPE}" = "HALOMI_RM" ]; then
        echo "Run identifier: ${RUN_IDENTIFIER}"
        /bin/bash ./reward_model.sh
      elif [ "${EXPERIMENT_TYPE}" = "HALOMI_SFT" ]; then
        echo "Run identifier: ${RUN_IDENTIFIER}"
        /bin/bash ./writer_sft.sh
        /bin/bash ./evaluator.sh
      elif [ "${EXPERIMENT_TYPE}" = "HALOMI_PERL" ]; then
        for KL_COEFF in "${KL_COEFFS[@]}"; do
          for RLOO_K in "${RLOO_K_GRID[@]}"; do
            for NUM_PPO_EPOCHS in "${NUM_PPO_EPOCHS_GRID[@]}"; do 
              for NUM_MINIBATCHES in "${NUM_MINIBATCHES_GRID[@]}"; do
                for TOTAL_EPISODES in "${TOTAL_EPISODES_GRID[@]}"; do

		  export KL_COEFF RLOO_K NUM_PPO_EPOCHS NUM_MINIBATCHES TOTAL_EPISODES
    	          # Append PERL-specific hyperparameters to run identifier.
  	          export RUN_IDENTIFIER="${RUN_IDENTIFIER}_klcoeff_${KL_COEFF}_rlook_${RLOO_K}_episodes_${TOTAL_EPISODES}"
                  echo "Run identifier: ${RUN_IDENTIFIER}"
                  /bin/bash ./perl.sh
                  /bin/bash ./evaluator.sh
                done
              done
            done
          done
	done
      fi
    done
  done
done

# Uncomment to shutdown instance after finished
# sudo shutdown -h now
