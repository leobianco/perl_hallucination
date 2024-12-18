#!/bin/bash

# Hyperparameters
export USER="leobianco"
export EXPERIMENT_TYPE="HALOMI_SFT"
export SEED=130104
declare -a NEPOCHS=(10)
declare -a LEARNING_RATES=(2e-6 5e-6)
declare -a LORA_RANKS=(4)

# Loop through grid and execute scripts
for NUM_TRAIN_EPOCHS in "${NEPOCHS[@]}"; do
  export NUM_TRAIN_EPOCHS
  for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
    export LEARNING_RATE
    for LORA_RANK in "${LORA_RANKS[@]}"; do
      export LORA_RANK

      export RUN_IDENTIFIER="${USER}/${EXPERIMENT_TYPE}_seed_${SEED}_epochs_${NUM_TRAIN_EPOCHS}_lr_${LEARNING_RATE}_lora_${LORA_RANK}"
      echo "Run identifier: ${RUN_IDENTIFIER}"

      /bin/bash ./writer_sft.sh
      /bin/bash ./evaluator.sh
    done
  done
done

# Uncomment to shutdown instance after finished
sudo shutdown -h now
