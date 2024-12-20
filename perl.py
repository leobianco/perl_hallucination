"""
TODO: write proper docstring.
"""

import os
from dataclasses import dataclass

from transformers import (
  HfArgumentParser, set_seed, AutoTokenizer, AutoModelForSequenceClassification,
  AutoModelForCausalLM,
)
from trl import ModelConfig, RLOOConfig, RLOOTrainer
from peft import (
  AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM
)

from data import *
from writer_sft import CustomLoraConfig
from utils import ScriptArguments


def main():

  #########
  # SETUP #
  #########

  parser = HfArgumentParser(
    (
      ScriptArguments,
      RLOOConfig,
      CustomLoraConfig,
    )
  )

  (
    script_args,
    training_args,
    peft_args,
  ) = parser.parse_args_into_dataclasses()

  name_for_saving = training_args.run_name.split("/")[1]

  # Set seed before instantiating the model, for reproducibility.
  set_seed(training_args.seed)

  tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_identifier,
    padding_side="right",
  )

  ########
  # DATA #
  ########

  perl_data_halomi = load_dataset(script_args.dataset_name)

  perl_data_halomi["train"] = (
    perl_data_halomi["train"]
    .select_columns(["input_ids", "attention_mask"])
  )

  ################
  # REWARD MODEL #
  ################

  id2label = {
    0: 'Yes',
    1: 'No',
  }
  
  label2id = {
    'Yes': 0,
    'No': 1,
  }

  reward_model = (
    AutoModelForSequenceClassification.from_pretrained(
      training_args.reward_model_path,
      num_labels=2,
      id2label=id2label,
      label2id=label2id,
      attn_implementation="eager",
    )
  )
  
  #############################
  # REFERENCE POLICY + POLICY #
  #############################

  ref_policy = AutoModelForCausalLM.from_pretrained(
    training_args.sft_model_path,
    attn_implementation="eager",
  )
  
  policy = AutoPeftModelForCausalLM.from_pretrained(
    training_args.sft_model_path,
    is_trainable=True,
    attn_implementation="eager",
  )

  ###########
  # TRAINER #
  ###########

  trainer = RLOOTrainer(
    config=training_args,
    processing_class=tokenizer,
    ref_policy=ref_policy,
    policy=policy,
    reward_model=reward_model,
    train_dataset=perl_data_halomi['train'],
    eval_dataset=perl_data_halomi['test'],
  )

  ############
  # TRAINING #
  ############

  if training_args.do_train:
    trainer.train()
    policy.save_pretrained(f"checkpoints/{name_for_saving}/")
    if training_args.push_to_hub:
      policy.push_to_hub(training_args.hub_model_id)

  filepath = f"logs/{name_for_saving}/logs.txt"
  os.makedirs(os.path.dirname(pathname), exist_ok=True)
  with open(filepath, "w") as f:
    for d in trainer.state.log_history:
      f.write(str(d) + "\n----------\n")
  print(f"Logs saved to {filepath}")

if __name__=="__main__":
  main()

