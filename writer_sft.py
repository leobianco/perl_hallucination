"""TODO: write docstring.
"""


from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import (
  TrlParser, SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM,
)

from data import *
from utils import ScriptArguments


# Work around HfArgumentParser bug...
@dataclass
class CustomLoraConfig(LoraConfig):
  init_lora_weights: bool = field(default=True)
  layers_to_transform: int = field(default=None)
  loftq_config: dict = field(default_factory=dict)


def main():

  #########
  # SETUP #
  #########

  parser = TrlParser((ScriptArguments, SFTConfig, CustomLoraConfig))
  (
    script_args,
    training_args,
    peft_args
  ) = parser.parse_args_into_dataclasses()

  # Set seed before instantiating the model, for reproducibility.
  set_seed(training_args.seed)

  tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_identifier,
    padding_side="right",
  )

  # Train on completions only.
  response_template = "\nTranslated text:"
  response_template_ids = tokenizer.encode(
      response_template,
      add_special_tokens=False
  )
  collator_completions = DataCollatorForCompletionOnlyLM(
      response_template_ids,
      tokenizer=tokenizer,
  )

  ########
  # DATA #
  ########

  sft_data_halomi = load_dataset(script_args.dataset_name, split="train")

  ################
  # WRITER MODEL #
  ################

  model = AutoModelForCausalLM.from_pretrained(
    script_args.model_identifier,
    attn_implementation="eager",
  )

  ###############
  # SFT TRAINER #
  ###############

  trainer = SFTTrainer(
    model,
    args=training_args,
    data_collator=collator_completions,
    train_dataset=sft_data_halomi,
    processing_class=tokenizer,
    peft_config=peft_args,
    formatting_func=formatting_prompts_func,
  )

  if training_args.do_train:
    trainer.train()
    trainer.save_model()
    
    if training_args.push_to_hub:
      trainer.push_to_hub()
  

if __name__=="__main__":
  main()

