"""TODO: write docstring.
"""


from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import (
  TrlParser, SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM,
)

from data import *
from utils import ScriptArguments, CustomLoraConfig


def formatting_prompts_func(entry):
  """Formatting function for training examples passed to SFTTrainer."""

  template = (
    "Translate a text originally written in {src_lang} into {tgt_lang}. "
    "Generate only the translated text, and nothing else."
    "\nOriginal text: {src_text}"
  )

  output_texts = []

  for i in range(len(entry['src_text'])):
    formatted_prompt = template.format(
        src_lang=entry["src_lang"][i],
        tgt_lang=entry["tgt_lang"][i],
        src_text=entry["src_text"][i],
    )

    text = (
        f"{formatted_prompt}\nTranslated text: {entry['completion'][i]}"
    )

    output_texts.append(text)

  return output_texts


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

  model = get_peft_model(model, peft_args)

  ###############
  # SFT TRAINER #
  ###############

  trainer = SFTTrainer(
    model,
    args=training_args,
    data_collator=collator_completions,
    train_dataset=sft_data_halomi,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
  )

  if training_args.do_train:
    trainer.train()
    model.save_pretrained(training_args.output_dir)
    
    if training_args.push_to_hub:
      model.push_to_hub(training_args.hub_model_id)
  

if __name__=="__main__":
  main()

