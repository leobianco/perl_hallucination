"""TODO: write docstring."""


from datasets import load_dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

from data import formatting_prompts_func
from utils import CustomLoraConfig, ScriptArguments


def main():
    #########
    # SETUP #
    #########

    parser = TrlParser((ScriptArguments, SFTConfig, CustomLoraConfig))
    (script_args, training_args, peft_args) = (
        parser.parse_args_into_dataclasses()
    )

    name_for_saving = training_args.run_name.split("/")[1]

    # Set seed before instantiating the model, for reproducibility.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_identifier,
        padding_side="left",
    )

    # Train on completions only.
    response_template = (
        "\nTranslated text:<end_of_turn>\n<start_of_turn>model\n"
    )
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
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
        model.save_pretrained(f"checkpoints/{name_for_saving}/")

        if training_args.push_to_hub:
            model.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    main()
