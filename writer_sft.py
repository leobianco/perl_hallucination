"""TODO: write docstring."""

from datasets import load_dataset
from peft import get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import (
    DataCollatorForCompletionOnlyLM,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

from data import (
    halomi_formatting_prompts_func,
    npov_formatting_prompts_func,
    npov_formatting_prompts_func_from_fewshot_examples,
    owkin_formatting_prompts_func,
)
from utils import CustomLoraConfig, ScriptArguments


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, CustomLoraConfig))
    (script_args, training_args, peft_args) = (
        parser.parse_args_into_dataclasses()
    )
    set_seed(training_args.seed)

    sft_data = load_dataset(script_args.dataset_repo_id, split="train")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_repo_id,
        padding_side="left",
    )

    # For training on completions only.
    if script_args.task == "halomi":
        response_template = (
            "\nTranslated text:<end_of_turn>\n<start_of_turn>model\n"
        )
        formatting_prompts_func = halomi_formatting_prompts_func
    elif script_args.task == "npov":
        response_template = "Neutral point-of-view answer to user query, rewriting provided arguments in natural language:<end_of_turn>\n<start_of_turn>model\n"
        if script_args.num_fewshot == 0 or script_args.num_fewshot is None:
            formatting_prompts_func = npov_formatting_prompts_func
        else:
            # Get fewshot examples and add to formatting_prompts_func
            fewshot_examples = sft_data.shuffle(seed=training_args.seed).select(
                range(script_args.num_fewshot)
            )
            formatting_prompts_func = (
                npov_formatting_prompts_func_from_fewshot_examples(
                    fewshot_examples=fewshot_examples
                )
            )
    elif script_args.task == "owkin":
        response_template = "Output:<end_of_turn>\n<start_of_turn>model\n"
        formatting_prompts_func = owkin_formatting_prompts_func
    else:
        raise ValueError("Invalid dataset.")

    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )
    collator_completions = DataCollatorForCompletionOnlyLM(
        response_template_ids,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_repo_id,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    model = get_peft_model(model, peft_args)

    trainer = SFTTrainer(
        model,
        args=training_args,
        data_collator=collator_completions,
        train_dataset=sft_data,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )

    if training_args.do_train:
        trainer.train()


if __name__ == "__main__":
    main()
