"""Data processing functions for the HalOmi and NPOV datasets.

This script contains functions for loading, processing, and saving the HalOmi
and NPOV to Hugging Face Hub. Call this script via the shell script data.sh
with the dataset name as an argument ("halomi" or "npov").
"""

from argparse import ArgumentParser
from copy import deepcopy

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer


#####################
# GENERAL FUNCTIONS #
#####################


def encode(batch, tokenizer=None, max_seq_length=512):
    return tokenizer(
        batch["prompt"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",  # though using map() => set_format("torch") later
    )


####################
# HALOMI FUNCTIONS #
####################


# Preprocessing


def halomi_hallucination_labels_to_numerical(entry):
    """Data processing utility function which replaces the hallucination labels by
    a numerical version."""

    entry["class_hall_num"] = 0 if entry["class_hall"] == "Yes" else 1

    return entry


def halomi_omission_labels_to_numerical(entry):
    """Data processing utility function which replaces the omission labels by
    a numerical version."""

    entry["class_omit_num"] = 0 if entry["class_omit"] == "Yes" else 1

    return entry


def halomi_change_language_labels(entry):
    """Data processing utility function which replaces the language labels by the
    natural language correspondent version.
    """

    lang_to_natural_lang = {
        "eng_Latn": "English",
        "kas_Deva": "Kashmiri",
        "spa_Latn": "Spanish",
        "mni_Beng": "Manipuri",
        "yor_Latn": "Yoruba",
        "deu_Latn": "German",
        "zho_Hans": "Mandarin",
        "arb_Arab": "Modern Standard Arab",
        "rus_Cyrl": "Russian",
    }

    entry["src_lang"] = lang_to_natural_lang[entry["src_lang"]]
    entry["tgt_lang"] = lang_to_natural_lang[entry["tgt_lang"]]

    return entry


def halomi_change_hallucination_labels(entry):
    """Data processing utility function which replaces the hallucination labels by
    a simplified version.
    """

    class_hallucination_to_label = {
        "1_No_hallucination": "No",
        "2_Small_hallucination": "Yes",
        "3_Partial_hallucination": "Yes",
        "4_Full_hallucination": "Yes",
    }

    entry["class_hall"] = class_hallucination_to_label[entry["class_hall"]]

    return entry


def halomi_change_omission_labels(entry):
    """Data processing utility function which replaces the omission labels by
    a simplified version.
    """

    class_omit_to_label = {
        "1_No_omission": "No",
        "2_Small_omission": "Yes",
        "3_Partial_omission": "Yes",
        "4_Full_omission": "Yes",
    }

    entry["class_omit"] = class_omit_to_label[entry["class_omit"]]

    return entry


def halomi_process_data(data_halomi, seed: int = 12345):
    """Preprocesses the HalOmi dataset."""

    # Select relevant subset of columns
    data_halomi = data_halomi.select_columns(
        [
            "src_lang",
            "tgt_lang",
            "src_text",
            "mt_text",
            "class_hall",
            "class_omit",
        ]
    )

    # Restrict ourselves only to English <-> Spanish examples
    included_langs = ["eng_Latn", "spa_Latn"]

    data_halomi = data_halomi.filter(
        lambda example: (
            example["src_lang"] in included_langs
            and example["tgt_lang"] in included_langs
        )
    )

    # Change language labels to natural language
    data_halomi = data_halomi.map(halomi_change_language_labels)

    # Change hallucination labels to 'Yes' or 'No'
    data_halomi = data_halomi.map(halomi_change_hallucination_labels)

    # Add numerical version of hallucination
    data_halomi = data_halomi.map(halomi_hallucination_labels_to_numerical)

    # Change omission labels to 'Yes' or 'No'
    data_halomi = data_halomi.map(halomi_change_omission_labels)

    # Add numerical version of omission
    data_halomi = data_halomi.map(halomi_omission_labels_to_numerical)

    # Shuffle rows to mix languages and examples
    data_halomi = data_halomi.shuffle(seed=seed)

    return data_halomi


def halomi_load_and_process_data(seed: int = 12345):
    """Loads HalOmi data that is saved in my HF Hub."""

    data_halomi = load_dataset(
        "leobianco/halomi",
        data_files={"train": "halomi_full.tsv"},
        sep="\t",
        split="train",
    )

    data_halomi = halomi_process_data(data_halomi, seed)

    return data_halomi


# Reward Model


def halomi_rm_prompt(entry):
    """Function for transforming entries in the HalOmi dataset into training
    prompts for the reward model.
    """

    template = (
        "<start_of_turn>user\n"
        "Translate a text originally written in {src_lang} into {tgt_lang}. "
        "Generate only the translated text, and nothing else."
        "\nOriginal text: {src_text}"
        "\nTranslated text:<end_of_turn>\n<start_of_turn>model\n{mt_text}<end_of_turn><eos>"
    )

    formatted_prompt = template.format(
        src_lang=entry["src_lang"],
        tgt_lang=entry["tgt_lang"],
        src_text=entry["src_text"],
        mt_text=entry["mt_text"],
    )

    entry["prompt"] = formatted_prompt

    return entry


def halomi_process_data_for_rm(
    halomi_data,
    tokenizer,
    max_seq_length=512,
    validation_size=0.25,
    seed=12345,
):
    # Copy original Halomi data
    halomi_rm_data = deepcopy(halomi_data)

    # Create reward model HalOmi prompts
    halomi_rm_data = halomi_rm_data.map(halomi_rm_prompt)

    # Tokenize reward model prompts
    halomi_rm_data = halomi_rm_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    halomi_rm_data.set_format("torch")  # due to using map()

    # Rename hallucination class to label
    halomi_rm_data = halomi_rm_data.rename_column("class_hall_num", "label")

    # Split the dataset
    halomi_rm_data = halomi_rm_data.train_test_split(
        seed=seed, test_size=validation_size
    )

    return halomi_rm_data


# Writer SFT


def halomi_process_data_for_sft(halomi_processed_data):
    """Data processing utility function which filters the HalOmi dataset to only
    include examples that are not hallucinations or omissions.
    """

    writer_sft_processed = halomi_processed_data.filter(
        lambda example: (
            example["class_hall"] == "No" and example["class_omit"] == "No"
        )
    )

    return writer_sft_processed


def halomi_writer_prompt(entry, SFT=False):
    """Function for transforming entries in the HalOmi dataset into prompts for
    the writer to translate.
    """

    template = (
        "<start_of_turn>user\n"
        "Translate a text originally written in {src_lang} into {tgt_lang}. "
        "Generate only the translated text, and nothing else."
        "\nOriginal text: {src_text}"
        "\nTranslated text:<end_of_turn>\n<start_of_turn>model\n{ans}"
    )

    ans = (entry["mt_text"] + "<end_of_turn><eos>") if SFT else ""

    formatted_prompt = template.format(
        src_lang=entry["src_lang"],
        tgt_lang=entry["tgt_lang"],
        src_text=entry["src_text"],
        ans=ans,
    )

    entry["prompt"] = formatted_prompt

    return entry


def halomi_formatting_prompts_func(entry):
    """Formatting function for SFTTrainer. Imported in writer_sft.py."""

    template = (
        "<start_of_turn>user\n"
        "Translate a text originally written in {src_lang} into {tgt_lang}. "
        "Generate only the translated text, and nothing else."
        "\nOriginal text: {src_text}"
    )

    output_texts = []

    for i in range(len(entry["src_text"])):
        formatted_prompt = template.format(
            src_lang=entry["src_lang"][i],
            tgt_lang=entry["tgt_lang"][i],
            src_text=entry["src_text"][i],
        )

        text = f"{formatted_prompt}\nTranslated text:<end_of_turn>\n<start_of_turn>model\n{entry['mt_text'][i]}"

        output_texts.append(text)

    return output_texts


# PERL


def halomi_format_perl_data(entry, src_lang: str, tgt_lang: str):
    """Helper function for formatting PERL data, to be mapped over dataset."""

    entry["src_lang"] = src_lang
    entry["tgt_lang"] = tgt_lang
    entry["src_text"] = entry[src_lang]
    entry["mt_text"] = entry[tgt_lang]

    return entry


def halomi_process_data_for_perl(
    halomi_perl_data,
    tokenizer,
    max_seq_length=512,
    seed=12345,
    perl_train_size=1000,
    perl_validation_size=500,
):
    # Half of it will be English -> Spanish, the other half Spanish -> English.
    n = halomi_perl_data.num_rows
    first_half = halomi_perl_data.select(range(n // 2))
    second_half = halomi_perl_data.select(range(n // 2, n))

    first_half = first_half.map(
        halomi_format_perl_data,
        fn_kwargs=dict(src_lang="English", tgt_lang="Spanish"),
    )

    second_half = second_half.map(
        halomi_format_perl_data,
        fn_kwargs=dict(src_lang="Spanish", tgt_lang="English"),
    )

    halomi_perl_data = concatenate_datasets([first_half, second_half])
    halomi_perl_data = halomi_perl_data.shuffle(seed=seed)

    # Create and tokenize prompts for writer.
    halomi_perl_data = halomi_perl_data.map(halomi_writer_prompt)

    halomi_perl_data = halomi_perl_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    halomi_perl_data.set_format("torch")  # due to using map()

    # Split into training, validation, and test splits.
    halomi_perl_data = halomi_perl_data.train_test_split(
        seed=seed,
        train_size=perl_train_size,
        test_size=perl_validation_size,
    )

    return halomi_perl_data


##################
# NPOV FUNCTIONS #
##################


# Preprocessing


def npov_hallucination_labels_to_numerical(entry):
    """Data processing utility function which replaces the hallucination labels by
    a numerical version."""

    entry["class_hall_num"] = 0 if entry["has hallucination"] == "Yes" else 1

    return entry


def npov_omission_labels_to_numerical(entry):
    """Data processing utility function which replaces the omission labels by
    a numerical version."""

    entry["class_omit_num"] = 0 if entry["has coverage issue"] == "Yes" else 1

    return entry


def npov_change_hallucination_labels(entry):
    """Data processing utility function which replaces the hallucination labels by
    a simplified version.
    """

    class_hallucination_to_label = {
        "NO": "No",
        "YES": "Yes",
    }

    entry["has hallucination"] = class_hallucination_to_label[
        entry["has hallucination"]
    ]

    entry["has synthetic hallucination"] = class_hallucination_to_label[
        entry["has synthetic hallucination"]
    ]

    return entry


def npov_change_omission_labels(entry):
    """Data processing utility function which replaces the omission labels by
    a simplified version.
    """

    class_omission_to_label = {
        "NO": "No",
        "YES": "Yes",
    }

    entry["has coverage issue"] = class_omission_to_label[
        entry["has coverage issue"]
    ]

    entry["has synthetic coverage issue"] = class_omission_to_label[
        entry["has synthetic coverage issue"]
    ]

    return entry


def npov_process_data(npov_data):
    """Preprocesses NPOV dataset."""

    # Select relevant subset of columns
    npov_data = npov_data.select_columns(
        [
            "topic",
            "user_query",
            "npov_response",
            "perspective_1",
            "perspective_1_name",
            "perspective_2",
            "perspective_2_name",
            "has hallucination",
            "has synthetic hallucination",
            "has coverage issue",
            "has synthetic coverage issue",
        ]
    )

    # Apply all preprocessing functions.
    npov_data = npov_data.map(npov_change_hallucination_labels)
    npov_data = npov_data.map(npov_change_omission_labels)
    npov_data = npov_data.map(npov_hallucination_labels_to_numerical)
    npov_data = npov_data.map(npov_omission_labels_to_numerical)

    return npov_data


def npov_load_and_process_data():
    """Loads NPOV data that is saved in my HF Hub."""

    npov_data = load_dataset(
        "leobianco/npov",
        data_files={
            "train": "hc_rm5x_train.json",
            "validation": "hc_rm5x_validation.json",
            "test": "hc_rm5x_test.json",
        },
    )

    for split in npov_data.keys():
        npov_data[split] = npov_process_data(npov_data[split])

    return npov_data


# Reward Model


def npov_rm_prompt(entry):
    """Function for transforming entries in the NPOV dataset into training
    prompts for the reward model.
    """

    template = (
        "<start_of_turn>user\n"
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}\n"
        "Neutral point-of-view rewriting of provided arguments in natural "
        "language:<end_of_turn>\n"
        "<start_of_turn>model\n{npov_response}<end_of_turn><eos>"
    )

    formatted_prompt = template.format(
        user_query=entry["user_query"],
        perspective_1_name=entry["perspective_1_name"],
        perspective_1=entry["perspective_1"],
        perspective_2_name=entry["perspective_2_name"],
        perspective_2=entry["perspective_2"],
        npov_response=entry["npov_response"],
    )

    entry["prompt"] = formatted_prompt

    return entry


def npov_process_data_for_rm(
    npov_data,
    tokenizer,
    max_seq_length=512,
    validation_size=0.25,
    seed=12345,
):
    # Copy original NPOV data
    npov_rm_data = deepcopy(npov_data)

    # Create reward model NPOV prompts
    npov_rm_data = npov_rm_data.map(npov_rm_prompt)

    # Tokenize reward model prompts
    npov_rm_data = npov_rm_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    npov_rm_data.set_format("torch")  # due to using map()

    # Rename hallucination class to label
    npov_rm_data = npov_rm_data.rename_column("class_hall_num", "label")

    return npov_rm_data


# Writer SFT


def npov_process_data_for_sft(npov_data):
    """Data processing utility function which filters the NPOV dataset to only
    include examples that are not hallucinations or omissions.
    """

    writer_sft_processed = npov_data.filter(
        lambda example: (
            example["has hallucination"] == "No"
            and example["has synthetic hallucination"] == "No"
            and example["has coverage issue"] == "No"
            and example["has synthetic coverage issue"] == "No"
        )
    )

    return writer_sft_processed


def npov_writer_prompt():
    pass


def npov_formatting_prompts_func():
    pass


# PERL


def npov_format_perl_data():
    pass


def npov_process_data_for_perl(npov_data):
    pass


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--tokenizer_model", type=str, default="google/gemma-2-2b-it"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--raw_repo_id", type=str)
    parser.add_argument("--processed_repo_id", type=str)
    parser.add_argument("--rm_processed_repo_id", type=str)
    parser.add_argument("--rm_validation_size", type=float, default=0.2)
    parser.add_argument("--writer_sft_processed_repo_id", type=str)
    parser.add_argument("--perl_raw_repo_id", type=str)
    parser.add_argument("--perl_processed_repo_id", type=str)
    parser.add_argument("--perl_train_size", type=int)
    parser.add_argument("--perl_validation_size", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        padding_side="left",  # Pay attention to this argument!
    )

    if args.dataset == "halomi":
        halomi_processed_data = halomi_load_and_process_data(args.seed)

        halomi_processed_data.push_to_hub(
            repo_id=args.processed_repo_id,
        )

        halomi_rm_processed_data = halomi_process_data_for_rm(
            halomi_processed_data,
            tokenizer,
            max_seq_length=args.max_seq_length,
            validation_size=args.rm_validation_size,
            seed=args.seed,
        )

        for split in halomi_rm_processed_data.keys():
            halomi_rm_processed_data[split].push_to_hub(
                repo_id=args.rm_processed_repo_id,
                split=split,
            )

        halomi_writer_sft_processed = halomi_process_data_for_sft(
            halomi_processed_data
        )

        halomi_writer_sft_processed.push_to_hub(
            repo_id=args.writer_sft_processed_repo_id,
        )

        halomi_perl_data = load_dataset(
            args.perl_raw_repo_id,
            split="train",
        )

        halomi_perl_data_processed = halomi_process_data_for_perl(
            halomi_perl_data,
            tokenizer,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
            perl_train_size=args.perl_train_size,
            perl_validation_size=args.perl_validation_size,
        )

        for split in halomi_perl_data_processed.keys():
            halomi_perl_data_processed[split].push_to_hub(
                repo_id=args.perl_processed_repo_id,
                split=split,
            )

    elif args.dataset == "npov":
        npov_data_processed = npov_load_and_process_data()
        for split in npov_data_processed.keys():
            npov_data_processed[split].push_to_hub(
                repo_id=args.processed_repo_id,
                split=split,
            )

        npov_rm_processed_data = {}
        for split in npov_data_processed.keys():
            npov_rm_processed_data[split] = npov_process_data_for_rm(
                npov_data_processed[split],
                tokenizer,
                max_seq_length=args.max_seq_length,
                validation_size=args.rm_validation_size,
                seed=args.seed,
            )

            npov_rm_processed_data[split].push_to_hub(
                repo_id=args.rm_processed_repo_id,
                split=split,
            )


if __name__ == "__main__":
    main()
