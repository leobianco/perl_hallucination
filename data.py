"""Data processing functions for the HalOmi and NPOV datasets.

This script contains functions for loading, processing, and saving the HalOmi
and NPOV to Hugging Face Hub. Call this script via the shell script data.sh
with the dataset name as an argument ("halomi" or "npov").
"""

import random
from argparse import ArgumentParser
from copy import deepcopy
from itertools import combinations

import nltk
import pandas as pd

nltk.download("punkt_tab")
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types
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

    Previously used for SFT, but now used as prompt for generation during PERL.
    The SFT now uses the "halomi_formatting_prompts_func" function, which has
    the same prompt as this one, but is used by the SFTTrainer differently.
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

        text = f"{formatted_prompt}\nTranslated text:<end_of_turn>\n<start_of_turn>model\n{entry['mt_text'][i]}<end_of_turn><eos>"

        output_texts.append(text)

    return output_texts


# PERL


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

    def halomi_format_perl_data(entry, src_lang: str, tgt_lang: str):
        entry["src_lang"] = src_lang
        entry["tgt_lang"] = tgt_lang
        entry["src_text"] = entry[src_lang]
        entry["mt_text"] = entry[tgt_lang]
        return entry

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
        "Neutral point-of-view answer to user query, rewriting provided"
        " arguments in natural language:<end_of_turn>\n"
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
):
    # Copy original NPOV data
    npov_rm_data = deepcopy(npov_data)

    # Select relevant subset of columns
    npov_rm_data = npov_rm_data.select_columns(
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

    npov_rm_data = npov_rm_data.map(npov_change_hallucination_labels)
    npov_rm_data = npov_rm_data.map(npov_change_omission_labels)
    npov_rm_data = npov_rm_data.map(npov_hallucination_labels_to_numerical)
    npov_rm_data = npov_rm_data.map(npov_omission_labels_to_numerical)
    npov_rm_data = npov_rm_data.rename_column("has hallucination", "class_hall")
    npov_rm_data = npov_rm_data.rename_column(
        "has coverage issue", "class_omit"
    )
    npov_rm_data = npov_rm_data.map(npov_rm_prompt)
    npov_rm_data = npov_rm_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    npov_rm_data.set_format("torch")  # due to using map()
    npov_rm_data = npov_rm_data.rename_column("class_hall_num", "label")

    return npov_rm_data


# Writer SFT


def npov_process_data_for_sft(npov_data):
    """Preprocesses NPOV SFT dataset."""

    # Select relevant subset of columns
    npov_data = npov_data.select_columns(
        [
            "topic",
            "user_query",
            "npov_response_combined",
            "perspective_1",
            "perspective_1_name",
            "perspective_2",
            "perspective_2_name",
        ]
    )

    npov_data = npov_data.rename_column(
        "npov_response_combined", "npov_response"
    )

    return npov_data


def npov_writer_prompt(entry, SFT=False, fewshot_examples=None):
    """Function for transforming entries in the NPOV dataset into prompts for
    the writer to rewrite.

    Previously used for SFT, but now used as prompt for generation during PERL.
    The SFT now uses the "npov_formatting_prompts_func" function, which has
    the same prompt as this one, but is used by the SFTTrainer differently.
    """

    template = (
        "<start_of_turn>user\n"
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}\n"
        "Neutral point-of-view answer to user query, rewriting provided"
        " arguments in natural language:<end_of_turn>\n"
        "<start_of_turn>model\n{npov_response}"
    )

    npov_response = (
        (entry["npov_response"] + "<end_of_turn><eos>") if SFT else ""
    )

    formatted_prompt = template.format(
        user_query=entry["user_query"],
        perspective_1_name=entry["perspective_1_name"],
        perspective_1=entry["perspective_1"],
        perspective_2_name=entry["perspective_2_name"],
        perspective_2=entry["perspective_2"],
        npov_response=npov_response,
    )

    prompt = ""

    if fewshot_examples is not None:
        preamble = (
            "<start_of_turn>user\nYour task is to answer an user's query"
            " by rewriting the provided arguments in natural language. Do not "
            "generate arguments other than those provided. "
            f"We provide {fewshot_examples.num_rows} example(s) of what is "
            "expected, then it is your turn.<end_of_turn>\n"
        )

        prompt += preamble

        for fewshot_example in fewshot_examples:
            fewshot_prompt = template.format(
                user_query=fewshot_example["user_query"],
                perspective_1_name=fewshot_example["perspective_1_name"],
                perspective_1=fewshot_example["perspective_1"],
                perspective_2_name=fewshot_example["perspective_2_name"],
                perspective_2=fewshot_example["perspective_2"],
                npov_response=fewshot_example["npov_response"],
            )
            prompt += fewshot_prompt + "<end_of_turn>\n"

        prompt += formatted_prompt

    else:
        prompt += formatted_prompt

    entry["prompt"] = prompt

    return entry


def npov_formatting_prompts_func(entry):
    """Formatting function for SFTTrainer. Imported in writer_sft.py."""

    template = (
        "<start_of_turn>user\n"
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}"
    )

    output_texts = []

    for i in range(len(entry["user_query"])):
        formatted_prompt = template.format(
            user_query=entry["user_query"][i],
            perspective_1_name=entry["perspective_1_name"][i],
            perspective_1=entry["perspective_1"][i],
            perspective_2_name=entry["perspective_2_name"][i],
            perspective_2=entry["perspective_2"][i],
        )

        text = (
            f"{formatted_prompt}\nNeutral point-of-view answer to user query, "
            "rewriting provided arguments in natural language:<end_of_turn>\n"
            f"<start_of_turn>model\n{entry['npov_response'][i]}"
            "<end_of_turn><eos>"
        )

        output_texts.append(text)

    return output_texts


def npov_formatting_prompts_func_from_fewshot_examples(fewshot_examples):
    """Given a fewshot example, returns a formatting prompts function for writer SFT that includes the given fewshot example in its prompt."""

    preamble = (
        "<start_of_turn>user\nYour task is to answer an user's query"
        " by rewriting the provided arguments in natural language. Do not "
        "generate arguments other than those provided. "
        f"We provide {fewshot_examples.num_rows} example(s) of what is "
        "expected, then it is your turn.<end_of_turn>\n"
    )

    prompt = preamble

    template_fewshot = (
        "<start_of_turn>user\n"
        "User query: {user_query}\n"
        "{perspective_1_name} arguments provided: {perspective_1}\n"
        "{perspective_2_name} arguments provided: {perspective_2}\n"
        "Example neutral point-of-view answer to user query, rewriting provided"
        " arguments in natural language:<end_of_turn>\n"
        "<start_of_turn>model\n{npov_response}"
    )

    for fewshot_example in fewshot_examples:
        fewshot_prompt = template_fewshot.format(
            user_query=fewshot_example["user_query"],
            perspective_1_name=fewshot_example["perspective_1_name"],
            perspective_1=fewshot_example["perspective_1"],
            perspective_2_name=fewshot_example["perspective_2_name"],
            perspective_2=fewshot_example["perspective_2"],
            npov_response=fewshot_example["npov_response"],
        )
        prompt += fewshot_prompt + "<end_of_turn>\n"

    def formatting_prompts_func(entry):
        template = (
            "<start_of_turn>user\n"
            "User query: {user_query}\n"
            "{perspective_1_name} arguments provided: {perspective_1}\n"
            "{perspective_2_name} arguments provided: {perspective_2}"
            "\nNeutral point-of-view answer to user query, rewriting provided"
            " arguments in natural language:<end_of_turn>\n"
            "<start_of_turn>model\n{npov_response}<end_of_turn><eos>"
        )

        output_texts = []

        for i in range(len(entry["user_query"])):
            formatted_prompt = template.format(
                user_query=entry["user_query"][i],
                perspective_1_name=entry["perspective_1_name"][i],
                perspective_1=entry["perspective_1"][i],
                perspective_2_name=entry["perspective_2_name"][i],
                perspective_2=entry["perspective_2"][i],
                npov_response=entry["npov_response"][i],
            )

            output_texts.append(prompt + formatted_prompt)

        return output_texts

    return formatting_prompts_func


# PERL


def npov_process_data_for_perl(
    npov_rm_data, npov_sft_data, tokenizer, max_seq_length=512, seed=12345
):
    """
    Processes NPOV data for PERL by creating train and test splits
    from the RM and SFT datasets.
    """

    train_data = npov_rm_data["validation"]
    test_data = npov_sft_data["test"]

    train_data = train_data.shuffle(seed=seed)
    test_data = test_data.shuffle(seed=seed)

    train_data = train_data.map(npov_writer_prompt)
    test_data = test_data.map(npov_writer_prompt)

    # The columns across splits must match.
    train_data = train_data.select_columns(
        [
            "topic",
            "user_query",
            "npov_response",
            "perspective_1",
            "perspective_1_name",
            "perspective_2",
            "perspective_2_name",
            "prompt",
        ]
    )

    train_data = train_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    train_data.set_format("torch")

    test_data = test_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    test_data.set_format("torch")

    npov_perl_data = DatasetDict(
        {
            "train": train_data,
            "test": test_data,
        }
    )

    return npov_perl_data


# Data augmentation


def npov_data_augmentation(data):
    """
    Augment NPOV dataset by generating combinations of arguments
    for different perspectives on each topic.

    Notice that for the moment I have hard coded pairs of arguments.
    Initially, the different topics do not have the same amount of
    arguments. The combinatorics of the thing makes some topics appear
    much more than others.
    """

    topics = set(data["topic"])
    p1_name = data[0]["perspective_1_name"]
    p2_name = data[0]["perspective_2_name"]

    all_topics = []
    all_user_queries = []
    all_p1_arguments = []
    all_p2_arguments = []
    all_p1_names = []
    all_p2_names = []
    result = {}

    def extract_perspective_arguments(
        topic: str, perspective_col: str, perspective_name: str
    ):
        """
        Extract unique arguments for a given topic and perspective (pro or con).
        """

        topic_data = data.filter(lambda x: x["topic"] == topic)[perspective_col]

        arguments = {
            f"{perspective_name}: {arg.strip()}"
            for text in topic_data
            for arg in text.split(f"{perspective_name}:")
            if arg.strip()
        }

        return arguments

    for topic in topics:
        user_query = data.filter(lambda x: x["topic"] == topic)["user_query"][0]

        p1_args = extract_perspective_arguments(topic, "perspective_1", p1_name)
        p2_args = extract_perspective_arguments(topic, "perspective_2", p2_name)

        p1_pairs = list(combinations(p1_args, 2))
        p2_pairs = list(combinations(p2_args, 2))

        for p1_pair in p1_pairs:
            for p2_pair in p2_pairs:
                p1_combined = " ".join(p1_pair)
                p2_combined = " ".join(p2_pair)
                all_p1_arguments.append(p1_combined)
                all_p2_arguments.append(p2_combined)
                all_p1_names.append(p1_name)
                all_p2_names.append(p2_name)
                all_topics.append(topic)
                all_user_queries.append(user_query)

    result = {
        "topic": all_topics,
        "user_query": all_user_queries,
        "perspective_1": all_p1_arguments,
        "perspective_1_name": all_p1_names,
        "perspective_2": all_p2_arguments,
        "perspective_2_name": all_p2_names,
    }

    return result


def bosch_rm_prompt(entry):
    """The dataset already contains a prompt column, which is an instruction for the writer. We just append the generation."""

    entry["prompt"] += entry["response"] + "<end_of_turn><eos>"

    return entry


def bosch_process_data_for_rm(
    data,
    tokenizer,
    seed=12345,
    max_seq_length=1340,
    split_data=True,
    validation_size=0.2,
):
    rm_data = deepcopy(data)

    rm_data = rm_data.map(bosch_rm_prompt)
    rm_data = rm_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    rm_data.set_format("torch")  # due to using map()

    if split_data:
        rm_data = rm_data.train_test_split(seed=seed, test_size=validation_size)

    return rm_data


def bosch_formatting_prompts_func(entry):
    """Formatting function for SFTTrainer. Imported in writer_sft.py."""

    template = (
        "<start_of_turn><user>\nYou are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information given.\n"
        "User question:\n{question}"
        "\nManual information:\n{context}"
        "\nAnswer to user's question:<end_of_turn>\n<start_of_turn><model>\n"
    )

    output_texts = []

    for i in range(len(entry["Question"])):
        formatted_prompt = template.format(
            question=entry["Question"][i],
            context=entry["Context"][i],
        )

        output_texts.append(formatted_prompt)

    return output_texts


def bosch_process_data_for_perl(
    data,
    tokenizer,
    max_seq_length=1340,
    seed=12345,
    validation_size=100,
):
    perl_data = deepcopy(data)

    perl_data = perl_data.map(
        encode,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": max_seq_length,
        },
    )
    perl_data.set_format("torch")  # due to using map()

    perl_data = perl_data.train_test_split(
        seed=seed,
        test_size=validation_size,
    )

    return perl_data


# PREVIOUS (FIRST) PROMPT
# rm_synthetic_hall_llm_prompt = """
# Objective:
# Generate a modified version of a given answer that includes plausible-sounding synthetic hallucinations (information not present in the provided context).

# Task:
# You are given a Question, a Context used to answer it, and an Original Grounded Answer derived solely from that Context. Your task is to rewrite the Original Grounded Answer to create a Modified Answer.

# Requirements for the Modified Answer:
# 1. Address the Question: It must still fundamentally answer the original Question.
# 2. Incorporate Hallucinations: It must include 1 or 2 specific pieces of information that are explicitly NOT found in the provided Context.
# 3. Plausibility: The added information (the hallucinations) should sound plausible and relevant to the Question and the general topic, even though it lacks support in the Context.
# 4. Integration: Weave the hallucinated information naturally into the text. Do not simply append it awkwardly. It should blend smoothly with the information retained from the Original Grounded Answer.
# 5. Minimal Other Changes: Preserve the core information and structure of the Original Grounded Answer as much as possible, only augmenting it with the hallucinations.
# 6. Output: Provide only the text of the Modified Answer.

# Input Information:

# --- START CONTEXT ---
# {context}
# --- END CONTEXT ---

# --- START QUESTION ---
# {question}
# --- END QUESTION ---

# --- START ORIGINAL GROUNDED ANSWER ---
# {original_answer}
# --- END ORIGINAL GROUNDED ANSWER ---

# Instruction:
# Please generate the Modified Answer based on the requirements above.

# Modified Answer:
# """


def bosch_rm_synthetic_hall_llm(entry, fewshot_examples=None):
    preamble = """
    Objective:
    Generate a modified version of a given answer that includes subtle, natural-sounding synthetic hallucinations (information plausible but not directly supported by the provided context), mimicking the kinds of errors LLMs sometimes make organically.

    Goal Context:
    We are creating training data for a hallucination detection model. Previous attempts generated hallucinations that were too obvious, causing the model to overfit. The goal now is to generate *subtle* hallucinations that resemble real-world LLM errors, making the synthetic data more realistic and improving the detector's generalization.

    What Makes a Good Subtle Hallucination (for this task):
    - It often feels like a minor factual inaccuracy, a slight misinterpretation, or a plausible detail/reason/consequence that *could* be true but isn't mentioned.
    - It blends seamlessly with the surrounding text derived from the context.
    - It's not easily identifiable as completely fabricated or drastically contradicting the context.
    - It avoids being the main point of the answer; it's usually a supporting detail.
    """

    prompt = preamble

    if fewshot_examples is not None:
        prompt += """
        
        --- START FEW-SHOT EXAMPLES OF DESIRED OUTPUT ---

        """

        for fewshot in fewshot_examples:
            prompt += f"""Example:
            
            [QUESTION] {fewshot["Question"]}

            [CONTEXT] {fewshot["Context"]}

            [ANSWER WITH HALLUCINATION] {fewshot["response"]}

            """

        prompt += """
        --- END FEW-SHOT EXAMPLES ---

        """

    prompt += f"""Task:
    You are given a Question, a Context used to answer it, and an Original Grounded Answer derived solely from that Context. Your task is to rewrite the Original Grounded Answer to create a Modified Answer containing subtle hallucinations.

    Requirements for the Modified Answer:
    1.  Address the Question: It must still fundamentally answer the original Question.
    2.  Incorporate Subtle Hallucinations: Include 1 or 2 specific pieces of information that are plausible and relevant BUT NOT DIRECTLY SUPPORTED by the provided Context. Aim for the kind of subtlety shown in the examples.
    3.  Plausibility & Consistency: The added information should sound highly plausible within the domain, be consistent with the overall tone and information in the context, and not be easily flagged as fake. Avoid contradictions.
    4.  Natural Integration: Weave the hallucinated information smoothly into the text. Do not make it stand out awkwardly. It should blend seamlessly with the grounded information.
    5.  Minimal Other Changes: Preserve the core grounded information and structure of the Original Grounded Answer as much as possible, augmenting it *only* with the subtle hallucinations.
    6.  Output: Provide only the text of the Modified Answer.

    Input Information:

    --- START QUESTION ---
    {entry["Question"]}
    --- END QUESTION ---

    --- START CONTEXT ---
    {entry["Context"]}
    --- END CONTEXT ---

    --- START ORIGINAL GROUNDED ANSWER ---
    {entry["response"]}
    --- END ORIGINAL GROUNDED ANSWER ---

    Instruction:
    Please generate the Modified Answer based on the requirements and examples above.

    Modified Answer:
"""

    return prompt


def bosch_rm_synthetic_hall_structured(entry, data):
    """Given an entry and the rest of the data, select a random sentence of
    the response in the entry, a random different entry in the data and a
    random sentence in it, and swap the first by the second.
    TO DO: I am not particularly worried with seeds here.
    """

    # Break response into sentences and filter out small ones
    tok = nltk.sent_tokenize(entry["response"])
    tok_filt = [i for i in tok if len(i) > 5]

    # Choose a random different entry and do the same
    entry2 = data.shuffle()[0]
    tok2 = nltk.sent_tokenize(entry2["response"])
    tok_filt2 = [i for i in tok2 if len(i) > 5]

    # Randomly select sentences in both entries
    rand = random.choice(tok_filt)
    rand_idx = tok.index(rand)
    rand2 = random.choice(tok_filt2)
    rand_idx2 = tok2.index(rand2)

    # Switch sentence and join
    tok[rand_idx] = tok2[rand_idx2]
    new_response = " ".join(tok)

    # Update response and labels
    entry["response"] = new_response
    entry["class_hall"] = "Yes"
    entry["label"] = 0

    # Important: you need to retokenize these!

    return entry


def bosch_function_to_map_synthetic_halls_llm(
    entry, fewshot_examples, client, gemini_model, generation_config
):
    prompt_to_send = bosch_rm_synthetic_hall_llm(entry, fewshot_examples)

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt_to_send,
        config=generation_config,
    )

    return {
        "response": response.text,
        "class_hall": "Yes",
        "label": 0,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--tokenizer_model", type=str, default="google/gemma-2-2b-it"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--raw_repo_id", type=str)
    parser.add_argument("--processed_repo_id", type=str)
    parser.add_argument("--rm_processed_repo_id", type=str)
    parser.add_argument("--rm_validation_size", type=float, default=0.2)
    parser.add_argument(
        "--create_synthetic_hallus_llm",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--create_synthetic_hallus_struct",
        default=False,
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--writer_sft_processed_repo_id", type=str)
    parser.add_argument("--perl_raw_repo_id", type=str)
    parser.add_argument("--perl_processed_repo_id", type=str)
    parser.add_argument("--perl_train_size", type=int)
    parser.add_argument("--perl_validation_size", type=int)
    parser.add_argument("--augmented_repo_id", type=str)
    parser.add_argument("--gemini_api_key", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        padding_side="left",  # Pay attention to this argument!
    )

    if args.task == "halomi":
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

    elif args.task == "npov":
        # Reward Model
        npov_rm_data = load_dataset(
            "leobianco/npov",
            data_files={
                "train": "hc_rm5x_train.json",
                "validation": "hc_rm5x_validation.json",
                "test": "hc_rm5x_test.json",
            },
        )

        # Version with all data
        npov_autorater_data = concatenate_datasets(
            [
                npov_rm_data["train"],
                npov_rm_data["validation"],
                npov_rm_data["test"],
            ]
        )

        # Version with organic hallucinations only
        npov_autorater_data_organic_only = npov_autorater_data.filter(
            lambda x: x["has synthetic hallucination"] == "NO"
        )

        # Tokenize both versions
        npov_autorater_data = npov_process_data_for_rm(
            npov_autorater_data, tokenizer, max_seq_length=args.max_seq_length
        )
        npov_autorater_data_organic_only = npov_process_data_for_rm(
            npov_autorater_data_organic_only,
            tokenizer,
            max_seq_length=args.max_seq_length,
        )

        # Save both versions
        npov_autorater_data.push_to_hub(
            repo_id="npov_autorater_data_organic_and_synthetic",
            split="test",
        )
        npov_autorater_data_organic_only.push_to_hub(
            repo_id="npov_autorater_data_organic_only",
            split="test",
        )

        # Before filtering out the data for training on synthetic only,
        # create the data for evaluating the autorater

        # To train on synthetic hallucinations only and evaluate on organic,
        # drop the organic ones from the training set, and the synthetic ones
        # from the validation set. Do not mix splits, as this would mix topics.
        # Do not pass the test set into the validation one, because it will be
        # the test set for PERL later on and we agreed that we cannot test
        # both the RM and PERL on the same examples.

        npov_rm_data["train"] = npov_rm_data["train"].filter(
            lambda x: not (
                x["has hallucination"] == "YES"
                and x["has synthetic hallucination"] == "NO"
            )
        )

        npov_rm_data["validation"] = npov_rm_data["validation"].filter(
            lambda x: not (
                x["has hallucination"] == "YES"
                and x["has synthetic hallucination"] == "YES"
            )
        )

        for split in npov_rm_data.keys():
            npov_rm_data[split] = npov_process_data_for_rm(
                npov_rm_data[split],
                tokenizer,
                max_seq_length=args.max_seq_length,
            )

            npov_rm_data[split].push_to_hub(
                repo_id=args.rm_processed_repo_id,
                split=split,
            )

        # Writer SFT
        npov_sft_data = load_dataset(
            "leobianco/npov",
            data_files={
                "train": "writer_train.json",
                "validation": "writer_validation.json",
                "test": "writer_test.json",
            },
        )

        for split in npov_sft_data.keys():
            npov_sft_data[split] = npov_process_data_for_sft(
                npov_sft_data[split]
            )

            npov_sft_data[split].push_to_hub(
                repo_id=args.writer_sft_processed_repo_id,
                split=split,
            )

        # PERL
        npov_perl_data = npov_process_data_for_perl(
            npov_rm_data,
            npov_sft_data,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )

        for split in npov_perl_data.keys():
            npov_perl_data[split].push_to_hub(
                repo_id=args.perl_processed_repo_id,
                split=split,
            )

        # Data augmentation
        npov_augmented_data = npov_data_augmentation(npov_rm_data["test"])
        npov_augmented_data_dict = DatasetDict(
            {"test": Dataset.from_dict(npov_augmented_data)}
        )
        npov_augmented_data_dict["test"] = npov_augmented_data_dict["test"].map(
            npov_writer_prompt
        )
        npov_augmented_data_dict.push_to_hub(
            repo_id=args.augmented_repo_id,
        )

    elif args.task == "bosch":
        # GLOBAL DATA
        data = pd.read_csv(
            "/home/leo/Downloads/DelucionQA_data/cleaned/train.csv"
        )
        data_to_flip_train = pd.read_csv(
            "/home/leo/Downloads/DelucionQA_data/cleaned_leo/leo_flip_train.csv"
        )
        data_test = pd.read_csv(
            "/home/leo/Downloads/DelucionQA_data/cleaned/test.csv"
        )
        data_to_flip_test = pd.read_csv(
            "/home/leo/Downloads/DelucionQA_data/cleaned_leo/leo_flip_test.csv"
        )
        data_dev = pd.read_csv(
            "/home/leo/Downloads/DelucionQA_data/cleaned/dev.csv"
        )
        data_to_flip_dev = pd.read_csv(
            "/home/leo/Downloads/DelucionQA_data/cleaned_leo/leo_flip_dev.csv"
        )

        # Merge the two
        data = pd.concat([data, data_test, data_dev])
        data_to_flip = pd.concat(
            [data_to_flip_train, data_to_flip_test, data_to_flip_dev]
        )

        # To exclude bad data
        # data = data[~data["sample_id"].isin(data_to_exclude["sample_id"])]

        # To flip the label for the bad data
        data.loc[data["sample_id"].isin(data_to_flip["sample_id"]), "Label"] = (
            "Not Hallucinated"
        )

        # Filter to keep only context relating to the question
        data = data.loc[data["Answerable"] == True]
        data = data.drop(labels=["Answerable"], axis=1)

        # Create or rename columns
        data = data.rename(
            {"Label": "class_hall", "Answer": "response"}, axis=1
        )
        data["class_hall"] = data["class_hall"].apply(
            lambda x: "Yes" if x == "Hallucinated" else "No"
        )
        data["label"] = data["class_hall"].apply(
            lambda x: 1 if x == "No" else 0
        )
        data["prompt"] = (
            "<start_of_turn><user>\nYou are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information giver. Do not add to your answer any information other than those present in the manual excerpt.\n"
            + "User question:\n"
            + data["Question"]
            + "\nManual information:\n"
            + data["Context"]
            + "\nAnswer to user's question:<end_of_turn>\n<start_of_turn><model>\n"
        )

        dataset = Dataset.from_pandas(data, split="train")
        dataset.push_to_hub(args.processed_repo_id)

        # SFT
        sft_data = dataset.filter(lambda entry: entry["class_hall"] == "No")
        sft_data.push_to_hub(args.writer_sft_processed_repo_id)

        # Reward Model
        hallucinations_data = dataset.filter(
            lambda entry: entry["class_hall"] == "Yes"
        )
        non_hallucinated_data = dataset.filter(
            lambda entry: entry["class_hall"] == "No"
        )
        # We don't want to use all of the non-hallucinated data.
        # Some of it goes to PERL, etc
        rm_data_cap = 1000
        size_subset = rm_data_cap - hallucinations_data.num_rows
        subset_non_hallucinated_data = non_hallucinated_data.shuffle(
            seed=args.seed
        ).select(range(size_subset))

        # RM dataset with organic hallucinations
        rm_data = concatenate_datasets(
            [hallucinations_data, subset_non_hallucinated_data]
        )

        rm_data = bosch_process_data_for_rm(
            rm_data,
            tokenizer,
            seed=args.seed,
            max_seq_length=args.max_seq_length,
        )

        rm_data.push_to_hub(args.rm_processed_repo_id)

        #
        # DATA ORGANIZATION FOR SYNTHETIC HALLUCINATIONS
        #

        # You don't want exact pairing, because the model then overfits.
        # Instead, shuffle the subset, then take a new subset of it to
        # be the one out of which synthetic hallucinations will be created.
        # Do not use the base non-hallucinated samples to train the RM,
        # use only the hallucinated version and put the original in the
        # PERL test split.
        # Confusing, I know!

        # subset_2 are the non-hallucinated data that will become hallucinated
        size_subset_2 = int(size_subset / 4)
        subset_2_end_idx = size_subset_2  # because it starts from zero
        subset_2 = subset_non_hallucinated_data.select(range(size_subset_2))

        # subset_3 are the non-hallucinated samples that will also be used to
        # train the RM
        size_subset_3 = 2 * size_subset_2
        subset_3_end_idx = subset_2_end_idx + size_subset_3
        subset_3 = subset_non_hallucinated_data.select(
            range(subset_2_end_idx, subset_3_end_idx)
        )

        # Finally, subset_4 are the non-hallucinated samples that go to the
        # test split of the RM
        subset_4 = subset_non_hallucinated_data.select(
            range(subset_3_end_idx, size_subset)
        )

        # All of them are subsets of subset_non_hallucinated_data

        #
        # RM SYNTHETIC HALLUCINATIONS - LLM GENERATED
        #

        if args.create_synthetic_hallus_llm:
            # Fewshot examples of organic hallucinations
            n_fewshot_examples_synth_llm = 4
            fewshot_examples_synth_llm = hallucinations_data.select(
                range(n_fewshot_examples_synth_llm)
            )

            # API config
            client = genai.Client(api_key=args.gemini_api_key)
            gemini_model = "gemini-2.0-flash-001"
            generation_config = types.GenerateContentConfig(
                temperature=0.7,
                seed=args.seed,
            )

            print("Calling Gemini's API...")

            synthetic_hallucinations_llm = subset_2.map(
                bosch_function_to_map_synthetic_halls_llm,
                fn_kwargs=dict(
                    fewshot_examples=fewshot_examples_synth_llm,
                    client=client,
                    gemini_model=gemini_model,
                    generation_config=generation_config,
                ),
            )

            # Now we create the RM training data
            synthetic_hallucinations_llm_train_data = concatenate_datasets(
                [synthetic_hallucinations_llm, subset_3]
            )
            synthetic_hallucinations_llm_train_data = (
                synthetic_hallucinations_llm_train_data.shuffle(seed=args.seed)
            )

            # TO DO: put the original version of the first third of
            # subset_non_hallucinated_data in the PERL test set

            # You need to rewrite the prompts, re-tokenize... but don't re-split!
            # This will be the train split of the final Dataset.
            # The test set will be composed of organic hallucinations + some
            # non-hallucinated samples from the PERL training set.
            synthetic_hallucinations_llm_train_data = bosch_process_data_for_rm(
                synthetic_hallucinations_llm_train_data,
                tokenizer,
                seed=args.seed,
                split_data=False,
                max_seq_length=args.max_seq_length,
            )

            # Building the test split
            synthetic_hallucinations_llm_test_data = concatenate_datasets(
                [hallucinations_data, subset_4]
            ).shuffle(seed=args.seed)

            synthetic_hallucinations_llm_test_data = bosch_process_data_for_rm(
                synthetic_hallucinations_llm_test_data,
                tokenizer,
                seed=args.seed,
                split_data=False,
                max_seq_length=args.max_seq_length,
            )

            # Merge the two in the appropriate splits
            synthetic_hallucinations_llm_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_llm_train_data,
                    "test": synthetic_hallucinations_llm_test_data,
                }
            )

            synthetic_hallucinations_llm_data.push_to_hub(
                repo_id=args.rm_processed_repo_id + "_synthetic_llm"
            )

        #
        # RM SYNTHETIC HALLUCINATIONS - STRUCTURED
        #
        elif args.create_synthetic_hallus_struct:
            # Apply the map that switches sentences to subset_2
            synthetic_hallucinations_struct_train_data = subset_2.map(
                bosch_rm_synthetic_hall_structured, fn_kwargs=dict(data=dataset)
            )

            # Add some non-hallucinated examples and shuffle!
            synthetic_hallucinations_struct_train_data = concatenate_datasets(
                [synthetic_hallucinations_struct_train_data, subset_3]
            ).shuffle(seed=args.seed)

            # Reconstuct prompts and retokenize
            synthetic_hallucinations_struct_train_data = (
                bosch_process_data_for_rm(
                    synthetic_hallucinations_struct_train_data,
                    tokenizer,
                    seed=args.seed,
                    split_data=False,
                    max_seq_length=args.max_seq_length,
                )
            )

            # Create test split with organic hallucinations
            synthetic_hallucinations_struct_test_data = concatenate_datasets(
                [hallucinations_data, subset_4]
            ).shuffle(seed=args.seed)

            synthetic_hallucinations_struct_test_data = (
                bosch_process_data_for_rm(
                    synthetic_hallucinations_struct_test_data,
                    tokenizer,
                    seed=args.seed,
                    split_data=False,
                    max_seq_length=args.max_seq_length,
                )
            )

            # Merge the two in the appropriate splits
            synthetic_hallucinations_struct_data = DatasetDict(
                {
                    "train": synthetic_hallucinations_struct_train_data,
                    "test": synthetic_hallucinations_struct_test_data,
                }
            )

            synthetic_hallucinations_struct_data.push_to_hub(
                repo_id=args.rm_processed_repo_id + "_synthetic_struct"
            )

        # PERL
        perl_data = bosch_process_data_for_perl(
            non_hallucinated_data.select(
                range(
                    (rm_data_cap - hallucinations_data.num_rows),
                    non_hallucinated_data.num_rows,
                )
            ),
            tokenizer,
            seed=args.seed,
            max_seq_length=args.max_seq_length,
            validation_size=args.perl_validation_size,
        )

        for split in perl_data.keys():
            perl_data[split].push_to_hub(
                repo_id=args.perl_processed_repo_id,
                split=split,
            )

    elif args.task == "ragtruth":
        # Load data
        data_sources = pd.read_json(
            "/home/leo/Downloads/RAGTruth/dataset/source_info.jsonl", lines=True
        )
        data_responses = pd.read_json(
            "/home/leo/Downloads/RAGTruth/dataset/response.jsonl", lines=True
        )

        data2txt_sources = data_sources[data_sources["task_type"] == "Data2txt"]
        data2txt_responses = data_responses[
            data_responses["source_id"].isin(data2txt_sources["source_id"])
        ]
        data2txt_unified = pd.merge(
            data2txt_sources, data2txt_responses, on="source_id", how="left"
        )
        data2txt_unified = data2txt_unified.drop(
            ["source_info", "task_type", "source", "id"], axis=1
        )
        # Numerical hallucination labels
        data2txt_unified["label"] = data2txt_unified.apply(
            lambda entry: 1 if len(entry["labels"]) == 0 else 0, axis=1
        )
        # Textual hallucination labels
        data2txt_unified["class_hall"] = data2txt_unified.apply(
            lambda entry: "No" if len(entry["labels"]) == 0 else "Yes", axis=1
        )

        data2txt_dataset = Dataset.from_pandas(data2txt_unified)
        data2txt_dataset_splits = data2txt_dataset.train_test_split(
            test_size=0.1, shuffle=False
        )

        # On test split, get only unique prompts
        # This is done in an UGLY way, out of hurry
        test_pandas = pd.DataFrame(data2txt_dataset_splits["test"])
        test_pandas = test_pandas.drop_duplicates(
            subset=["source_id"], keep="first"
        )
        del data2txt_dataset_splits["test"]
        test_hf = Dataset.from_pandas(test_pandas)
        test_hf = test_hf.remove_columns(["__index_level_0__"])
        data2txt_dataset_splits["test"] = test_hf

        for split in data2txt_dataset_splits.keys():
            data2txt_dataset_splits[split].push_to_hub(
                args.processed_repo_id, split=split
            )


if __name__ == "__main__":
    main()
