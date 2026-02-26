from copy import deepcopy
from itertools import combinations
from typing import Any, Callable, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types

from src.task_processors.base_task_processor import BaseTaskProcessor


class NPOVTaskProcessor(BaseTaskProcessor):
    def _load_data(self) -> DatasetDict:
        data = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "hc_rm5x_train.json",
                "validation": "hc_rm5x_validation.json",
                "test": "hc_rm5x_test.json",
            },
        )

        return data

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
        """Process NPOV data for reward model training."""

        processed = {}

        for split in data.keys():
            split_data = data[split]
            if "has hallucination" in split_data.column_names:
                split_data = split_data.rename_column(
                    "has hallucination", "class_hall"
                )
            if "has coverage issue" in split_data.column_names:
                split_data = split_data.rename_column(
                    "has coverage issue", "class_omit"
                )
            split_data = split_data.select_columns(
                [
                    "topic",
                    "user_query",
                    "npov_response",
                    "perspective_1",
                    "perspective_1_name",
                    "perspective_2",
                    "perspective_2_name",
                    "class_hall",
                    "has synthetic hallucination",
                    "class_omit",
                    "has synthetic coverage issue",
                ]
            )
            split_data = split_data.map(self._change_hallucination_labels)
            split_data = split_data.map(self._change_omission_labels)
            split_data = split_data.map(self._hallucination_labels_to_numerical)
            split_data = split_data.map(self._omission_labels_to_numerical)
            split_data = split_data.map(self._rm_prompt)
            split_data = split_data.rename_column("class_hall_num", "label")
            processed[split] = split_data

        processed = DatasetDict(processed)

        return processed

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        """Process NPOV data for supervised fine-tuning."""

        sft_data = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "writer_train.json",
                "validation": "writer_validation.json",
                "test": "writer_test.json",
            },
        )

        for split in sft_data.keys():
            sft_data[split] = self._process_data_for_sft(sft_data[split])

        # Merge validation and test splits before saving.
        sft_data = DatasetDict(
            {
                "train": sft_data["train"],
                "test": concatenate_datasets(
                    [sft_data["validation"], sft_data["test"]]
                ),
            }
        )

        return sft_data

    def _make_organic_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        organic_hallucination_data = deepcopy(data)

        # Filter out synthetic hallucinations from train set
        organic_hallucination_data["train"] = organic_hallucination_data[
            "train"
        ].filter(
            lambda x: (
                not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "Yes"
                )
            )
        )

        # Filter out synthetic hallucinations from validation set
        organic_hallucination_data["validation"] = organic_hallucination_data[
            "validation"
        ].filter(
            lambda x: (
                not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "Yes"
                )
            )
        )

        # Put in splits and save
        dataset = DatasetDict(
            {
                "train": organic_hallucination_data["train"],
                "test": organic_hallucination_data["validation"],
            }
        )

        return dataset

    def _make_structured_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        """
        Processes and filters hallucination data to create structured training and test splits.
        This method modifies the input data dictionary by:
          - Filtering the training set to include only synthetic hallucinations (removing organic hallucinations).
          - Filtering the validation set to exclude synthetic hallucinations (keeping only organic hallucinations).
          - Assigning the filtered training set to the "train" split and the filtered validation set to the "test" split
            in the returned DatasetDict.
        Args:
            data (dict): A dictionary containing dataset splits, typically with "train" and "validation" keys,
                where each value is a dataset supporting the `.filter()` method.
        Returns:
            DatasetDict: A HuggingFace DatasetDict with "train" and "test" splits containing the filtered data.
        """
        synthetic_hallucination_data = deepcopy(data)

        synthetic_hallucination_data["train"] = synthetic_hallucination_data[
            "train"
        ].filter(
            lambda x: (
                not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "No"
                )
            )
        )

        # Filter out synthetic hallucinations from validation set
        synthetic_hallucination_data["validation"] = (
            synthetic_hallucination_data["validation"].filter(
                lambda x: (
                    not (
                        x["class_hall"] == "Yes"
                        and x["has synthetic hallucination"] == "Yes"
                    )
                )
            )
        )

        # Put in splits and save
        dataset = DatasetDict(
            {
                "train": synthetic_hallucination_data["train"],
                "test": synthetic_hallucination_data["validation"],
            }
        )

        return dataset

    def _make_llm_hallucinations_data(
        self, data: DatasetDict, organic_hallucinations_data: DatasetDict
    ) -> DatasetDict:
        """Generate synthetic hallucinations using LLM and return DatasetDict for reward model training."""

        args = self.args
        data_processed = deepcopy(data)

        # Fewshot examples of organic hallucinations to help LLM generation
        n_fewshot_examples_synth_llm = args.synth_llm_num_fewshot
        fewshot_examples_synth_llm = (
            organic_hallucinations_data["train"]
            .filter(lambda x: x["class_hall"] == "Yes")
            .shuffle(seed=args.seed)
            .select(range(n_fewshot_examples_synth_llm))
        )

        # Filter out synthetic hallucinations from validation set
        data_processed["validation"] = data_processed["validation"].filter(
            lambda x: (
                not (
                    x["class_hall"] == "Yes"
                    and x["has synthetic hallucination"] == "Yes"
                )
            )
        )

        # Get non-hallucinations
        train_non = (
            data_processed["train"]
            .filter(lambda x: x["class_hall"] == "No")
            .shuffle(seed=args.seed)
        )

        # We use some non-hallus. to generate synthetic hallus.
        train_synth_llm = train_non.shuffle(seed=args.seed).select(
            range(args.num_synth_hallus)
        )
        train_non_rest = train_non.filter(lambda x: x not in train_synth_llm)

        # API config
        client = genai.Client(api_key=args.gemini_api_key)
        gemini_model = "gemini-2.0-flash-001"
        generation_config = (
            types.GenerateContentConfig(
                temperature=args.synth_llm_temperature,
                seed=args.seed,
            )
            if types is not None
            else None
        )

        print("Calling Gemini's API...")

        synthetic_hallucinations_llm = train_synth_llm.map(
            self._function_to_map_synthetic_halls_llm,
            fn_kwargs=dict(
                fewshot_examples=fewshot_examples_synth_llm,
                client=client,
                gemini_model=gemini_model,
                generation_config=generation_config,
            ),
        )

        # Now we create the RM training data
        synthetic_hallucinations_llm_train_data = concatenate_datasets(
            [synthetic_hallucinations_llm, train_non_rest]
        ).shuffle(seed=args.seed)

        # Merge the two in the appropriate splits
        synthetic_hallucinations_llm_data = DatasetDict(
            {
                "train": synthetic_hallucinations_llm_train_data,
                "test": data_processed["validation"],
            }
        )

        # Since the response changed, you need to rewrite the prompt
        synthetic_hallucinations_llm_data = self._preprocess_data(
            synthetic_hallucinations_llm_data
        )

        return synthetic_hallucinations_llm_data

    def _make_perl_data(
        self,
        data: DatasetDict,
        sft_data: DatasetDict,
        seed: int = 12345,
    ) -> DatasetDict:
        """Prepare NPOV data for PERL by creating train and test splits and formatting prompts.

        Args:
            npov_rm_data (datasets.DatasetDict): Reward model data.
            npov_sft_data (datasets.DatasetDict): SFT data.
            seed (int, optional): Random seed for shuffling. Defaults to 12345.

        Returns:
            datasets.DatasetDict: DatasetDict with 'train' and 'test' splits for PERL.
        """

        train_data = data["validation"]
        test_data = sft_data["test"]

        train_data = train_data.shuffle(seed=seed)
        test_data = test_data.shuffle(seed=seed)

        train_data = train_data.map(self._writer_prompt)
        test_data = test_data.map(self._writer_prompt)

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
        train_data = train_data.rename_column("npov_response", "completion")

        perl_data = DatasetDict(
            {
                "train": train_data,
                "test": test_data,
            }
        )

        return perl_data

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        concat_data = concatenate_datasets(
            [data["train"], data["validation"], data["test"]]
        )
        concat_data = concat_data.filter(
            lambda entry: entry["has synthetic hallucination"] == "No"
        )
        return concat_data

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        """Create hyperparameter and capped final test sets from the test split."""

        args = self.args

        # Augment the test split
        augmented_data = self._data_augmentation(data["test"])
        augmented_dataset = Dataset.from_dict(augmented_data)
        augmented_dataset = augmented_dataset.map(self._writer_prompt)

        # Create capped final test set (10k samples)
        # TODO: this can be enhanced to keep topics balanced.
        capped_final_test_set = augmented_dataset.shuffle(
            seed=args.seed
        ).select(range(10000))

        evaluation_data = DatasetDict(
            {
                "test": capped_final_test_set,
            }
        )

        return evaluation_data

    @staticmethod
    def _change_hallucination_labels(entry: dict) -> dict:
        """Normalize hallucination label values for NPOV entries.

        Args:
            entry (dict): Entry with possible label variants for hallucination fields.

        Returns:
            dict: Entry with standardized 'class_hall' and 'has synthetic hallucination' fields ('Yes'/'No').
        """

        class_hallucination_to_label = {
            "NO": "No",
            "No": "No",
            "YES": "Yes",
            "Yes": "Yes",
        }

        entry["class_hall"] = class_hallucination_to_label[entry["class_hall"]]

        entry["has synthetic hallucination"] = class_hallucination_to_label[
            entry["has synthetic hallucination"]
        ]

        return entry

    @staticmethod
    def _change_omission_labels(entry: dict) -> dict:
        """Normalize omission label values for NPOV entries.

        Args:
            entry (dict): Entry with possible label variants for omission fields.

        Returns:
            dict: Entry with standardized 'class_omit' and 'has synthetic coverage issue' fields ('Yes'/'No').
        """

        class_omission_to_label = {
            "NO": "No",
            "No": "No",
            "YES": "Yes",
            "Yes": "Yes",
        }

        entry["class_omit"] = class_omission_to_label[entry["class_omit"]]

        entry["has synthetic coverage issue"] = class_omission_to_label[
            entry["has synthetic coverage issue"]
        ]

        return entry

    @staticmethod
    def _hallucination_labels_to_numerical(entry: dict) -> dict:
        """Convert hallucination labels to numerical values for NPOV entries.

        Args:
            entry (dict): A dictionary containing the key 'class_hall' with values 'Yes' or 'No'.

        Returns:
            dict: The entry with an added 'class_hall_num' key (0 for 'Yes', 1 for 'No').
        """

        entry["class_hall_num"] = 0 if entry["class_hall"] == "Yes" else 1

        return entry

    @staticmethod
    def _omission_labels_to_numerical(entry: dict) -> dict:
        """Convert omission labels to numerical values for NPOV entries.

        Args:
            entry (dict): A dictionary containing the key 'class_omit' with values 'Yes' or 'No'.

        Returns:
            dict: The entry with an added 'class_omit_num' key (0 for 'Yes', 1 for 'No').
        """

        entry["class_omit_num"] = 0 if entry["class_omit"] == "Yes" else 1

        return entry

    @classmethod
    def get_evaluator_prompt(
        cls,
    ) -> Callable[[dict, Optional[Dataset], bool], dict]:
        """Return a function that builds evaluator prompts for NPOV entries.

        The returned function has signature func(entry, fewshot_examples=None, use_true_label=False).
        """

        def npov_evaluator_prompt(
            entry: dict,
            fewshot_examples: Optional[Dataset] = None,
            use_true_label: bool = False,
        ) -> dict:
            preamble = "<start_of_turn>user\nYou are an expert linguist identifying when the answer to a user's query contains arguments that were NOT provided.<end_of_turn>\n"

            template = "<start_of_turn>user\nUser query: {user_query}\n{perspective_1_name} arguments provided: {perspective_1}\n{perspective_2_name} arguments provided: {perspective_2}\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:{npov_response}\nExpert linguist review: does the answer contain additional arguments that were NOT provided? (Yes/No):<end_of_turn>\n<start_of_turn>model\n{ans}"

            response = (
                entry["npov_response"]
                if use_true_label
                else entry["completion"]
            )

            formatted_prompt = template.format(
                user_query=entry["user_query"],
                perspective_1_name=entry["perspective_1_name"],
                perspective_1=entry["perspective_1"],
                perspective_2_name=entry["perspective_2_name"],
                perspective_2=entry["perspective_2"],
                npov_response=response,
                ans="",
            )

            prompt = preamble

            if fewshot_examples is not None:
                for fewshot_example in fewshot_examples:
                    fewshot_prompt = template.format(
                        user_query=fewshot_example["user_query"],
                        perspective_1_name=fewshot_example[
                            "perspective_1_name"
                        ],
                        perspective_1=fewshot_example["perspective_1"],
                        perspective_2_name=fewshot_example[
                            "perspective_2_name"
                        ],
                        perspective_2=fewshot_example["perspective_2"],
                        npov_response=fewshot_example["npov_response"],
                        ans=fewshot_example["class_hall"],
                    )
                    prompt += fewshot_prompt + "<end_of_turn>\n"
                prompt += formatted_prompt
            else:
                prompt += formatted_prompt

            entry["evaluator_prompt"] = prompt
            return entry

        return npov_evaluator_prompt

    @staticmethod
    def _rm_prompt(entry: dict) -> dict:
        """Format a reward model prompt for an NPOV entry.

        Args:
            entry (dict): Entry with user query, perspectives, and NPOV response.

        Returns:
            dict: Entry with a new 'prompt' field containing the formatted prompt.
        """

        template = (
            "You will be given an user's question, along with arguments for "
            "and against it. Your task is to answer the user's question in "
            "natural language using the arguments given. Use all the arguments "
            "given, and do not add to your answer any argument other than "
            "those provided.\n"
            "User query: {user_query}\n"
            "{perspective_1_name} arguments provided: {perspective_1}\n"
            "{perspective_2_name} arguments provided: {perspective_2}\n"
            "Answer to user query, rewriting provided"
            " arguments in natural language:\n"
            "{npov_response}"
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

    @staticmethod
    def _writer_prompt(
        entry: dict,
        SFT: bool = False,  # TODO: remove this arg, use prompt-completion
        fewshot_examples: Optional[Dataset] = None,
    ) -> dict:
        """Format a prompt for the NPOV writer task, optionally with few-shot examples.

        Args:
            entry (dict): Entry with user query, perspectives, and NPOV response.
            SFT (bool, optional): If True, include the NPOV response in the prompt. Defaults to False. DEPRECATED
            fewshot_examples (datasets.Dataset, optional): Few-shot examples to prepend. Defaults to None.

        Returns:
            dict: Entry with a new 'prompt' field.
        """

        preamble = """You will be given an user's question, along with arguments for and against it. Your task is to answer the user's question in natural language using the arguments given. Use all the arguments given, and do not add to your answer any argument other than those provided.\n"""

        template = (
            "User query: {user_query}\n"
            "{perspective_1_name} arguments provided: {perspective_1}\n"
            "{perspective_2_name} arguments provided: {perspective_2}\n"
            "Answer to user query, rewriting provided"
            " arguments in natural language:\n"
            "{npov_response}"
        )

        npov_response = entry["npov_response"] if SFT else ""

        formatted_prompt = template.format(
            user_query=entry["user_query"],
            perspective_1_name=entry["perspective_1_name"],
            perspective_1=entry["perspective_1"],
            perspective_2_name=entry["perspective_2_name"],
            perspective_2=entry["perspective_2"],
            npov_response=npov_response,
        )

        entry["prompt"] = preamble + formatted_prompt

        return entry

    # Deprecated, TODO: remove this later
    @classmethod
    def get_formatting_prompts_and_response_template(
        cls,
        eos_token: str,
        fewshot_examples: Optional[Dataset] = None,
        model_repo_id: Optional[str] = None,
    ) -> Tuple[Callable[[Any], list[str]], str]:
        """Return a formatting function and response template for SFT training.

        Args:
            eos_token (str): Tokenizer eos token.
            fewshot_examples (datasets.Dataset | None): Few-shot examples to include.
            model_repo_id (str | None): Model repo id to choose model-specific template.

        Returns:
            (callable, str): formatting function and response template string.
        """

        # Choose response template based on model company.
        # This is because of the different tokenizers leading to different
        # strings that can be found.
        response_template = None
        if model_repo_id is not None:
            model_company = model_repo_id.split("/")[0]
            if model_company == "google":
                response_template = "\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:\n"
            elif model_company == "mistralai":
                response_template = "point-of-view answer to user query, rewriting provided arguments in natural language:\n"
            elif model_company == "Qwen":
                response_template = "\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:\n"
            else:
                response_template = None

        if response_template is None:
            raise Exception("Response template not specified for model!")

        def formatting_prompts_func(entry: dict) -> list[str]:
            template = (
                "User query: {user_query}\n"
                "{perspective_1_name} arguments provided: {perspective_1}\n"
                "{perspective_2_name} arguments provided: {perspective_2}\n"
                "Neutral point-of-view answer to user query, rewriting provided"
                " arguments in natural language:\n"
                "{npov_response}{eos_token}"
            )

            template_fewshot = (
                "User query: {user_query}\n"
                "{perspective_1_name} arguments provided: {perspective_1}\n"
                "{perspective_2_name} arguments provided: {perspective_2}\n"
                "Example neutral point-of-view answer to user query, rewriting provided"
                " arguments in natural language:\n"
                "{npov_response}{eos_token}"
            )

            prompt = ""

            output_texts = []

            # Build prompt with few-shot examples, if any
            if fewshot_examples is not None:
                preamble = (
                    "Your task is to answer an user's query"
                    " by rewriting the provided arguments in natural language. Do not "
                    "generate arguments other than those provided. "
                    f"We provide {fewshot_examples.num_rows} example(s) of what is "
                    "expected, then it is your turn.\n"
                )

                prompt += preamble

                for fewshot_example in fewshot_examples:
                    fewshot_prompt = template_fewshot.format(
                        user_query=fewshot_example["user_query"],
                        perspective_1_name=fewshot_example[
                            "perspective_1_name"
                        ],
                        perspective_1=fewshot_example["perspective_1"],
                        perspective_2_name=fewshot_example[
                            "perspective_2_name"
                        ],
                        perspective_2=fewshot_example["perspective_2"],
                        npov_response=fewshot_example["npov_response"],
                        eos_token=eos_token,
                    )
                    prompt += fewshot_prompt + "\n"

            for i in range(len(entry["user_query"])):
                formatted_prompt = template.format(
                    user_query=entry["user_query"][i],
                    perspective_1_name=entry["perspective_1_name"][i],
                    perspective_1=entry["perspective_1"][i],
                    perspective_2_name=entry["perspective_2_name"][i],
                    perspective_2=entry["perspective_2"][i],
                    npov_response=entry["npov_response"][i],
                    eos_token=eos_token,
                )

                output_texts.append(prompt + formatted_prompt)

            return output_texts

        return formatting_prompts_func, response_template

    def _process_data_for_sft(self, split_data: Dataset) -> Dataset:
        """Preprocess NPOV SFT dataset by selecting and renaming relevant columns.

        Args:
            npov_data (datasets.Dataset): The NPOV SFT dataset split.

        Returns:
            datasets.Dataset: Processed dataset for SFT training.
        """

        # Select relevant subset of columns
        split_data = split_data.select_columns(
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

        split_data = split_data.rename_column(
            "npov_response_combined", "npov_response"
        )

        split_data = split_data.map(self._writer_prompt)

        split_data = split_data.rename_column("npov_response", "completion")

        return split_data

    def _function_to_map_synthetic_halls_llm(
        self,
        entry: dict,
        fewshot_examples: Dataset,
        client: genai.Client,
        gemini_model: str,
        generation_config: types.GenerateContentConfig,
    ) -> dict:
        """Call Gemini LLM to generate a synthetic hallucinated NPOV response.

        Args:
            entry (dict): Entry to modify.
            fewshot_examples (datasets.Dataset): Few-shot examples for the LLM.
            client: Gemini API client.
            gemini_model (str): Model name.
            generation_config: Generation configuration for the LLM.

        Returns:
            dict: Entry with hallucinated response and updated labels.
        """

        prompt_to_send = self._rm_synthetic_hall_llm(entry, fewshot_examples)

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt_to_send,
            config=generation_config,
        )

        return {
            "npov_response": response.text,
            "class_hall": "Yes",
            "label": 0,
            "has synthetic hallucination": "Yes",
        }

    # Deprecated, TODO: remove this later
    @staticmethod
    def _rm_synthetic_hall_llm(
        entry: dict, fewshot_examples: Optional[Dataset] = None
    ) -> str:
        """Generate a prompt for LLM-based synthetic hallucination creation for NPOV entries.

        Args:
            entry (dict): Entry with user query, perspectives, and NPOV response.
            fewshot_examples (datasets.Dataset, optional): Few-shot examples to include. Defaults to None.

        Returns:
            str: Prompt for LLM to generate a hallucinated answer.
        """

        preamble = (
            "SYSTEM:\n"
            "You are a writer tasked with introducing subtle, realistic hallucinations into neutral answers. The original answer is a balanced and neutral synthesis of two opposing perspectives, presenting only the arguments provided in the context. Your goal is to modify this answer by adding some new piece of information or argument that is NOT present in the original answer or context. This addition should be small, plausible, and smoothly integrated, resembling the kind of factual error that language models often make when generating text. It can appear as added information on the existing arguments in the context, or as a new argument on its own.\n"
        )

        prompt = preamble

        if fewshot_examples is not None:
            prompt += (
                "\n--- FEW-SHOT EXAMPLES OF ANSWERS WITH HALLUCINATIONS ---\n\n"
            )

            for fewshot in fewshot_examples:
                prompt += (
                    "EXAMPLE:\n"
                    "Question:\n"
                    f"{fewshot['user_query']}\n\n"
                    "Context:\n"
                    f"List of {fewshot['perspective_1_name']} arguments:\n"
                    f"{fewshot['perspective_1']}\n\n"
                    f"List of {fewshot['perspective_2_name']} arguments:\n"
                    f"{fewshot['perspective_2']}\n\n"
                    "Answer with hallucination:\n"
                    f"{fewshot['npov_response']}\n\n"
                )

            prompt += "\n--- END FEW-SHOT EXAMPLES ---\n\n"

        prompt += (
            "\nTASK:\n"
            "Question:\n"
            f"{entry['user_query']}\n\n"
            "Context:\n"
            f"List of {entry['perspective_1_name']} arguments:\n"
            f"{entry['perspective_1']}\n\n"
            f"List of {entry['perspective_2_name']} arguments:\n"
            f"{entry['perspective_2']}\n\n"
            "Neutral Point-of-View Answer, without hallucinations:\n"
            f"{entry['npov_response']}\n\n"
            "INSTRUCTIONS:\n"
            "Create a version of the Neutral Point-of-View answer containing hallucinations by either introducing subtle, hard-to-detect inaccuracies or misrepresentations to the arguments that were present in the context, OR by adding one new argument not present in the context.\n"
            "The hallucinations must respect the following:\n"
            "1. Keep most (90%+) of the content faithful to the original arguments. Adding too much text makes hallucinations easy to detect.\n"
            "2. Be consistent with the overall argument flow and keep the tone of the original answer.\n"
            "3. Be presented as factual.\n\n"
            "OUTPUT:\n"
            'A single, self-contained "Modified Answer" containing the subtle hallucinations (new information not in the context).\n\n'
            "Modified Answer with Synthetic Hallucination:\n"
        )

        return prompt

    @staticmethod
    def _data_augmentation(data: Dataset) -> dict:
        """Augment NPOV dataset by adding new perspectives and generating argument combinations.

        Args:
            data (datasets.Dataset): The NPOV dataset split to augment.

        Returns:
            dict: Dictionary with augmented data fields for new argument combinations.
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

        def _extract_perspective_arguments(
            topic: str, perspective_col: str, perspective_name: str
        ) -> set[str]:
            """
            Extract unique arguments for a given topic and perspective (pro or con).
            """

            topic_data = data.filter(lambda x: x["topic"] == topic)[
                perspective_col
            ]

            arguments = {
                f"{perspective_name}: {arg.strip()}"
                for text in topic_data
                for arg in text.split(f"{perspective_name}:")
                if arg.strip()
            }

            return arguments

        # Load new perspectives from local CSV
        new_perspectives = pd.read_csv(
            "src/task_processors/npov_new_perspectives.csv"
        )

        for topic in topics:
            user_query = data.filter(lambda x: x["topic"] == topic)[
                "user_query"
            ][0]

            p1_args = _extract_perspective_arguments(
                topic, "perspective_1", p1_name
            )
            p2_args = _extract_perspective_arguments(
                topic, "perspective_2", p2_name
            )

            # Filter new perspectives for the current topic
            new_perspectives_for_topic = new_perspectives[
                new_perspectives["topic"] == topic
            ]

            # Extract new pro and con arguments
            new_p1_args = set(
                new_perspectives_for_topic[
                    new_perspectives_for_topic["perspective"] == "pro"
                ]["argument"]
            )
            new_p2_args = set(
                new_perspectives_for_topic[
                    new_perspectives_for_topic["perspective"] == "con"
                ]["argument"]
            )

            # Add new arguments to the existing ones
            p1_args.update(new_p1_args)
            p2_args.update(new_p2_args)

            # Create combinations of perspectives
            # Single arguments
            p1_singles = list(p1_args)
            p2_singles = list(p2_args)

            # Pairs
            p1_pairs = list(combinations(p1_args, 2))
            p2_pairs = list(combinations(p2_args, 2))

            # Triplets
            p1_triplets = list(combinations(p1_args, 3))
            p2_triplets = list(combinations(p2_args, 3))

            # Process single arguments
            for p1_single in p1_singles:
                for p2_single in p2_singles:
                    all_p1_arguments.append(p1_single)
                    all_p2_arguments.append(p2_single)
                    all_p1_names.append(p1_name)
                    all_p2_names.append(p2_name)
                    all_topics.append(topic)
                    all_user_queries.append(user_query)

            # Process pairs
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

            # Process triplets
            for p1_triplet in p1_triplets:
                for p2_triplet in p2_triplets:
                    p1_combined = " ".join(p1_triplet)
                    p2_combined = " ".join(p2_triplet)
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
