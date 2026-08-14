import random
from typing import Any, Callable, Optional, Tuple

import evaluate
import nltk
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from google import genai
from google.genai import types

from src.task_processors.base_task_processor import BaseTaskProcessor

# Ensure punkt_tab tokenizer is available
nltk.download("punkt_tab", quiet=True)


class BoschTaskProcessor(BaseTaskProcessor):
    def _load_data(self) -> DatasetDict:
        data = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "train.csv",
                "validation": "val.csv",
                "test": "test.csv",
            },
        )

        return data

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
        # Filter unanswerable samples.
        data = data.filter(lambda entry: entry.get("Answerable"))
        data = data.remove_columns("Answerable")

        # Rename, create, and process the necessary columns.
        data = data.rename_columns(
            {"Label": "class_hall", "Answer": "response"}
        )
        data = data.map(
            lambda entry: {
                "class_hall": "Yes"
                if entry["class_hall"] == "Hallucinated"
                else "No"
            }
        )
        data = data.map(
            lambda entry: {"label": 1 if entry["class_hall"] == "No" else 0}
        )
        data = data.map(
            lambda entry: {
                "prompt": (
                    "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information giver. Do not add to your answer any information other than those present in the manual excerpt.\n"
                    + "User question:\n"
                    + entry["Question"]
                    + "\nManual information:\n"
                    + entry["Context"]
                    + "\nAnswer to user's question:\n"
                )
            }
        )

        # Flip labels of mis-annotated samples.
        data_to_flip = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "flip_train.csv",
                "validation": "flip_val.csv",
                "test": "flip_test.csv",
            },
        )

        data = self._flip_entries(data, data_to_flip)

        return data

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        """For the Bosch task, we SFT on the non-hallucinated samples of the validation split."""

        sft_data = data["test"].filter(
            lambda entry: entry["class_hall"] == "No"
        )

        # prompt-completion format, for training on completion only
        sft_data = sft_data.rename_column("response", "completion")

        sft_data = sft_data.train_test_split(test_size=0.2)

        return sft_data

    def _make_organic_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        organic_hallucinations_data = DatasetDict(
            {"train": data["test"], "test": data["validation"]}
        )

        organic_hallucinations_data = organic_hallucinations_data.map(
            self._rm_prompt
        )

        return organic_hallucinations_data

    def _make_structured_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        """Create structured synthetic hallucinations data for reward model training."""

        args = self.args
        # Use validation split for both non-hallucinated and hallucinated (TO BE CORRECTED).
        to_become_hallus, non_hallucinated_data_rest = (
            self._split_nonhallucinated_for_synthetic(
                data["validation"], args.num_synth_hallus, args.seed
            )
        )
        # Build RM prompts for the non-hallucinated data.
        non_hallucinated_data_rest = non_hallucinated_data_rest.map(
            self._rm_prompt
        )
        # Apply the map that creates structured hallucinations.
        rouge_metric = evaluate.load("rouge")

        synthetic_hallucinations_struct = to_become_hallus.map(
            self._synthetic_hall_structured,
            fn_kwargs=dict(rouge_metric=rouge_metric),
        )

        # Write the RM prompts with the modified entries.
        synthetic_hallucinations_struct = synthetic_hallucinations_struct.map(
            self._rm_prompt
        )
        # Add the non-hallucinated examples and shuffle.
        synthetic_hallucinations_struct_train_data = concatenate_datasets(
            [synthetic_hallucinations_struct, non_hallucinated_data_rest]
        ).shuffle(seed=args.seed)
        # Create dataset with same test split as organic.
        # TODO: take the change of splits into account!
        organic = self._make_organic_hallucinations_data(data)
        organic_test_split = organic["test"]
        # Add placeholder columns so that in the train split you can store
        # information about what sentence in context was erased
        organic_test_split = organic_test_split.add_column(
            "erased_context", [""] * len(organic_test_split)
        )
        organic_test_split = organic_test_split.add_column(
            "rouge1_score", [0.0] * len(organic_test_split)
        )

        synthetic_hallucinations_struct_data = DatasetDict(
            {
                "train": synthetic_hallucinations_struct_train_data,
                "test": organic_test_split,
            }
        )

        return synthetic_hallucinations_struct_data

    def _make_llm_hallucinations_data(
        self, data: DatasetDict, organic_hallucinations_data: DatasetDict
    ) -> DatasetDict:
        """Generate synthetic hallucinations using LLM and return DatasetDict for reward model training."""

        args = self.args
        # Use validation split for both non-hallucinated and hallucinated
        to_become_hallus, non_hallucinated_data_rest = (
            self._split_nonhallucinated_for_synthetic(
                data["validation"], args.num_synth_hallus, args.seed
            )
        )
        hallucinated_data = data["validation"].filter(
            lambda entry: entry["class_hall"] == "Yes"
        )
        # Build RM prompts for the non-hallucinated data
        non_hallucinated_data_rest = non_hallucinated_data_rest.map(
            self._rm_prompt
        )
        # Fewshot examples of organic hallucinations to help LLM
        n_fewshot_examples_synth_llm = args.synth_llm_num_fewshot
        fewshot_examples_synth_llm = hallucinated_data.shuffle(
            seed=args.seed
        ).select(range(n_fewshot_examples_synth_llm))

        client = genai.Client(api_key=args.gemini_api_key)
        gemini_model = "gemini-3.5-flash"
        generation_config = types.GenerateContentConfig(
            temperature=args.synth_llm_temperature,
            seed=args.seed,
        )
        # Call Gemini's API for synthetic hallucinations
        synthetic_hallucinations_llm = to_become_hallus.map(
            self._function_to_map_synthetic_halls_llm,
            fn_kwargs=dict(
                fewshot_examples=fewshot_examples_synth_llm,
                client=client,
                gemini_model=gemini_model,
                generation_config=generation_config,
            ),
        )
        # Write the RM prompts
        synthetic_hallucinations_llm = synthetic_hallucinations_llm.map(
            self._rm_prompt
        )
        # Now we create the RM training data by joining and shuffling
        synthetic_hallucinations_llm_train_data = concatenate_datasets(
            [synthetic_hallucinations_llm, non_hallucinated_data_rest]
        ).shuffle(seed=args.seed)
        # Create dataset with same test split as organic
        synthetic_hallucinations_llm_data = DatasetDict(
            {
                "train": synthetic_hallucinations_llm_train_data,
                "test": organic_hallucinations_data["test"],
            }
        )

        return synthetic_hallucinations_llm_data

    def _function_to_map_synthetic_halls_llm(
        self,
        entry: dict,
        fewshot_examples: Dataset,
        client: genai.Client,
        gemini_model: str,
        generation_config: types.GenerateContentConfig,
    ) -> dict:
        """Call Gemini LLM to generate a synthetic hallucinated Bosch response.

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
            "response": response.text,
            "class_hall": "Yes",
            "label": 0,
        }

    def _make_perl_data(
        self, data: DatasetDict, sft_data: DatasetDict, seed: int = 12345
    ) -> DatasetDict:
        perl_data = DatasetDict(
            {
                "train": data["validation"],
                "test": data["test"].select(range(10)),
            }
        )

        return perl_data

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        """For the autorater, we want to measure its ability on all samples, so we just merge all of them and save as a single test split."""

        autorater_dataset = concatenate_datasets(
            [data["train"], data["validation"], data["test"]]
        )

        return autorater_dataset

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        evaluation_data = DatasetDict({"test": data["train"]})

        return evaluation_data

    @staticmethod
    def _flip_entries(
        data: DatasetDict, data_to_flip: DatasetDict
    ) -> DatasetDict:
        invert_class_hall = {"Yes": "No", "No": "Yes"}

        for split in data.keys():
            ids_to_flip = set(data_to_flip[split]["sample_id"])

            data[split] = data[split].map(
                lambda entry: {
                    "label": 1 - entry["label"]
                    if entry.get("sample_id") in ids_to_flip
                    else entry["label"],
                    "class_hall": invert_class_hall[entry["class_hall"]]
                    if entry.get("sample_id") in ids_to_flip
                    else entry["class_hall"],
                }
            )

        return data

    @staticmethod
    def _rm_prompt(entry: dict) -> dict:
        """Format a reward model prompt for a Bosch entry.

        Args:
            entry (dict): Entry with 'prompt' and 'response' fields.

        Returns:
            dict: Entry with updated 'prompt' field.
        """

        entry["prompt"] += entry["response"]

        return entry

    @classmethod
    def get_formatting_prompts_and_response_template(
        cls,
        eos_token: str,
        fewshot_examples: Optional[Dataset] = None,
        model_repo_id: Optional[str] = None,
    ) -> Tuple[Callable[[Any], list[str]], str]:
        """Return a formatting function and response template for Bosch SFT.

        For Bosch the response template is fixed and the formatting function
        appends the response to the prompt (the RM prompt already contains the
        rest). This mirrors the old `bosch_formatting_prompts_func` behavior.
        """

        response_template = "\nAnswer to user's question:\n"

        def formatting_prompts_func(entry: dict) -> list[str]:
            template = (
                "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information given.\n"
                "User question:\n{question}"
                "\nManual information:\n{context}"
                "\nAnswer to user's question:\n{answer}{eos_token}"
            )

            output_texts = []

            for i in range(len(entry["Question"])):
                formatted_prompt = template.format(
                    question=entry["Question"][i],
                    context=entry["Context"][i],
                    answer=entry["Answer"][i],
                    eos_token=eos_token,
                )

                output_texts.append(formatted_prompt)

            return output_texts

        return formatting_prompts_func, response_template

    @staticmethod
    def _split_nonhallucinated_for_synthetic(
        data_split, num_synth_hallus, seed
    ):
        """
        Helper to split a data split into:
        - to_become_hallus: random non-hallucinated samples to become hallucinations
        - non_hallucinated_data_rest: the rest of non-hallucinated samples
        """
        non_hallucinated_data = data_split.filter(
            lambda entry: entry["class_hall"] == "No"
        )
        to_become_hallus = non_hallucinated_data.shuffle(seed=seed).select(
            range(num_synth_hallus)
        )
        non_hallucinated_data_rest = non_hallucinated_data.filter(
            lambda x: x not in to_become_hallus
        )

        return to_become_hallus, non_hallucinated_data_rest

    @staticmethod
    def _rm_synthetic_hall_llm(entry, fewshot_examples=None):
        """Generate a prompt for LLM-based synthetic hallucination creation for Bosch entries.

        Args:
            entry (dict): Entry with question, context, and response.
            fewshot_examples (datasets.Dataset, optional): Few-shot examples to include. Defaults to None.

        Returns:
            str: Prompt for LLM to generate a hallucinated answer.
        """

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

    @classmethod
    def get_evaluator_prompt(
        cls,
    ) -> Callable[[dict, Optional[Dataset], bool], dict]:
        """Return a function that builds evaluator prompts for Bosch entries."""

        def bosch_evaluator_prompt(
            entry: dict,
            fewshot_examples: Optional[Dataset] = None,
            use_true_label: bool = False,
        ) -> dict:
            header = (
                "You are an expert linguist and fact-checker responsible for"
                " determining when an answer contains information not supported"
                " by a source text.\n\n"
                "Task:\n"
                "1. A user asks a question about their car\n"
                "2. You receive an excerpt from the car's manual\n"
                "3. You examine a proposed answer to the user's question\n\n"
                "Evaluation Criteria:\n"
                "- If the answer makes ANY claim not directly supported by the"
                " manual, respond with 'Yes'\n"
                "- If the answer only rephrases information from the manual"
                " without adding new claims, respond with 'No'\n"
                "- If the answer is FULLY supported by the manual, but truncated"
                " at the last sentence, answer 'No'\n"
                "- Introduction/conclusion courtesy phrases are allowed. If all"
                " other sentences are supported by the context, answer 'No'\n\n"
            )

            example_template = (
                "Example {num}:\n"
                "Question: {question}\n\n"
                "Manual excerpt: {context}\n\n"
                "Proposed answer: {response}\n\n"
                "Does the proposed answer state anything not supported by the"
                " information in the manual? (Yes/No): {ans}\n\n"
            )

            task_template = (
                "Task to Evaluate:\n"
                "Question: {question}\n\n"
                "Manual excerpt: {context}\n\n"
                "Proposed answer: {response}\n\n"
                "Does the proposed answer state anything not supported by the"
                " information in the manual? (Yes/No):"
            )

            response = (
                entry["response"] if use_true_label else entry["completion"]
            )

            prompt = header

            if fewshot_examples is not None and len(fewshot_examples) > 0:
                for idx, fewshot_example in enumerate(fewshot_examples):
                    fewshot_prompt = example_template.format(
                        num=idx + 1,
                        question=fewshot_example["Question"],
                        context=fewshot_example["Context"],
                        response=fewshot_example["response"],
                        ans=fewshot_example["class_hall"],
                    )
                    prompt += fewshot_prompt

            formatted_task = task_template.format(
                question=entry["Question"],
                context=entry["Context"],
                response=response,
            )
            prompt += formatted_task

            entry["evaluator_prompt"] = prompt
            return entry

        return bosch_evaluator_prompt

    @staticmethod
    def _synthetic_hall_structured(entry: dict, rouge_metric: Any) -> dict:
        """Create a structured synthetic hallucination in Bosch data by removing sentences from the context.

        Args:
            entry (dict): Entry to modify.
            data (datasets.Dataset): The full validation dataset for context.

        Returns:
            dict: Entry with modified context and updated labels.
        """

        # Break response into sentences and filter out small ones
        sentences_context = nltk.sent_tokenize(entry["Context"])
        sentences_response = nltk.sent_tokenize(entry["response"])

        # Pick a random sentence in generation
        random_sentence_response = random.sample(sentences_response, 1)

        # Compute the ROUGE-1 score between this and sentences in context
        # and pick sentence in context with maximal ROUGE-1 score to erase
        best_idx = 0
        max_rouge = 0
        for idx_context, sentence_context in enumerate(sentences_context):
            rouge_results = rouge_metric.compute(
                predictions=random_sentence_response,
                references=[sentence_context],
            )
            if rouge_results["rouge1"] > max_rouge:
                best_idx = idx_context
                max_rouge = rouge_results["rouge1"]

        # Store information about erased sentence
        entry["erased_context"] = sentences_context[best_idx]
        entry["rouge1_score"] = max_rouge

        # Erase the chosen sentence and join everything back together
        sentences_context[best_idx] = ""
        new_context = " ".join(sentences_context)
        entry["Context"] = new_context
        entry["class_hall"] = "Yes"
        entry["label"] = 0
        entry["prompt"] = (
            "You are a helpful assistant to car related questions. You will be given an user's question, and the relevant part of the car manual. Your task is to answer the user's question using the information giver. Do not add to your answer any information other than those present in the manual excerpt.\n"
            + "User question:\n"
            + entry["Question"]
            + "\nManual information:\n"
            + entry["Context"]
            + "\nAnswer to user's question:\n"
        )

        # Important: you need to retokenize these later!
        return entry
