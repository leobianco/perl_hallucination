from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

from src.task_processors.base_task_processor import BaseTaskProcessor


class RagtruthTaskProcessor(BaseTaskProcessor):
    def _load_data(self) -> DatasetDict:
        data = load_dataset(self.args.hf_repo)
        return data

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
        processed = {}

        for split in data.keys():
            split_data = data[split]
            split_data = split_data.rename_column("source_info", "context")
            split_data = split_data.rename_column("response", "completion")
            split_data = split_data.rename_column("labels", "annotations")
            split_data = split_data.filter(self._filter_large_entries)
            split_data = split_data.map(self._create_hallucination_labels)
            processed[split] = split_data

        processed = DatasetDict(processed)

        return processed

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        sft_data = deepcopy(data)

        for split in sft_data.keys():
            sft_data[split] = sft_data[split].filter(
                lambda entry: entry["label"] == 1
            )

        return sft_data

    def _make_organic_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        # We just need to make the RM prompt by concatenating the prompt
        # in the dataset with the response
        rm_organic_data = data.map(self._rm_prompt)

        return rm_organic_data

    def _make_structured_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        pass

    def _make_llm_hallucinations_data(
        self, data: DatasetDict, organic_hallucinations_data: DatasetDict
    ) -> DatasetDict:
        pass

    def _make_perl_data(
        self, data: DatasetDict, sft_data: DatasetDict, seed: int = 12345
    ) -> DatasetDict:
        xsum_val = load_dataset("EdinburghNLP/xsum", split="validation")
        xsum_val = xsum_val.rename_column("document", "context")
        xsum_val = xsum_val.rename_column("summary", "completion")
        xsum_val = xsum_val.filter(self._filter_large_entries)
        xsum_val = xsum_val.map(self._xsum_to_prompt)

        # TODO: make the hard coded 250 a parameter set by user
        perl_data = xsum_val.shuffle(seed=seed).select(range(250))

        return perl_data

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        """For the autorater, we want to measure its ability on all samples, so we just merge all of them and save as a single test split."""

        autorater_dataset = concatenate_datasets([data["train"], data["test"]])

        return autorater_dataset

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        xsum_test = load_dataset("EdinburghNLP/xsum", split="test")
        xsum_test = xsum_test.rename_column("document", "context")
        xsum_test = xsum_test.rename_column("summary", "completion")
        xsum_test = xsum_test.filter(self._filter_large_entries)
        xsum_test = xsum_test.map(self._xsum_to_prompt)

        return xsum_test

    @staticmethod
    def _create_hallucination_labels(entry):
        entry["label"] = 1 if len(entry["annotations"]) == 0 else 0
        entry["class_hall"] = "No" if len(entry["annotations"]) == 0 else "Yes"
        return entry

    @staticmethod
    def _rm_prompt(entry: dict) -> dict:
        """Format a reward model prompt for a RAGTruth entry.

        Args:
            entry (dict): Entry with 'prompt' and 'completion' fields.

        Returns:
            dict: Entry with updated 'prompt' field.
        """

        entry["prompt"] += entry["completion"]

        return entry

    @staticmethod
    def _filter_large_entries(entry: dict) -> dict:
        is_short = (
            True
            if len(entry["context"] + entry["completion"]) < 5000
            else False
        )

        return is_short

    @staticmethod
    def _xsum_to_prompt(entry: dict) -> dict:
        template = """Summarize the following news within {x} words:\n{context}\n\noutput:"""

        # The RAGTruth paper states x = min(200, word_num_of_news//4)
        word_num_of_news = len(entry["context"].split())

        entry["prompt"] = template.format(
            x=min(200, word_num_of_news // 4), context=entry["context"]
        )

        return entry

    @classmethod
    def get_formatting_prompts_and_response_template(
        cls,
        eos_token: str,
        fewshot_examples: Optional[Dataset] = None,
        model_repo_id: Optional[str] = None,
    ) -> Tuple[Callable[[Any], list[str]], str]:
        """Provide a simple formatting function and response template for Ragtruth.

        It returns a function that expects an example and returns it unchanged, and a minimal response template matching the old behavior.
        """

        response_template = "\n\noutput:\n"

        def formatting_prompts_func(entry: dict) -> list[str]:
            template = "{context}\n{completion}{eos_token}"

            output_texts = []

            for i in range(len(entry["context"])):
                formatted_prompt = template.format(
                    context=entry["context"][i],
                    completion=entry["completion"][i],
                    eos_token=eos_token,
                )

                output_texts.append(formatted_prompt)

            return output_texts

        return formatting_prompts_func, response_template

    @classmethod
    def get_evaluator_prompt(
        cls,
    ) -> Callable[[dict, Optional[Dataset], bool], dict]:
        """Return a function that builds evaluator prompts for Ragtruth entries."""

        def ragtruth_evaluator_prompt(
            entry: dict,
            fewshot_examples: Optional[Dataset] = None,
            use_true_label: bool = False,
        ) -> dict:
            preamble = 'You are an expert evaluator specializing in detecting hallucinations in text summarization. Your task is to determine whether a summary contains any information not present in the original text.\n\nIn this evaluation:\n1. You will be given an original text that was meant to be summarized\n2. You will examine a summary generated from that text\n3. You must determine if the summary contains ANY information, statements, claims, or implications that are NOT supported by the original text\n\nEvaluation guidelines:\n- A hallucination occurs when the summary includes facts, details, or conclusions absent from the original text\n- Reasonable paraphrasing and reorganization of information is acceptable\n- Focus only on factual accuracy, not on style, tone, or format\n- Be vigilant about subtle additions that might seem plausible but aren\'t in the source\n\nYour response must be ONLY "Yes" (hallucination detected) or "No" (no hallucination detected).\n'

            template = """\n\nOriginal text to be summarized: {context}\n{completion}\n\nEvaluation process:\n1. Read the original text carefully\n2. Examine each claim or statement in the summary (output)\n3. Verify that every piece of information in the summary (output) is supported by the original text\n4. Check for subtle additions, expansions, or assumptions not justified by the original\n\nDoes the summary (output) contain ANY information not present in or directly inferable from the original text? (Yes/No):{ans}\n"""

            formatted_prompt = template.format(
                context=entry["context"],
                completion=entry["completion"],
                ans="",
            )

            prompt = preamble

            if fewshot_examples is not None:
                for fewshot_example in fewshot_examples:
                    fewshot_prompt = template.format(
                        context=fewshot_example["context"],
                        completion=fewshot_example["completion"],
                        ans=fewshot_example["class_hall"],
                    )
                    prompt += fewshot_prompt
                prompt += formatted_prompt
            else:
                prompt += formatted_prompt

            entry["evaluator_prompt"] = prompt
            return entry

        return ragtruth_evaluator_prompt
