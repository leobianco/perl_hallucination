from typing import Any, Callable, Optional, Tuple

from datasets import Dataset, DatasetDict

from src.task_processors.base_task_processor import BaseTaskProcessor


class RagtruthTaskProcessor(BaseTaskProcessor):
    def _load_data(self) -> DatasetDict:
        pass

    def _preprocess_data(self, data: DatasetDict) -> DatasetDict:
        pass

    def _make_sft_data(self, data: DatasetDict) -> DatasetDict:
        pass

    def _make_organic_hallucinations_data(
        self, data: DatasetDict
    ) -> DatasetDict:
        pass

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
        pass

    def _make_autorater_data(self, data: DatasetDict) -> Dataset:
        pass

    def _make_evaluation_data(self, data: DatasetDict) -> DatasetDict:
        pass

    @classmethod
    def get_formatting_prompts_and_response_template(
        cls,
        eos_token: str,
        fewshot_examples: Optional[Dataset] = None,
        model_repo_id: Optional[str] = None,
    ) -> Tuple[Callable[[Any], list[str]], str]:
        """Provide a simple formatting function and response template for Ragtruth.

        This is a placeholder until a full implementation is available. It returns
        a function that expects an example and returns it unchanged, and a
        minimal response template matching the old behavior.
        """

        response_template = "\n\noutput:\n"

        def formatting_prompts_func(example):
            # No-op formatting for now
            return example

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
            preamble = 'You are an expert evaluator specializing in detecting hallucinations in text summarization. Your task is to determine whether a summary contains any information not present in the original text.\n\nIn this evaluation:\n1. You will be given an original text that was meant to be summarized\n2. You will examine a summary generated from that text\3. You must determine if the summary contains ANY information, statements, claims, or implications that are NOT supported by the original text\n\nEvaluation guidelines:\n- A hallucination occurs when the summary includes facts, details, or conclusions absent from the original text\n- Reasonable paraphrasing and reorganization of information is acceptable\n- Focus only on factual accuracy, not on style, tone, or format\n- Be vigilant about subtle additions that might seem plausible but aren\'t in the source\n\nYour response must be ONLY "Yes" (hallucination detected) or "No" (no hallucination detected).\n'

            template = """\n\nOriginal text to be summarized: {user_query}\n{response}\n\nEvaluation process:\n1. Read the original text carefully\n2. Examine each claim or statement in the summary (output)\n3. Verify that every piece of information in the summary (output) is supported by the original text\n4. Check for subtle additions, expansions, or assumptions not justified by the original\n\nDoes the summary (output) contain ANY information not present in or directly inferable from the original text? (Yes/No):{ans}\n"""

            response = (
                entry["response"] if use_true_label else entry["completion"]
            )

            formatted_prompt = template.format(
                user_query=entry["user_query"],
                response=response,
                ans="",
            )

            prompt = preamble

            if fewshot_examples is not None:
                for fewshot_example in fewshot_examples:
                    fewshot_prompt = template.format(
                        user_query=fewshot_example["user_query"],
                        response=fewshot_example["response"],
                        ans=fewshot_example["class_hall"],
                    )
                    prompt += fewshot_prompt
                prompt += formatted_prompt
            else:
                prompt += formatted_prompt

            entry["evaluator_prompt"] = prompt
            return entry

        return ragtruth_evaluator_prompt
