from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    task_name: str = field(
        metadata={"help": "Name of the task (NPOV, HalOmi)."}
    )

    user: str = field(
        metadata={
            "help": "The user to use for writing and loading to and from HF."
        },
    )

    writer_model_lora: str = field(
        metadata={"help": "The path to the LoRA adapters of the writer model."}
    )

    dataset_labels: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset with hallucination labels, for evaluation of evaluator or for getting few-shot examples."
        },
    )

    dataset_labels_split: Optional[str] = field(
        default=None,
        metadata={"help": "What split of the dataset_labels to use."},
    )

    dataset_prompts: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset with prompts to be used for generation (not necessarily has hallucination labels)."
        },
    )

    dataset_prompts_split: Optional[str] = field(
        default=None,
        metadata={"help": "What split of the dataset_prompts to use."},
    )

    writer_model_base: Optional[str] = field(
        default=None,
        metadata={"help": "The base model for the writer (name or path)."},
    )

    evaluator_model: str = field(
        default="google/gemma-2-27b-it",
        metadata={
            "help": "The model name or path to the model to use as evaluator."
        },
    )

    use_gemini: bool = field(
        default=True,
        metadata={"help": "Using the latest Gemini model as evaluator"},
    )

    gemini_api_key: Optional[str] = field(
        default="", metadata={"help": "API key for calling Gemini"}
    )

    seed: int = field(default=12345)

    eval_batch_size: int = field(default=1)

    max_tokens: int = field(default=128)

    temperature: float = field(default=1)

    top_p: float = field(default=1)

    top_k: int = field(
        default=0,
        metadata={
            "help": "The number of highest probability vocabulary tokens to keep for top-k-filtering. 0 means no top-k filtering."
        },
    )

    evaluator_num_fewshot: Optional[int] = field(
        default=0,
        metadata={
            "help": "The number of fewshot examples to give to the evaluator. Half will be positive (contain hallucination), half will be negative."
        },
    )

    evaluate_evaluator: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation of the evaluator or not."},
    )

    threshold: float = field(
        default=0.5,
        metadata={
            "help": "The value of the threshold to turn scores into classif."
        },
    )

    dataset_with_completions: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the dataset with completions on HF Hub. Required when evaluate_evaluator is False and not generating."
        },
    )

    writer_num_fewshot: int = field(
        default=0,
        metadata={
            "help": "Number of few-shot examples to prepend to each prompt. If zero, no few-shot examples are prepended."
        },
    )
