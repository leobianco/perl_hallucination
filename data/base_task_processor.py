import abc


class BaseTaskProcessor(abc.ABC):
    def __init__(self, args):
        self.args = args

    @abc.abstractmethod
    def _load_data(self):
        """Load raw dataset(s) from source (local files, HF repo, etc).

        Returns:
            Raw dataset object(s) as expected by downstream methods.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _preprocess_data(self, data):
        """Preprocess and normalize raw data for later pipeline steps.

        Args:
            data: Raw dataset(s) loaded by `_load_data`.

        Returns:
            Preprocessed dataset(s) suitable for RM/SFT creation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_sft_data(self, data):
        """Create supervised fine-tuning dataset from preprocessed data.

        Args:
            data: Preprocessed dataset(s).

        Returns:
            Dataset (or DatasetDict) for SFT.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_organic_hallucinations_data(self, data):
        """Extract organic (human/annotated) hallucination examples.

        Args:
            data: Preprocessed dataset(s).

        Returns:
            DatasetDict with 'train' and 'test' splits for organic hallucination RM training.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_structured_hallucinations_data(self, data):
        """Create structured synthetic hallucination examples (rule-based).

        Args:
            data: Preprocessed dataset(s).

        Returns:
            DatasetDict with structured synthetic hallucination training/test splits.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_llm_hallucinations_data(self, data):
        """Generate synthetic hallucinations using an LLM and return dataset.

        Args:
            data: Preprocessed dataset(s).

        Returns:
            DatasetDict with LLM-generated hallucination training/test splits.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_perl_data(self, data, sft_data, seed=12345):
        """Create PERL dataset from RM and SFT data.

        Args:
            data: Preprocessed RM dataset(s).
            sft_data: SFT dataset(s).
            seed: Seed for deterministic shuffling.

        Returns:
            DatasetDict for PERL training/evaluation.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_autorater_data(self, data):
        """Prepare dataset for the autorater (combined test set or similar).

        Args:
            data: Preprocessed dataset(s).

        Returns:
            Dataset or DatasetDict used by the autorater.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_evaluation_data(self, data):
        """Prepare evaluation datasets (capped final test sets, etc).

        Args:
            data: Preprocessed dataset(s).

        Returns:
            DatasetDict containing final evaluation splits.
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def get_formatting_prompts_and_response_template(
        cls, eos_token, fewshot_examples=None, model_repo_id=None
    ):
        """Return a tuple (formatting_func, response_template).

        - formatting_func: a callable that formats an example (or batch) into the prompt/inputs expected by the trainer.
        - response_template: a string template used by the trainer for completion handling.

        Subclasses must implement this and may use `fewshot_examples` and `model_repo_id` to customize behavior.
        """
        raise NotImplementedError()

    def run(self):
        args = self.args

        data = self._load_data()
        data = self._preprocess_data(data)
        data.push_to_hub(repo_id=args.task_name + "_processed")

        autorater_data = self._make_autorater_data(data)
        autorater_data.push_to_hub(
            repo_id=args.task_name + "_autorater", split="test"
        )

        organic_hallucinations_data = self._make_organic_hallucinations_data(
            data
        )
        organic_hallucinations_data.push_to_hub(
            repo_id=args.task_name + "_rm_organic"
        )

        if args.synthetic_hallus_struct:
            structured_hallucinations_data = (
                self._make_structured_hallucinations_data(data)
            )
            structured_hallucinations_data.push_to_hub(
                repo_id=args.task_name + "_rm_synthetic_struct"
            )

        if args.synthetic_hallus_llm:
            llm_hallucinations_data = self._make_llm_hallucinations_data(data, organic_hallucinations_data)
            llm_hallucinations_data.push_to_hub(
                repo_id=args.task_name + "_rm_synthetic_llm"
            )

        sft_data = self._make_sft_data(data)
        sft_data.push_to_hub(repo_id=args.task_name + "_sft")

        perl_data = self._make_perl_data(data, sft_data, seed=args.seed)
        perl_data.push_to_hub(repo_id=args.task_name + "_perl")

        evaluation_data = self._make_evaluation_data(data)
        evaluation_data.push_to_hub(
            repo_id=args.task_name + "_final_test_set"
        )
