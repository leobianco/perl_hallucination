import abc


class BaseTaskProcessor(abc.ABC):
    def __init__(self, args):
        self.args = args

    @abc.abstractmethod
    def _load_data(self):
        pass

    @abc.abstractmethod
    def _preprocess_data(self, data):
        pass

    @abc.abstractmethod
    def _make_sft_data(self, data):
        pass

    @abc.abstractmethod
    def _make_organic_hallucinations_data(self, data):
        pass

    @abc.abstractmethod
    def _make_structured_hallucinations_data(self, data):
        pass

    @abc.abstractmethod
    def _make_llm_hallucinations_data(self, data):
        pass

    @abc.abstractmethod
    def _make_perl_data(self, data, sft_data, seed=12345):
        pass

    @abc.abstractmethod
    def _make_autorater_data(self, data):
        pass

    @abc.abstractmethod
    def _make_evaluation_data(self, data):
        pass

    def run(self):
        data = self._load_data()
        data = self._preprocess_data(data)
        data.push_to_hub(repo_id=self.args.task_name + "_processed")

        autorater_data = self._make_autorater_data(data)
        autorater_data.push_to_hub(
            repo_id=self.args.task_name + "_autorater", split="test"
        )

        organic_hallucinations_data = self._make_organic_hallucinations_data(
            data
        )
        organic_hallucinations_data.push_to_hub(
            repo_id=self.args.task_name + "_rm_organic"
        )

        structured_hallucinations_data = (
            self._make_structured_hallucinations_data(data)
        )
        structured_hallucinations_data.push_to_hub(
            repo_id=self.args.task_name + "_rm_synthetic_struct"
        )

        llm_hallucinations_data = self._make_llm_hallucinations_data(data, organic_hallucinations_data)
        llm_hallucinations_data.push_to_hub(
            repo_id=self.args.task_name + "_rm_synthetic_llm"
        )

        sft_data = self._make_sft_data(data)
        sft_data.push_to_hub(repo_id=self.args.task_name + "_sft")

        perl_data = self._make_perl_data(data)
        perl_data.push_to_hub(repo_id=self.args.task_name + "_perl")

        evaluation_data = self._make_evaluation_data(data)
        evaluation_data.push_to_hub(
            repo_id=self.args.task_name + "_final_test_set"
        )
