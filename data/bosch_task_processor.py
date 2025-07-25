from data.base_task_processor import BaseTaskProcessor
from datasets import load_dataset, concatenate_datasets


class BoschTaskProcessor(BaseTaskProcessor):
    def _load_data(self):
        data = load_dataset(
            self.args.hf_repo,
            data_files={
                "train": "train.csv",
                "validation": "val.csv",
                "test": "test.csv",
            },
        )

        return data

    def _preprocess_data(self, data):
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

    def _make_sft_data(self, data):
        """For the Bosch task, we SFT on the non-hallucinated samples of the validation split."""

        sft_data = data["test"].filter(
            lambda entry: entry["class_hall"] == "No"
        )

        return sft_data

    def _make_organic_hallucinations_data(self, data):
        organic_hallucinations_data = data["test"]

        return organic_hallucinations_data

    def _make_structured_hallucinations_data(self, data):
        pass

    def _make_llm_hallucinations_data(self, data, organic_hallucinations_data):
        pass

    def _make_perl_data(self, data, sft_data, seed=12345):
        pass

    def _make_autorater_data(self, data):
        """For the autorater, we want to measure its ability on all samples, so we just merge all of them and save as a single test split."""

        autorater_dataset = concatenate_datasets(
            data["train"], data["validation"], data["test"]
        )

        return autorater_dataset

    def _make_evaluation_data(self, data):
        pass

    @staticmethod
    def _flip_entries(data, data_to_flip):
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
