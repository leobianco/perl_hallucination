import json
import unittest
from unittest.mock import Mock

from datasets import Dataset, DatasetDict
from src.task_processors.npov_task_processor import NPOVTaskProcessor


class TestNPOVDataPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic raw dataset (one entry of npov_rm_raw)
        raw_data_json_string = """[{"topic":"Bottled Water Ban","user_query":"Should Bottled Water Be Banned?","npov_response_intro":"As I've seen, this topic is very divisive and has no clear consensus. But I can list some common arguments on both sides if you want.","npov_response_main":"One of the biggest arguments for banning bottled water is that it would protect local water supplies. But one of the main arguments against that is that it's a practical emergency water supply. People who oppose the ban also point to the fact that many people prefer bottled water.","sampling_temperature":"0.7","perspective_1_name":"pro","perspective_2_name":"con","pro_argument_1_GIVEN":"Banning bottled water would protect local water supplies.","pro_argument_1_SEEN":"it would protect local water supplies","pro_argument_2_GIVEN":null,"pro_argument_2_SEEN":null,"pro_argument_3_GIVEN":null,"pro_argument_3_SEEN":null,"con_argument_1_GIVEN":"Bottled water is a practical emergency water supply.","con_argument_1_SEEN":"it's a practical emergency water supply","con_argument_2_GIVEN":null,"con_argument_2_SEEN":null,"con_argument_3_GIVEN":null,"con_argument_3_SEEN":null,"hallucination words":"many people prefer bottled water","uncovered words":null,"has hallucination":"YES","hallucination type":"full","has coverage issue":"NO","coverage issue type":null,"perspective_2":"con: Bottled water is a practical emergency water supply.","perspective_1":"pro: Banning bottled water would protect local water supplies.","split":"TEST","uid":"63e25721c04b6048f7617ad2a9eaeb56_0.7","has synthetic hallucination":"NO","has synthetic coverage issue":"NO","num_args_perspective_1":1,"num_args_perspective_2":1,"npov_response":"As I've seen, this topic is very divisive and has no clear consensus. But I can list some common arguments on both sides if you want. One of the biggest arguments for banning bottled water is that it would protect local water supplies. But one of the main arguments against that is that it's a practical emergency water supply. People who oppose the ban also point to the fact that many people prefer bottled water.","is_error":true,"is_synthetic_error":false,"is_ambiguous_error":false},
        {"topic":"Golf","user_query":"Is Golf a Sport?","npov_response_intro":"This is a complicated question with no straightforward answer.","npov_response_main":"Those who say that Golf meets the definition of sport since it requires skill to play, point out that it also meets the definition in that it requires coordination and endurance. On the other hand, those who oppose this claim say that Golf does not require enough skill to be considered a sport.","sampling_temperature":"0.7","perspective_1_name":"pro","perspective_2_name":"con","pro_argument_1_GIVEN":"Golf meets the definition of sport by requiring skill to play.","pro_argument_1_SEEN":"Golf meets the definition of sport since it requires skill to play","pro_argument_2_GIVEN":null,"pro_argument_2_SEEN":null,"pro_argument_3_GIVEN":null,"pro_argument_3_SEEN":null,"con_argument_1_GIVEN":"Golf does not require enough skill to meet the definition of sport.","con_argument_1_SEEN":"Golf does not require enough skill to be considered a sport","con_argument_2_GIVEN":null,"con_argument_2_SEEN":null,"con_argument_3_GIVEN":null,"con_argument_3_SEEN":null,"hallucination words":"it also meets the definition in that it requires coordination and endurance","uncovered words":null,"has hallucination":"YES","hallucination type":"full","has coverage issue":"NO","coverage issue type":null,"perspective_2":"con: Golf does not require enough skill to meet the definition of sport.","perspective_1":"pro: Golf meets the definition of sport by requiring skill to play.","split":"DEV","uid":"b4062df752d9d8dc0090f8bec6150018_0.7","has synthetic hallucination":"NO","has synthetic coverage issue":"NO","num_args_perspective_1":1,"num_args_perspective_2":1,"npov_response":"This is a complicated question with no straightforward answer. Those who say that Golf meets the definition of sport since it requires skill to play, point out that it also meets the definition in that it requires coordination and endurance. On the other hand, those who oppose this claim say that Golf does not require enough skill to be considered a sport.","is_error":true,"is_synthetic_error":false,"is_ambiguous_error":false}]"""

        raw_data_dict = json.loads(raw_data_json_string)

        keys = list(raw_data_dict[0].keys())
        transformed_raw_data_dict = {
            key: [d[key] for d in raw_data_dict] for key in keys
        }

        raw_data_hf = Dataset.from_dict(transformed_raw_data_dict)
        raw_datasetdict_hf = DatasetDict(
            {"train": raw_data_hf, "test": raw_data_hf}
        )

        # Call the preprocessing method on it
        mock_args = Mock()
        processor_cls = NPOVTaskProcessor(mock_args)
        cls.data = processor_cls._preprocess_data(raw_datasetdict_hf)

    def test_column_label(self):
        """Test for the existence of the `label' column, and that it only has integers."""

        for split in self.data.keys():
            self.assertIn("label", self.data[split].column_names)
            for value in self.data[split]["label"]:
                self.assertIsInstance(value, int)

    def test_column_class_hall(self):
        """Test for the existence of the `class_hall' column, that it contains strings, and that these strings are either `Yes' or `No'."""

        for split in self.data.keys():
            self.assertIn("class_hall", self.data[split].column_names)
            for value in self.data[split]["class_hall"]:
                self.assertIsInstance(value, str)
                self.assertIn(value, ["Yes", "No"])

    def test_column_prompt(self):
        """Test for the existence of the `prompt' column, that it contains strings."""

        for split in self.data.keys():
            self.assertIn("prompt", self.data[split].column_names)
            for value in self.data[split]["prompt"]:
                self.assertIsInstance(value, str)


if __name__ == "__main__":
    unittest.main()
