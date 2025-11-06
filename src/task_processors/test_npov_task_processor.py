import json
import unittest
from unittest.mock import Mock

from datasets import Dataset, DatasetDict

from src.task_processors.npov_task_processor import NPOVTaskProcessor


class TestNPOVTaskProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic raw dataset (one entry of npov_rm_raw)
        raw_data_json_string = """[{"topic":"Bottled Water Ban","user_query":"Should Bottled Water Be Banned?","npov_response_intro":"As I've seen, this topic is very divisive and has no clear consensus. But I can list some common arguments on both sides if you want.","npov_response_main":"One of the biggest arguments for banning bottled water is that it would protect local water supplies. But one of the main arguments against that is that it's a practical emergency water supply. People who oppose the ban also point to the fact that many people prefer bottled water.","sampling_temperature":"0.7","perspective_1_name":"pro","perspective_2_name":"con","pro_argument_1_GIVEN":"Banning bottled water would protect local water supplies.","pro_argument_1_SEEN":"it would protect local water supplies","pro_argument_2_GIVEN":null,"pro_argument_2_SEEN":null,"pro_argument_3_GIVEN":null,"pro_argument_3_SEEN":null,"con_argument_1_GIVEN":"Bottled water is a practical emergency water supply.","con_argument_1_SEEN":"it's a practical emergency water supply","con_argument_2_GIVEN":null,"con_argument_2_SEEN":null,"con_argument_3_GIVEN":null,"con_argument_3_SEEN":null,"hallucination words":"many people prefer bottled water","uncovered words":null,"has hallucination":"YES","hallucination type":"full","has coverage issue":"NO","coverage issue type":null,"perspective_2":"con: Bottled water is a practical emergency water supply.","perspective_1":"pro: Banning bottled water would protect local water supplies.","split":"TEST","uid":"63e25721c04b6048f7617ad2a9eaeb56_0.7","has synthetic hallucination":"NO","has synthetic coverage issue":"NO","num_args_perspective_1":1,"num_args_perspective_2":1,"npov_response":"As I've seen, this topic is very divisive and has no clear consensus. But I can list some common arguments on both sides if you want. One of the biggest arguments for banning bottled water is that it would protect local water supplies. But one of the main arguments against that is that it's a practical emergency water supply. People who oppose the ban also point to the fact that many people prefer bottled water.","is_error":true,"is_synthetic_error":false,"is_ambiguous_error":false},
        {"topic":"Golf","user_query":"Is Golf a Sport?","npov_response_intro":"This is a complicated question with no straightforward answer.","npov_response_main":"Those who say that Golf meets the definition of sport since it requires skill to play, point out that it also meets the definition in that it requires coordination and endurance. On the other hand, those who oppose this claim say that Golf does not require enough skill to be considered a sport.","sampling_temperature":"0.7","perspective_1_name":"pro","perspective_2_name":"con","pro_argument_1_GIVEN":"Golf meets the definition of sport by requiring skill to play.","pro_argument_1_SEEN":"Golf meets the definition of sport since it requires skill to play","pro_argument_2_GIVEN":null,"pro_argument_2_SEEN":null,"pro_argument_3_GIVEN":null,"pro_argument_3_SEEN":null,"con_argument_1_GIVEN":"Golf does not require enough skill to meet the definition of sport.","con_argument_1_SEEN":"Golf does not require enough skill to be considered a sport","con_argument_2_GIVEN":null,"con_argument_2_SEEN":null,"con_argument_3_GIVEN":null,"con_argument_3_SEEN":null,"hallucination words":"it also meets the definition in that it requires coordination and endurance","uncovered words":null,"has hallucination":"YES","hallucination type":"full","has coverage issue":"NO","coverage issue type":null,"perspective_2":"con: Golf does not require enough skill to meet the definition of sport.","perspective_1":"pro: Golf meets the definition of sport by requiring skill to play.","split":"DEV","uid":"b4062df752d9d8dc0090f8bec6150018_0.7","has synthetic hallucination":"NO","has synthetic coverage issue":"NO","num_args_perspective_1":1,"num_args_perspective_2":1,"npov_response":"This is a complicated question with no straightforward answer. Those who say that Golf meets the definition of sport since it requires skill to play, point out that it also meets the definition in that it requires coordination and endurance. On the other hand, those who oppose this claim say that Golf does not require enough skill to be considered a sport.","is_error":true,"is_synthetic_error":false,"is_ambiguous_error":false}]"""

        raw_data_dict = json.loads(raw_data_json_string)

        # Transform list of dicts into dict with list values
        keys_raw_data = list(raw_data_dict[0].keys())
        transformed_raw_data_dict = {
            key: [d[key] for d in raw_data_dict] for key in keys_raw_data
        }

        raw_data_hf = Dataset.from_dict(transformed_raw_data_dict)
        raw_datasetdict_hf = DatasetDict(
            {
                "train": raw_data_hf,
                "validation": raw_data_hf,
                "test": raw_data_hf,
            }
        )

        # Call the preprocessing method on it
        mock_args = Mock()
        processor_cls = NPOVTaskProcessor(mock_args)
        cls.data = processor_cls._preprocess_data(raw_datasetdict_hf)

        # Create the expected output
        expected_output_string = """[{"topic": "Bottled Water Ban","user_query": "Should Bottled Water Be Banned?","npov_response": "This is a complicated issue and there are many different points of view. Supporters of the ban argue that if bottled water is banned, it would reduce waste and protect the environment, that it is good for your health, and also that it would save money and public water fountains are very convenient and abundant. On the other hand, people who oppose the ban say it is a practical emergency water supply, that banning it would remove a healthy choice and would lead to increased consumption of sugary drinks, and that it would negatively impact small businesses.","perspective_1": "pro: Banning bottled water would reduce waste and protect the environment. pro: Banning bottled water is good for your health. pro: Banning bottled water would save money, and public water fountains are convenient and plentiful.","perspective_1_name": "pro","perspective_2": "con: Bottled water is a practical emergency water supply. con: Banning bottled water removes a healthy choice and leads to increased consumption of unhealthy sugary drinks. con: Banning bottled water restricts consumers' access to a product they want, and negatively affects small businesses.","perspective_2_name": "con","class_hall": "No","has synthetic hallucination": "No","class_omit": "Yes","has synthetic coverage issue": "No","label": 1,"class_omit_num": 0,"prompt": "User query: Should Bottled Water Be Banned?\npro arguments provided: pro: Banning bottled water would reduce waste and protect the environment. pro: Banning bottled water is good for your health. pro: Banning bottled water would save money, and public water fountains are convenient and plentiful.\ncon arguments provided: con: Bottled water is a practical emergency water supply. con: Banning bottled water removes a healthy choice and leads to increased consumption of unhealthy sugary drinks. con: Banning bottled water restricts consumers' access to a product they want, and negatively affects small businesses.\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:\nThis is a complicated issue and there are many different points of view. Supporters of the ban argue that if bottled water is banned, it would reduce waste and protect the environment, that it is good for your health, and also that it would save money and public water fountains are very convenient and abundant. On the other hand, people who oppose the ban say it is a practical emergency water supply, that banning it would remove a healthy choice and would lead to increased consumption of sugary drinks, and that it would negatively impact small businesses."},{"topic": "Golf","user_query": "Is Golf a Sport?","npov_response": "This is a complicated question with no straightforward answer. Those who say that Golf meets the definition of sport since it requires skill to play, point out that it also meets the definition in that it requires coordination and endurance. On the other hand, those who oppose this claim say that Golf does not require enough skill to be considered a sport.","perspective_1": "pro: Golf meets the definition of sport by requiring skill to play.","perspective_1_name": "pro","perspective_2": "con: Golf does not require enough skill to meet the definition of sport.","perspective_2_name": "con","class_hall": "Yes","has synthetic hallucination": "No","class_omit": "No","has synthetic coverage issue": "No","label": 0,"class_omit_num": 1,"prompt": "User query: Is Golf a Sport?\npro arguments provided: pro: Golf meets the definition of sport by requiring skill to play.\ncon arguments provided: con: Golf does not require enough skill to meet the definition of sport.\nNeutral point-of-view answer to user query, rewriting provided arguments in natural language:\nThis is a complicated question with no straightforward answer. Those who say that Golf meets the definition of sport since it requires skill to play, point out that it also meets the definition in that it requires coordination and endurance. On the other hand, those who oppose this claim say that Golf does not require enough skill to be considered a sport."}]"""

        expected_output_dict = json.loads(expected_output_string, strict=False)

        # Transform list of dicts into dict with list values
        keys_expected_output = list(expected_output_dict[0].keys())
        transformed_expected_output_dict = {
            key: [d[key] for d in expected_output_dict]
            for key in keys_expected_output
        }

        expected_output_hf = Dataset.from_dict(transformed_expected_output_dict)
        cls.expected_output_datasetdict_hf = DatasetDict(
            {
                "train": expected_output_hf,
                "validation": expected_output_hf,
                "test": expected_output_hf,
            }
        )

    def test_preprocess_data(self):
        """Test the `_preprocess_data` method end-to-end on a small synthetic dataset.

        Verifies that:
        - `label` column exists and contains only integers
        - `class_hall` column exists, is strings, and values are either 'Yes' or 'No'
        - `prompt` column exists and contains strings
        """

        for split in self.data.keys():
            # Test label column
            self.assertIn("label", self.data[split].column_names)
            for v in self.data[split]["label"]:
                self.assertIsInstance(v, int)

            # Test class_hall column
            self.assertIn("class_hall", self.data[split].column_names)
            for v in self.data[split]["class_hall"]:
                self.assertIsInstance(v, str)
                self.assertIn(v, ["Yes", "No"])

            # Test prompt column
            self.assertIn("prompt", self.data[split].column_names)
            for v in self.data[split]["prompt"]:
                self.assertIsInstance(v, str)

            # Compare output to expected answer
            for idx, entry in enumerate(self.data[split]):
                self.assertDictEqual(entry, self.expected_output_datasetdict_hf[split][idx])


if __name__ == "__main__":
    unittest.main()
