import json
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock

if "pandas" not in sys.modules:
  try:
    import pandas
  except ImportError:
    pandas_mock = types.ModuleType("pandas")
    pandas_mock.DataFrame = MagicMock
    pandas_mock.isna = lambda x: x is None
    sys.modules["pandas"] = pandas_mock

try:
  from datasets import Dataset, DatasetDict
except ImportError:

  class _MockDataset:

    def __init__(self, data):
      self._data = dict(data)
      self.column_names = list(data.keys())

    def __getitem__(self, key):
      return self._data[key]

    def __len__(self):
      first_col = next(iter(self._data.values())) if self._data else []
      return len(first_col)

    def to_dict(self):
      return dict(self._data)

    def add_column(self, column_name, values):
      new_data = dict(self._data)
      new_data[column_name] = values
      return _MockDataset(new_data)

    def remove_columns(self, column_name):
      new_data = {k: v for k, v in self._data.items() if k != column_name}
      return _MockDataset(new_data)

    def rename_column(self, original_column_name, new_column_name):
      new_data = {
          (new_column_name if k == original_column_name else k): v
          for k, v in self._data.items()
      }
      return _MockDataset(new_data)

    def select_columns(self, column_names):
      new_data = {k: v for k, v in self._data.items() if k in column_names}
      return _MockDataset(new_data)

    def select(self, indices):
      new_data = {k: [v[i] for i in indices] for k, v in self._data.items()}
      return _MockDataset(new_data)

    def shuffle(self, seed=42):
      return self

    def map(self, fn):
      num_rows = len(next(iter(self._data.values())))
      new_rows = []
      for i in range(num_rows):
        entry = {k: self._data[k][i] for k in self._data}
        res = fn(entry)
        entry.update(res)
        new_rows.append(entry)
      new_data = {k: [row[k] for row in new_rows] for k in new_rows[0]}
      return _MockDataset(new_data)

  class Dataset:

    @classmethod
    def from_dict(cls, d):
      return _MockDataset(d)

  class DatasetDict(dict):
    pass

  datasets_mock = types.ModuleType("datasets")
  datasets_mock.Dataset = Dataset
  datasets_mock.DatasetDict = DatasetDict
  datasets_mock.concatenate_datasets = lambda d_list: _MockDataset(
      {k: [item for d in d_list for item in d._data[k]] for k in d_list[0]._data}
  )
  datasets_mock.load_dataset = MagicMock()
  sys.modules["datasets"] = datasets_mock

if "google" not in sys.modules:
  google_mock = types.ModuleType("google")
  genai_mock = types.ModuleType("genai")
  genai_mock.Client = MagicMock
  types_mock = types.ModuleType("types")
  types_mock.GenerateContentConfig = MagicMock
  genai_mock.types = types_mock
  google_mock.genai = genai_mock
  sys.modules["google"] = google_mock
  sys.modules["google.genai"] = genai_mock
  sys.modules["google.genai.types"] = types_mock

from src.task_processors.npov_task_processor import NPOVTaskProcessor


class TestNPOVTaskProcessor(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
