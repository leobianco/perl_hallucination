"""Unit tests for the SSFO baseline implementation."""

import sys
import types
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

if "torch" not in sys.modules:
  try:
    import torch
  except ImportError:
    import contextlib
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = object
    torch.nn.Linear = MagicMock()
    torch.nn.Identity = MagicMock()
    torch.nn.BCEWithLogitsLoss = object
    torch.nn.CrossEntropyLoss = object
    torch.nn.MSELoss = object
    torch.no_grad = contextlib.nullcontext
    torch.device = lambda x: x
    torch.tensor = lambda x, **kw: MagicMock()
    torch.randn = lambda *x: MagicMock()
    torch.bernoulli = lambda x: MagicMock()
    torch.softmax = lambda x, **kw: MagicMock()
    torch.zeros_like = lambda x: MagicMock()
    torch.sort = lambda x, **kw: (MagicMock(), MagicMock())
    torch.cumsum = lambda x, **kw: MagicMock()
    torch.multinomial = lambda x, **kw: MagicMock()
    torch.full = lambda *x, **kw: MagicMock()
    torch.ones = lambda *x, **kw: MagicMock()
    torch.zeros = lambda *x, **kw: MagicMock()
    torch.where = lambda *x: MagicMock()
    torch.stack = lambda *x, **kw: MagicMock(tolist=lambda: [[10, 11], [12, 13]])
    torch.long = "long"
    torch.bool = "bool"
    torch.bfloat16 = "bfloat16"
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn

if "transformers" not in sys.modules or not hasattr(
    sys.modules["transformers"], "__path__"
):
  transformers_mod = types.ModuleType("transformers")
  transformers_mod.__path__ = []
  transformers_mod.TrainerCallback = object
  transformers_mod.TrainerControl = object
  transformers_mod.TrainerState = object
  transformers_mod.TrainingArguments = object
  transformers_mod.Trainer = object
  transformers_mod.AutoModelForCausalLM = MagicMock()
  transformers_mod.AutoModelForSequenceClassification = MagicMock()
  transformers_mod.AutoConfig = MagicMock()
  transformers_mod.AutoTokenizer = MagicMock()
  transformers_mod.PretrainedConfig = MagicMock()
  transformers_mod.PreTrainedModel = object
  transformers_mod.DataCollatorWithPadding = MagicMock()
  transformers_mod.HfArgumentParser = MagicMock()
  transformers_mod.set_seed = MagicMock()
  sys.modules["transformers"] = transformers_mod
  sys.modules["transformers.trainer_utils"] = types.ModuleType(
      "transformers.trainer_utils"
  )
  sys.modules["transformers.trainer_utils"].get_last_checkpoint = MagicMock(
      return_value=None
  )

for mod_name in [
    "transformers.configuration_utils",
    "transformers.integrations",
    "transformers.integrations.heterogeneity",
    "transformers.integrations.heterogeneity.configuration_utils",
    "trl",
    "peft",
    "vllm",
    "vllm.lora",
    "vllm.lora.request",
    "evaluate",
    "scipy",
    "scipy.special",
    "sklearn",
    "sklearn.metrics",
    "google",
    "google.genai",
    "google.genai.types",
    "huggingface_hub",
    "matplotlib",
    "matplotlib.pyplot",
    "numpy",
    "pandas",
    "nltk",
    "tqdm",
]:
  if mod_name not in sys.modules:
    sys.modules[mod_name] = MagicMock()

import torch


class _TestMockDataset:

  def __init__(self, data: dict[str, list[Any]]):
    self._data = dict(data)
    self.column_names = list(data.keys())

  def __getitem__(self, key: str):
    return self._data[key]

  def __len__(self):
    first_col = next(iter(self._data.values()))
    return len(first_col)

  def select(self, indices):
    new_data = {k: [self._data[k][i] for i in indices] for k in self._data}
    return _TestMockDataset(new_data)

  def shuffle(self, seed=None):
    return self


class DatasetDict(dict):

  def push_to_hub(self, repo_id, **kwargs):
    pass

  def save_to_disk(self, path, **kwargs):
    pass


class Dataset:

  @classmethod
  def from_dict(cls, d):
    return _TestMockDataset(d)


datasets_mod = types.ModuleType("datasets")
datasets_mod.Dataset = Dataset
datasets_mod.DatasetDict = DatasetDict
datasets_mod.Value = MagicMock()
datasets_mod.load_dataset = MagicMock()
datasets_mod.concatenate_datasets = MagicMock()
sys.modules["datasets"] = datasets_mod

from src.pipelines import SSFODataGenerationPipeline
from src.utils import SsfoDataGenArguments


class TestSSFODataGeneration(unittest.TestCase):
  """Test suite for SSFO synthetic preference data generation."""

  def setUp(self):
    self.pipeline = SSFODataGenerationPipeline()
    self.pipeline.args = SsfoDataGenArguments(
        task_name="npov",
        dataset_repo_id="test_user/test_sft",
        model_repo_id="google/gemma-4-E4B",
        sft_model_path="test_user/test_sft_model",
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=10,
        batch_size=2,
        seed=42,
        use_ground_truth_chosen=False,
        push_to_hub=False,
    )
    self.pipeline.device = torch.device("cpu")
    self.pipeline.enable_lora = False
    self.pipeline.lora_path = None
    self.pipeline.vllm_model = self.pipeline.args.model_repo_id

  def test_extract_prompts_npov(self):
    """Test prompt extraction with and without context for NPOV."""
    self.pipeline.args.task_name = "npov"
    entry = {
        "user_query": "Is coffee healthy?",
        "perspective_1_name": "Pro-coffee",
        "perspective_1": "Coffee has antioxidants.",
        "perspective_2_name": "Anti-coffee",
        "perspective_2": "Excess caffeine causes anxiety.",
        "npov_response": "Coffee has both antioxidants and caffeine effects.",
    }
    prompt_ctx, prompt_no_ctx, gt = self.pipeline._extract_prompts(entry)

    # Check that context prompt includes perspectives
    self.assertIn("Coffee has antioxidants.", prompt_ctx)
    self.assertIn("Excess caffeine causes anxiety.", prompt_ctx)
    self.assertIn("Is coffee healthy?", prompt_ctx)

    # Check that context-free prompt includes only query
    self.assertIn("Is coffee healthy?", prompt_no_ctx)
    self.assertNotIn("Coffee has antioxidants.", prompt_no_ctx)
    self.assertNotIn("Excess caffeine causes anxiety.", prompt_no_ctx)
    self.assertEqual(gt, "Coffee has both antioxidants and caffeine effects.")

  def test_extract_prompts_bosch(self):
    """Test prompt extraction with and without context for Bosch."""
    self.pipeline.args.task_name = "bosch"
    entry = {
        "Question": "How do I check tire pressure?",
        "Context": "Use the gauge located in the glove compartment.",
        "response": "Locate the gauge in the glove compartment to check.",
    }
    prompt_ctx, prompt_no_ctx, gt = self.pipeline._extract_prompts(entry)

    # Check that context prompt contains manual excerpt
    self.assertIn("Use the gauge located in the glove compartment.", prompt_ctx)
    self.assertIn("How do I check tire pressure?", prompt_ctx)

    # Check that context-free prompt strips the manual excerpt
    self.assertNotIn(
        "Use the gauge located in the glove compartment.", prompt_no_ctx
    )
    self.assertIn("How do I check tire pressure?", prompt_no_ctx)
    self.assertEqual(
        gt, "Locate the gauge in the glove compartment to check."
    )

  def test_setup_model_resolves_lora(self):
    """Test setup_model correctly resolves LoRA and vllm model."""
    self.pipeline.args.sft_model_path = "test_user/sft_adapter"
    self.pipeline.args.model_repo_id = "google/gemma-4-E4B-it"
    with patch.object(
        self.pipeline,
        "_resolve_lora_adapter_path",
        return_value=(True, "/resolved/lora/path"),
    ):
      self.pipeline.setup_model()
      self.assertTrue(self.pipeline.enable_lora)
      self.assertEqual(self.pipeline.lora_path, "/resolved/lora/path")
      self.assertEqual(self.pipeline.vllm_model, "google/gemma-4-E4B-it")

  def test_extract_prompts_ragtruth(self):
    """Test _extract_prompts for ragtruth task with context."""
    self.pipeline.args.task_name = "ragtruth"
    entry = {
        "user_query": "What causes earthquakes?",
        "passage": "Tectonic plates shift along faults.",
        "completion": "Tectonic plate movements cause earthquakes.",
    }
    prompt_ctx, prompt_no_ctx, gt = self.pipeline._extract_prompts(entry)
    self.assertIn("Context: Tectonic plates shift along faults.", prompt_ctx)
    self.assertIn("Question: What causes earthquakes?", prompt_ctx)
    self.assertEqual(
        prompt_no_ctx, "Question: What causes earthquakes?\nAnswer:\n"
    )
    self.assertEqual(gt, "Tectonic plate movements cause earthquakes.")

  def test_ssfo_output_dataset_repo_id_formatting(self):
    """Test default preference dataset repo ID construction contains model and timestamp."""
    self.pipeline.args.output_dataset_repo_id = None
    self.pipeline.train_split = []
    self.pipeline.raw_dataset = {"train": []}
    self.pipeline.args.push_to_hub = True

    with patch("src.pipelines.DatasetDict.push_to_hub") as mock_push:
      self.pipeline.run_and_save()
      mock_push.assert_called_once()
      pushed_repo = mock_push.call_args[0][0]
      self.assertIn("npov_ssfo_preference", pushed_repo)
      self.assertIn("gemma-4-E4B", pushed_repo)
      self.assertTrue(pushed_repo.startswith("test_user/"))

  def test_ssfo_output_dataset_repo_id_with_max_samples(self):
    """Test default preference dataset repo ID construction contains max_samples tag."""
    self.pipeline.args.output_dataset_repo_id = None
    self.pipeline.args.max_samples = 250
    self.pipeline.train_split = []
    self.pipeline.raw_dataset = {"train": []}
    self.pipeline.args.push_to_hub = True

    with patch("src.pipelines.DatasetDict.push_to_hub") as mock_push:
      self.pipeline.run_and_save()
      mock_push.assert_called_once()
      pushed_repo = mock_push.call_args[0][0]
      self.assertIn("_n250_", pushed_repo)

  def test_run_and_save_with_vllm(self):
    """Test SSFO data generation using mocked vLLM engine."""
    self.pipeline.use_vllm = True
    self.pipeline.lora_path = "mock/lora/path"
    mock_llm = MagicMock()
    mock_out_1 = MagicMock(outputs=[MagicMock(text="Chosen completion 1")])
    mock_out_2 = MagicMock(outputs=[MagicMock(text="Chosen completion 2")])
    mock_out_3 = MagicMock(outputs=[MagicMock(text="Rejected completion 1")])
    mock_out_4 = MagicMock(outputs=[MagicMock(text="Rejected completion 2")])
    mock_llm.generate.side_effect = [
        [mock_out_1, mock_out_2],
        [mock_out_3, mock_out_4],
    ]
    self.pipeline.llm = mock_llm
    self.pipeline.train_split = [
        {
            "user_query": "q1",
            "perspective_1_name": "Pro",
            "perspective_1": "arg1",
            "perspective_2_name": "Con",
            "perspective_2": "arg2",
            "npov_response": "gt1",
        },
        {
            "user_query": "q2",
            "perspective_1_name": "Pro",
            "perspective_1": "arg1",
            "perspective_2_name": "Con",
            "perspective_2": "arg2",
            "npov_response": "gt2",
        },
    ]
    self.pipeline.raw_dataset = {"train": self.pipeline.train_split}
    self.pipeline.args.push_to_hub = False
    self.pipeline.args.use_ground_truth_chosen = False

    with patch("src.pipelines.DatasetDict.save_to_disk"):
      self.pipeline.run_and_save()
      self.assertEqual(mock_llm.generate.call_count, 2)
      ctx_args = mock_llm.generate.call_args_list[0][0][0]
      no_ctx_args = mock_llm.generate.call_args_list[1][0][0]
      self.assertEqual(len(ctx_args), 2)
      self.assertEqual(len(no_ctx_args), 2)

  def test_setup_arguments_cleaning(self):
    """Test that setup_arguments correctly cleans CLI arguments with double dash."""
    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = [
          SsfoDataGenArguments(
              task_name="npov",
              dataset_repo_id="user/dataset",
              model_repo_id="google/gemma-4-E4B",
              sft_model_path="user/sft",
          )
      ]
      mock_parser_cls.return_value = mock_parser

      test_pipeline = SSFODataGenerationPipeline()
      test_pipeline.setup_arguments(
          "--",
          "--task_name",
          "npov",
          "--dataset_repo_id",
          "user/dataset",
          "--model_repo_id",
          "google/gemma-4-E4B",
          "--sft_model_path",
          "user/sft",
      )
      called_args = mock_parser.parse_args_into_dataclasses.call_args[0][0]
      self.assertNotIn("--", called_args)
      self.assertEqual(test_pipeline.args.task_name, "npov")
      self.assertEqual(test_pipeline.args.dataset_repo_id, "user/dataset")
      self.assertEqual(test_pipeline.args.model_repo_id, "google/gemma-4-E4B")


if __name__ == "__main__":
  unittest.main()
