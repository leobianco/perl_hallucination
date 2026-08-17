"""Unit tests for the SCOPE baseline implementation."""

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

from src.pipelines import DPOPipeline, ScopeDataGenerationPipeline
from src.utils import ScopeDataGenArguments


class TestScopeDataGeneration(unittest.TestCase):
  """Test suite for SCOPE noisy decoding and data generation."""

  def setUp(self):
    self.pipeline = ScopeDataGenerationPipeline()
    self.pipeline.args = ScopeDataGenArguments(
        task_name="npov",
        dataset_repo_id="test_user/test_sft",
        model_repo_id="google/gemma-4-E4B",
        sft_model_path="test_user/test_sft_model",
        alpha=0.5,
        split_ratio=0.5,
        sampling_mode="bernoulli",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=10,
        batch_size=2,
        seed=42,
        push_to_hub=False,
    )
    self.pipeline.device = torch.device("cpu")

  def test_scope_sft_splits(self):
    """Test splitting SFT data into D1 and D2 in ScopeDataGenerationPipeline."""
    self.pipeline.raw_dataset = DatasetDict({
        "train": Dataset.from_dict({
            "prompt": [f"prompt_{i}" for i in range(100)],
            "completion": [f"completion_{i}" for i in range(100)],
        }),
        "test": Dataset.from_dict({
            "prompt": [f"test_prompt_{i}" for i in range(20)],
            "completion": [f"test_completion_{i}" for i in range(20)],
        }),
    })
    self.pipeline.process_data()
    self.assertEqual(len(self.pipeline.d2_split), 50)

  def test_extract_prompt_and_chosen(self):
    """Test prompt and chosen extraction from dataset entries."""
    entry_with_npov = {
        "prompt": "Test prompt",
        "npov_response": "Test chosen completion",
    }
    prompt, chosen = self.pipeline._extract_prompt_and_chosen(entry_with_npov)
    self.assertEqual(prompt, "Test prompt")
    self.assertEqual(chosen, "Test chosen completion")

    entry_with_chosen = {
        "prompt": "Another prompt",
        "chosen": "Another chosen",
    }
    prompt, chosen = self.pipeline._extract_prompt_and_chosen(entry_with_chosen)
    self.assertEqual(prompt, "Another prompt")
    self.assertEqual(chosen, "Another chosen")

  def test_noisy_decoding_simulation(self):
    """Test noisy decoding step with mocked SFT and Base models."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.bos_token_id = 2
    mock_tokenizer.additional_special_tokens_ids = []
    inputs_dict = {
        "input_ids": MagicMock(shape=[1, 3]),
        "attention_mask": MagicMock(shape=[1, 3]),
    }
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = inputs_dict
    mock_inputs.__getitem__.side_effect = lambda k: inputs_dict[k]
    mock_inputs.get.side_effect = lambda k, d=None: inputs_dict.get(k, d)
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.batch_decode = MagicMock(
        return_value=["Generated unfaithful completion"]
    )
    mock_tokenizer.decode = MagicMock(
        return_value="Generated unfaithful completion"
    )

    self.pipeline.tokenizer = mock_tokenizer

    mock_sft = MagicMock()
    mock_base = MagicMock()
    mock_sft.return_value = MagicMock(logits=MagicMock(), past_key_values=None)
    mock_base.return_value = MagicMock(logits=MagicMock(), past_key_values=None)
    self.pipeline.sft_model = mock_sft
    self.pipeline.base_model = mock_base
    self.pipeline.args.max_new_tokens = 3

    # Test Bernoulli sampling single
    self.pipeline.args.sampling_mode = "bernoulli"
    output_bernoulli = self.pipeline._generate_unfaithful_sample("Test prompt")
    self.assertIsInstance(output_bernoulli, str)

    # Test prob mixing single
    self.pipeline.args.sampling_mode = "prob_mix"
    output_prob = self.pipeline._generate_unfaithful_sample("Test prompt")
    self.assertIsInstance(output_prob, str)

    # Test logit mixing single
    self.pipeline.args.sampling_mode = "logit_mix"
    output_logit = self.pipeline._generate_unfaithful_sample("Test prompt")
    self.assertIsInstance(output_logit, str)

  def test_batched_noisy_decoding(self):
    """Test batched noisy decoding with multiple prompts."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.bos_token_id = 2
    mock_tokenizer.additional_special_tokens_ids = []
    inputs_dict = {
        "input_ids": MagicMock(shape=[2, 3]),
        "attention_mask": MagicMock(shape=[2, 3]),
    }
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = inputs_dict
    mock_inputs.__getitem__.side_effect = lambda k: inputs_dict[k]
    mock_inputs.get.side_effect = lambda k, d=None: inputs_dict.get(k, d)
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.batch_decode = MagicMock(
        return_value=["Output 1", "Output 2"]
    )
    self.pipeline.tokenizer = mock_tokenizer

    mock_sft = MagicMock()
    mock_base = MagicMock()
    mock_sft.return_value = MagicMock(logits=MagicMock(), past_key_values=None)
    mock_base.return_value = MagicMock(logits=MagicMock(), past_key_values=None)
    self.pipeline.sft_model = mock_sft
    self.pipeline.base_model = mock_base
    self.pipeline.args.max_new_tokens = 3

    prompts = ["Prompt 1", "Prompt 2"]
    batch_outputs = self.pipeline._generate_unfaithful_samples_batch(prompts)
    self.assertEqual(len(batch_outputs), 2)
    self.assertEqual(batch_outputs[0], "Output 1")
    self.assertEqual(batch_outputs[1], "Output 2")

  def test_scope_output_dataset_repo_id_formatting(self):
    """Test default preference dataset repo ID construction contains model, alpha, and timestamp."""
    self.pipeline.args.output_dataset_repo_id = None
    self.pipeline.d2_split = []
    self.pipeline.raw_dataset = {"train": []}
    self.pipeline.args.push_to_hub = True

    with patch("src.pipelines.DatasetDict.push_to_hub") as mock_push:
      self.pipeline.run_and_save()
      mock_push.assert_called_once()
      pushed_repo = mock_push.call_args[0][0]
      self.assertIn("npov_scope_preference", pushed_repo)
      self.assertIn("gemma-4-E4B", pushed_repo)
      self.assertIn("alpha_0_5", pushed_repo)
      self.assertTrue(pushed_repo.startswith("test_user/"))

  def test_scope_output_dataset_repo_id_with_max_samples(self):
    """Test default preference dataset repo ID construction contains max_samples tag."""
    self.pipeline.args.output_dataset_repo_id = None
    self.pipeline.args.max_samples = 150
    self.pipeline.d2_split = []
    self.pipeline.raw_dataset = {"train": []}
    self.pipeline.args.push_to_hub = True

    with patch("src.pipelines.DatasetDict.push_to_hub") as mock_push:
      self.pipeline.run_and_save()
      mock_push.assert_called_once()
      pushed_repo = mock_push.call_args[0][0]
      self.assertIn("_n150_", pushed_repo)

  def test_process_data_with_max_samples(self):
    """Test that process_data correctly subsamples D2 to max_samples."""
    mock_raw = {
        "train": _TestMockDataset({"prompt": [f"p{i}" for i in range(100)]})
    }
    self.pipeline.raw_dataset = mock_raw
    self.pipeline.args.split_ratio = 0.5
    self.pipeline.args.max_samples = 20
    self.pipeline.process_data()
    self.assertEqual(len(self.pipeline.d2_split), 20)


class TestDPOPipeline(unittest.TestCase):
  """Test suite for DPO pipeline setup."""

  def test_dpo_pipeline_structure(self):
    pipeline = DPOPipeline()
    self.assertTrue(hasattr(pipeline, "setup_arguments"))
    self.assertTrue(hasattr(pipeline, "load_data"))
    self.assertTrue(hasattr(pipeline, "process_data"))
    self.assertTrue(hasattr(pipeline, "setup_model"))
    self.assertTrue(hasattr(pipeline, "setup_trainer"))
    self.assertTrue(hasattr(pipeline, "run_and_save"))

  def test_dpo_setup_arguments_with_max_prompt_length(self):
    pipeline = DPOPipeline()
    mock_script_args = MagicMock()
    mock_script_args.task_name = "npov"
    mock_script_args.max_prompt_length = 384
    mock_training_args = MagicMock()
    mock_training_args.seed = 42
    mock_training_args.learning_rate = 5e-6
    mock_training_args.beta = 0.1
    mock_training_args.run_name = "test_run"

    with patch("src.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_training_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(
          "--task_name",
          "npov",
          "--dataset_repo_id",
          "test/scope_pref",
          "--model_repo_id",
          "google/gemma-4-E4B",
          "--max_length",
          "512",
          "--max_prompt_length",
          "384",
      )
      self.assertEqual(pipeline.args.task_name, "npov")
      self.assertEqual(pipeline.training_args.learning_rate, 5e-6)
      if hasattr(mock_training_args, "max_prompt_length"):
        self.assertEqual(mock_training_args.max_prompt_length, 384)


if __name__ == "__main__":
  unittest.main()
