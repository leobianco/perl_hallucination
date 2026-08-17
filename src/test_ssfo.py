"""Unit tests for the SSFO baseline implementation."""

import sys
import types
from typing import Any
import unittest
from unittest.mock import MagicMock

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

  def test_generate_simulation(self):
    """Test SSFO generation step with mocked SFT model."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.additional_special_tokens_ids = []
    inputs_dict = {
        "input_ids": MagicMock(shape=[1, 3]),
        "attention_mask": MagicMock(shape=[1, 3]),
    }
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = inputs_dict
    mock_inputs.__getitem__.side_effect = lambda k: inputs_dict[k]
    mock_inputs.get.side_effect = lambda k, d=None: inputs_dict.get(k, d)
    mock_inputs.__contains__.side_effect = lambda k: k in inputs_dict
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.batch_decode = MagicMock(
        return_value=["Generated SSFO completion"]
    )
    mock_tokenizer.decode = MagicMock(
        return_value="Generated SSFO completion"
    )
    self.pipeline.tokenizer = mock_tokenizer

    mock_sft = MagicMock()
    mock_sft.generate = MagicMock(return_value=MagicMock())
    self.pipeline.sft_model = mock_sft
    self.pipeline.args.max_new_tokens = 5

    completion = self.pipeline._generate("Test prompt")
    self.assertEqual(completion, "Generated SSFO completion")

  def test_generate_batch_simulation(self):
    """Test SSFO batch generation with mocked SFT model."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.additional_special_tokens_ids = []
    inputs_dict = {
        "input_ids": MagicMock(shape=[2, 3]),
        "attention_mask": MagicMock(shape=[2, 3]),
    }
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = inputs_dict
    mock_inputs.__getitem__.side_effect = lambda k: inputs_dict[k]
    mock_inputs.get.side_effect = lambda k, d=None: inputs_dict.get(k, d)
    mock_inputs.__contains__.side_effect = lambda k: k in inputs_dict
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.batch_decode = MagicMock(
        return_value=["Output 1", "Output 2"]
    )
    self.pipeline.tokenizer = mock_tokenizer

    mock_sft = MagicMock()
    mock_sft.generate = MagicMock(return_value=MagicMock())
    self.pipeline.sft_model = mock_sft
    self.pipeline.args.max_new_tokens = 5

    completions = self.pipeline._generate_batch(["Prompt 1", "Prompt 2"])
    self.assertEqual(len(completions), 2)
    self.assertEqual(completions[0], "Output 1")
    self.assertEqual(completions[1], "Output 2")


if __name__ == "__main__":
  unittest.main()
