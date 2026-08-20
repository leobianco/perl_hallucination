"""Unit tests for the Token Mismatch experimental module."""

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
    torch.long = "long"
    torch.bool = "bool"
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
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
  transformers_mod.LogitsProcessor = object
  transformers_mod.LogitsProcessorList = list
  sys.modules["transformers"] = transformers_mod
  sys.modules["transformers.trainer_utils"] = types.ModuleType(
      "transformers.trainer_utils"
  )
  sys.modules["transformers.trainer_utils"].get_last_checkpoint = MagicMock(
      return_value=None
  )

for mod_name in [
    "datasets",
    "trl",
    "peft",
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
    "vllm",
    "vllm.lora",
    "vllm.lora.request",
    "transformers.configuration_utils",
    "transformers.integrations",
    "transformers.integrations.heterogeneity",
    "transformers.integrations.heterogeneity.configuration_utils",
]:
  if mod_name not in sys.modules:
    sys.modules[mod_name] = MagicMock()

from src.token_mismatch.pipelines import (
    TokenMismatchPERLPipeline,
    TokenMismatchRewardModelPipeline,
    apply_terminal_token_formatting,
    resolve_terminal_token_string,
)


class TestTokenMismatchFormatting(unittest.TestCase):
  """Tests for terminal token resolution and string formatting helpers."""

  def test_resolve_terminal_token_string(self):
    mock_tok = MagicMock()
    mock_tok.eos_token = "<eos>"

    self.assertEqual(resolve_terminal_token_string("eos", mock_tok), "<eos>")
    self.assertEqual(resolve_terminal_token_string("<eos>", mock_tok), "<eos>")
    self.assertEqual(
        resolve_terminal_token_string("end_of_turn", mock_tok), "<end_of_turn>"
    )
    self.assertEqual(
        resolve_terminal_token_string("<end_of_turn>", mock_tok), "<end_of_turn>"
    )
    self.assertEqual(
        resolve_terminal_token_string("eot", mock_tok), "<end_of_turn>"
    )
    self.assertIsNone(resolve_terminal_token_string("none", mock_tok))
    self.assertIsNone(resolve_terminal_token_string("as_generated", mock_tok))
    self.assertIsNone(resolve_terminal_token_string("raw", mock_tok))
    self.assertIsNone(resolve_terminal_token_string(None, mock_tok))
    self.assertEqual(
        resolve_terminal_token_string("<custom_stop>", mock_tok), "<custom_stop>"
    )

  def test_apply_terminal_token_formatting_stripping_and_appending(self):
    # Test stripping <end_of_turn> and appending <eos>
    text_with_eot = "This is a completion.<end_of_turn>"
    formatted = apply_terminal_token_formatting(
        text_with_eot, target_token="<eos>", strip_existing_terminal_tokens=True
    )
    self.assertEqual(formatted, "This is a completion.<eos>")

    # Test stripping <eos> and appending <end_of_turn>
    text_with_eos = "This is a completion.<eos>"
    formatted = apply_terminal_token_formatting(
        text_with_eos, target_token="<end_of_turn>", strip_existing_terminal_tokens=True
    )
    self.assertEqual(formatted, "This is a completion.<end_of_turn>")

    # Test stripping without target token (raw text)
    formatted = apply_terminal_token_formatting(
        text_with_eot, target_token=None, strip_existing_terminal_tokens=True
    )
    self.assertEqual(formatted, "This is a completion.")

    # Test raw text without existing delimiter
    raw_text = "Neutral answer here."
    formatted = apply_terminal_token_formatting(
        raw_text, target_token="<end_of_turn>", strip_existing_terminal_tokens=True
    )
    self.assertEqual(formatted, "Neutral answer here.<end_of_turn>")

  def test_apply_terminal_token_formatting_edge_cases(self):
    self.assertEqual(apply_terminal_token_formatting("", "<eos>"), "<eos>")
    self.assertEqual(apply_terminal_token_formatting(None, "<eos>"), "")
    self.assertEqual(
        apply_terminal_token_formatting("Completion   \n", "<eos>"),
        "Completion<eos>",
    )


class TestTokenMismatchPipelines(unittest.TestCase):
  """Tests for pipeline initialization, argument parsing, and reward scoring logic."""

  def test_reward_model_pipeline_setup_arguments(self):
    pipeline = TokenMismatchRewardModelPipeline()
    cli_args = [
        "--task_name",
        "npov",
        "--dataset_repo_id",
        "dummy/dataset",
        "--model_repo_id",
        "google/gemma-3-1b-it",
        "--rm_terminal_token",
        "end_of_turn",
        "--output_dir",
        "/tmp/test_rm",
    ]
    with patch("src.token_mismatch.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_script_args = MagicMock(task_name="npov", model_repo_id="google/gemma-3-1b-it")
      mock_llm_synth = MagicMock()
      mock_train_args = MagicMock(learning_rate=1e-3, seed=42, run_name=None, num_train_epochs=3)
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_llm_synth,
          mock_train_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(*cli_args)
      self.assertEqual(pipeline.rm_terminal_token, "end_of_turn")
      self.assertIn("tok_end_of_turn", pipeline.training_args.run_name)

  def test_perl_pipeline_setup_arguments(self):
    pipeline = TokenMismatchPERLPipeline()
    cli_args = [
        "--task_name",
        "npov",
        "--dataset_repo_id",
        "dummy/dataset",
        "--model_repo_id",
        "google/gemma-4-E2B-it",
        "--scoring_terminal_token",
        "eos",
        "--force_terminal_token_swap",
        "true",
        "--output_dir",
        "/tmp/test_perl",
    ]
    with patch("src.token_mismatch.pipelines.HfArgumentParser") as mock_parser_cls:
      mock_parser = MagicMock()
      mock_script_args = MagicMock(task_name="npov", model_repo_id="google/gemma-4-E2B-it")
      mock_train_args = MagicMock(
          learning_rate=2e-5,
          seed=42,
          run_name=None,
          beta=1e-4,
          temperature=0.1,
          num_train_epochs=1,
      )
      mock_parser.parse_args_into_dataclasses.return_value = (
          mock_script_args,
          mock_train_args,
      )
      mock_parser_cls.return_value = mock_parser

      pipeline.setup_arguments(*cli_args)
      self.assertEqual(pipeline.scoring_terminal_token, "eos")
      self.assertTrue(pipeline.force_terminal_token_swap)
      self.assertIn("score_tok_eos", pipeline.training_args.run_name)

  def test_perl_reward_function_token_formatting(self):
    """Test that reward_fn in TokenMismatchPERLPipeline formats prompt+completion correctly."""
    pipeline = TokenMismatchPERLPipeline()
    pipeline.scoring_terminal_token = "eos"
    pipeline.force_terminal_token_swap = True

    captured_inputs = []

    def mock_tokenize(texts, **kwargs):
      captured_inputs.extend(texts)
      m = MagicMock()
      m.to.return_value = {"input_ids": MagicMock()}
      return m

    mock_tokenizer = MagicMock(side_effect=mock_tokenize)
    mock_tokenizer.eos_token = "<eos>"
    pipeline.reward_tokenizer = mock_tokenizer

    prompts = ["User query: foo\nAnswer:\n", "User query: bar\nAnswer:\n"]
    completions = ["Response A<end_of_turn>", "Response B<end_of_turn>"]

    resolved_tok = resolve_terminal_token_string(
        pipeline.scoring_terminal_token, mock_tokenizer
    )

    def test_reward_fn(prompts, completions):
      formatted_texts = []
      for p, c in zip(prompts, completions):
        formatted_c = apply_terminal_token_formatting(
            c,
            target_token=resolved_tok,
            strip_existing_terminal_tokens=pipeline.force_terminal_token_swap,
        )
        formatted_texts.append(p + formatted_c)
      mock_tokenizer(formatted_texts)

    test_reward_fn(prompts, completions)

    self.assertEqual(len(captured_inputs), 2)
    self.assertEqual(
        captured_inputs[0], "User query: foo\nAnswer:\nResponse A<eos>"
    )
    self.assertEqual(
        captured_inputs[1], "User query: bar\nAnswer:\nResponse B<eos>"
    )


if __name__ == "__main__":
  unittest.main()
