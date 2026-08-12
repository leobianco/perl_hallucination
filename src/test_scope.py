"""Unit tests for the SCOPE baseline implementation."""

import unittest
from unittest.mock import MagicMock, patch
from datasets import Dataset, DatasetDict
from src.pipelines import DPOPipeline, ScopeDataGenerationPipeline
from src.task_processors.npov_task_processor import NPOVTaskProcessor
from src.utils import ScopeDataGenArguments, ScriptArguments
import torch


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
        seed=42,
        push_to_hub=False,
    )
    self.pipeline.device = torch.device("cpu")

  def test_scope_sft_splits(self):
    """Test splitting SFT data into D1 and D2."""
    processor = NPOVTaskProcessor()
    dummy_data = DatasetDict({
        "train": Dataset.from_dict({
            "prompt": [f"prompt_{i}" for i in range(100)],
            "completion": [f"completion_{i}" for i in range(100)],
        }),
        "test": Dataset.from_dict({
            "prompt": [f"test_prompt_{i}" for i in range(20)],
            "completion": [f"test_completion_{i}" for i in range(20)],
        }),
    })

    d1, d2 = processor._make_scope_sft_splits(
        dummy_data, split_ratio=0.5, seed=123
    )
    self.assertEqual(len(d1), 50)
    self.assertEqual(len(d2), 50)
    # Ensure no overlap between D1 and D2
    d1_prompts = set(d1["prompt"])
    d2_prompts = set(d2["prompt"])
    self.assertEqual(len(d1_prompts.intersection(d2_prompts)), 0)

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
    vocab_size = 50
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
    mock_tokenizer.bos_token_id = 2
    mock_tokenizer.additional_special_tokens_ids = []
    mock_tokenizer.return_value = MagicMock(
        input_ids=torch.tensor([[2, 10, 11]], dtype=torch.long)
    )
    mock_tokenizer.decode = MagicMock(
        side_effect=lambda token_ids, **kw: " ".join(
            f"tok_{t}" for t in token_ids
        )
    )

    self.pipeline.tokenizer = mock_tokenizer

    # Mock SFT and Base models
    mock_sft = MagicMock()
    mock_base = MagicMock()

    # Step 0: return logits
    sft_logits = torch.randn(1, 1, vocab_size)
    base_logits = torch.randn(1, 1, vocab_size)

    mock_sft.return_value = MagicMock(logits=sft_logits, past_key_values=None)
    mock_base.return_value = MagicMock(logits=base_logits, past_key_values=None)

    self.pipeline.sft_model = mock_sft
    self.pipeline.base_model = mock_base
    self.pipeline.args.max_new_tokens = 5

    # Test Bernoulli sampling
    self.pipeline.args.sampling_mode = "bernoulli"
    output_bernoulli = self.pipeline._generate_unfaithful_sample("Test prompt")
    self.assertIsInstance(output_bernoulli, str)

    # Test prob mixing
    self.pipeline.args.sampling_mode = "prob_mix"
    output_prob = self.pipeline._generate_unfaithful_sample("Test prompt")
    self.assertIsInstance(output_prob, str)

    # Test logit mixing
    self.pipeline.args.sampling_mode = "logit_mix"
    output_logit = self.pipeline._generate_unfaithful_sample("Test prompt")
    self.assertIsInstance(output_logit, str)


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


if __name__ == "__main__":
  unittest.main()
