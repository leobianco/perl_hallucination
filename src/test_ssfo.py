"""Unit tests for the SSFO baseline implementation."""

import unittest
from unittest.mock import MagicMock
from datasets import Dataset, DatasetDict
from src.pipelines import SSFODataGenerationPipeline
from src.utils import SsfoDataGenArguments
import torch


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
    vocab_size = 50
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 1
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

    mock_sft = MagicMock()
    sft_logits = torch.randn(1, 1, vocab_size)
    mock_sft.return_value = MagicMock(logits=sft_logits, past_key_values=None)
    self.pipeline.sft_model = mock_sft
    self.pipeline.args.max_new_tokens = 5

    completion = self.pipeline._generate("Test prompt")
    self.assertIsInstance(completion, str)
    self.assertTrue(len(completion) > 0)


if __name__ == "__main__":
  unittest.main()
