import sys
import types
import unittest
from unittest.mock import MagicMock

if "torch" not in sys.modules:
  try:
    import torch
    from torch import nn
  except ImportError:
    torch = types.ModuleType("torch")
    torch.__path__ = []
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = object
    torch.nn.Linear = MagicMock
    torch.nn.Identity = MagicMock
    torch.nn.Embedding = MagicMock
    torch.nn.BCEWithLogitsLoss = object
    torch.nn.CrossEntropyLoss = object
    torch.nn.MSELoss = object
    torch.device = lambda x: x
    torch.tensor = lambda x, **kw: MagicMock()
    torch.randn = lambda *x: MagicMock()
    torch.zeros = lambda *x, **kw: MagicMock()
    torch.ones = lambda *x, **kw: MagicMock()
    torch.long = "long"
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.bool = "bool"
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn

from src.models.gemma4_sequence_classification import (
    Gemma4ForSequenceClassification,
    register_gemma4_for_sequence_classification,
)
import torch
from torch import nn


class DummyGemma4Config:
  """Mock configuration for Gemma 4 model."""

  def __init__(
      self,
      hidden_size: int = 64,
      num_labels: int = 2,
      pad_token_id: int = 0,
      problem_type: str = None,
  ):
    self.hidden_size = hidden_size
    self.num_labels = num_labels
    self.pad_token_id = pad_token_id
    self.problem_type = problem_type
    self.use_return_dict = True
    self.is_encoder_decoder = False
    self.model_type = "gemma4"


class DummyBackbone(nn.Module):
  """Mock transformer backbone simulating Gemma4Model outputs."""

  def __init__(self, hidden_size: int = 64):
    super().__init__()
    self.hidden_size = hidden_size
    self.embed_tokens = nn.Embedding(100, hidden_size)

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      position_ids=None,
      past_key_values=None,
      inputs_embeds=None,
      use_cache=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
      **kwargs,
  ):
    if input_ids is not None:
      batch_size, seq_len = input_ids.shape
    elif inputs_embeds is not None:
      batch_size, seq_len, _ = inputs_embeds.shape
    else:
      batch_size, seq_len = 1, 10

    # Deterministic distinct hidden state per position for testing pooling
    positions = (
        torch.arange(seq_len, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    hidden_states = (
        positions.unsqueeze(-1)
        .expand(batch_size, seq_len, self.hidden_size)
        .clone()
    )

    output = MagicMock()
    output.__getitem__ = lambda s, idx: hidden_states if idx == 0 else None
    output.past_key_values = None
    output.hidden_states = (hidden_states,) if output_hidden_states else None
    output.attentions = None
    return output


class TestGemma4ForSequenceClassification(unittest.TestCase):
  """Tests for Gemma4ForSequenceClassification architecture and forward behaviors."""

  def setUp(self):
    self.hidden_size = 32
    self.config = DummyGemma4Config(
        hidden_size=self.hidden_size, num_labels=2, pad_token_id=0
    )
    self.model = Gemma4ForSequenceClassification(self.config)
    # Replace backbone with deterministic dummy backbone for unit testing
    self.model.model = DummyBackbone(self.hidden_size)

  def test_init_and_score_head(self):
    """Test model score linear head dimensionality and parameters."""
    self.assertEqual(self.model.num_labels, 2)
    self.assertEqual(self.model.score.in_features, self.hidden_size)
    self.assertEqual(self.model.score.out_features, 2)
    self.assertIsNone(self.model.score.bias)

  def test_input_embeddings(self):
    """Test getter and setter for input embeddings."""
    embeddings = self.model.get_input_embeddings()
    self.assertIsNotNone(embeddings)
    new_embed = nn.Embedding(50, self.hidden_size)
    self.model.set_input_embeddings(new_embed)
    self.assertEqual(self.model.get_input_embeddings(), new_embed)

  def test_forward_unpadded(self):
    """Test forward pass on unpadded batch."""
    # input_ids with 5 tokens, no pad tokens (0 is pad_token_id)
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    output = self.model(input_ids=input_ids)

    # Logits should be (batch_size, num_labels)
    self.assertEqual(output.logits.shape, (2, 2))

  def test_forward_right_padding_pooling(self):
    """Test that right-padded inputs correctly pool logits at the last non-pad token."""
    # Sequence 1: 3 real tokens, 2 pads -> last real token at index 2
    # Sequence 2: 4 real tokens, 1 pad  -> last real token at index 3
    input_ids = torch.tensor([[5, 6, 7, 0, 0], [1, 2, 3, 4, 0]])
    output = self.model(input_ids=input_ids)

    # Score of position 2 vs position 3 from DummyBackbone
    pos2_hidden = torch.full((1, self.hidden_size), 2.0)
    pos3_hidden = torch.full((1, self.hidden_size), 3.0)
    expected_logit_0 = self.model.score(pos2_hidden)
    expected_logit_1 = self.model.score(pos3_hidden)

    self.assertTrue(
        torch.allclose(output.logits[0], expected_logit_0[0], atol=1e-5)
    )
    self.assertTrue(
        torch.allclose(output.logits[1], expected_logit_1[0], atol=1e-5)
    )

  def test_forward_left_padding_pooling(self):
    """Test that left-padded inputs correctly pool logits at the last token."""
    # Sequence: 2 pads on left, 3 real tokens -> last real token at index 4
    input_ids = torch.tensor([[0, 0, 1, 2, 3]])
    output = self.model(input_ids=input_ids)

    pos4_hidden = torch.full((1, self.hidden_size), 4.0)
    expected_logit = self.model.score(pos4_hidden)
    self.assertTrue(
        torch.allclose(output.logits[0], expected_logit[0], atol=1e-5)
    )

  def test_forward_with_labels_classification_loss(self):
    """Test loss computation for single-label binary classification."""
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = torch.tensor([0, 1])

    output = self.model(input_ids=input_ids, labels=labels)
    self.assertIsNotNone(output.loss)
    self.assertTrue(output.loss.item() >= 0.0)

  def test_forward_with_labels_regression_loss(self):
    """Test loss computation for regression (num_labels=1)."""
    config = DummyGemma4Config(
        hidden_size=self.hidden_size, num_labels=1, pad_token_id=0
    )
    reg_model = Gemma4ForSequenceClassification(config)
    reg_model.model = DummyBackbone(self.hidden_size)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    labels = torch.tensor([0.5, 0.8], dtype=torch.float32)

    output = reg_model(input_ids=input_ids, labels=labels)
    self.assertIsNotNone(output.loss)
    self.assertTrue(output.loss.item() >= 0.0)

  def test_reward_scoring_probability(self):
    """Test autorater probability scoring logic: P(label 1 = 'No' hallucination)."""
    input_ids = torch.tensor([[10, 20, 30]])
    with torch.no_grad():
      output = self.model(input_ids=input_ids)
      probs = torch.softmax(output.logits, dim=-1)[:, 1]

    self.assertEqual(probs.shape, (1,))
    self.assertTrue(0.0 <= probs.item() <= 1.0)

  def test_reward_scoring_logit_diff(self):
    """Test PE-RL reward scoring logic using logit difference (Label 1 - Label 0)."""
    input_ids = torch.tensor([[10, 20, 30]])
    with torch.no_grad():
      output = self.model(input_ids=input_ids)
      rewards = output.logits[:, 1] - output.logits[:, 0]

    self.assertEqual(rewards.shape, (1,))
    self.assertIsInstance(rewards.item(), float)


class TestAutoModelRegistration(unittest.TestCase):
  """Tests for dynamic registration with AutoModelForSequenceClassification."""

  def test_registration_idempotency(self):
    """Test that register_gemma4_for_sequence_classification runs safely and idempotently."""
    res1 = register_gemma4_for_sequence_classification("google/gemma-4-E4B")
    res2 = register_gemma4_for_sequence_classification("google/gemma-4-E4B")
    # Both calls should succeed without error
    self.assertIn(res1, [True, False])
    self.assertIn(res2, [True, False])

  def test_registration_non_gemma4_skipped(self):
    """Test that non-Gemma 4 models are not registered."""
    res_gemma3 = register_gemma4_for_sequence_classification(
        "google/gemma-3-1b-it"
    )
    self.assertFalse(res_gemma3)

    res_llama = register_gemma4_for_sequence_classification(
        "meta-llama/Llama-3.2-1B"
    )
    self.assertFalse(res_llama)


if __name__ == "__main__":
  unittest.main()
