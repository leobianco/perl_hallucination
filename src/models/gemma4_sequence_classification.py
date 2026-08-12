"""Gemma 4 sequence classification model and AutoModel registration.

Implements Gemma4ForSequenceClassification for sequence classification and reward modeling
tasks with the Gemma 4 model family, and registers it with Hugging Face's
AutoModelForSequenceClassification registry cleanly without patching the transformers library.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

try:
  from transformers import (
      AutoModel,
      AutoModelForSequenceClassification,
      PreTrainedModel,
  )
  from transformers.modeling_outputs import SequenceClassifierOutputWithPast
except ImportError:
  AutoModel = None
  AutoModelForSequenceClassification = None
  PreTrainedModel = nn.Module
  SequenceClassifierOutputWithPast = None

# Attempt to import Gemma 4 specific classes from transformers
try:
  from transformers.models.gemma4 import (
      Gemma4Config,
      Gemma4Model,
      Gemma4PreTrainedModel,
  )
except (ImportError, AttributeError):
  try:
    from transformers import Gemma4Config
  except (ImportError, AttributeError):
    Gemma4Config = None
  Gemma4Model = None
  Gemma4PreTrainedModel = PreTrainedModel


class Gemma4ForSequenceClassification(Gemma4PreTrainedModel):
  """Gemma 4 model with a sequence classification / reward modeling head.

  This model wraps a Gemma 4 transformer backbone (Gemma4Model) and adds a linear
  classification head (`score`) on top of the last hidden state of the sequence.
  It is fully compatible with Hugging Face Trainer, PEFT LoRA (`task_type="SEQ_CLS"`),
  and DeepSpeed ZeRO.
  """

  config_class = Gemma4Config
  base_model_prefix = "model"
  _keys_to_ignore_on_load_missing = [r"score\.weight"]
  _keys_to_ignore_on_load_unexpected = [r"lm_head\.weight"]

  def __init__(self, config: Any):
    super().__init__(config)
    self.num_labels = getattr(config, "num_labels", 2)
    self.config = config

    if Gemma4Model is not None:
      self.model = Gemma4Model(config)
    elif AutoModel is not None:
      self.model = AutoModel.from_config(config)
    else:
      self.model = None

    hidden_size = getattr(
        config,
        "hidden_size",
        getattr(getattr(config, "text_config", None), "hidden_size", None),
    )
    if hidden_size is None:
      hidden_size = getattr(config, "d_model", getattr(config, "dim", 2048))

    self.score = nn.Linear(hidden_size, self.num_labels, bias=False)

    # Initialize weights and apply final processing
    if hasattr(self, "post_init"):
      self.post_init()

  def get_input_embeddings(self) -> Optional[nn.Module]:
    if hasattr(self.model, "embed_tokens"):
      return self.model.embed_tokens
    elif hasattr(self.model, "get_input_embeddings"):
      return self.model.get_input_embeddings()
    return None

  def set_input_embeddings(self, value: nn.Module) -> None:
    if hasattr(self.model, "embed_tokens"):
      self.model.embed_tokens = value
    elif hasattr(self.model, "set_input_embeddings"):
      self.model.set_input_embeddings(value)

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[Union[Any, List[torch.FloatTensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      **kwargs: Any,
  ) -> Union[Tuple[torch.Tensor, ...], Any]:
    """Forward pass for sequence classification.

    Args:
        input_ids (Optional[torch.LongTensor]): Input token IDs of shape (batch_size, seq_len).
        attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).
        position_ids (Optional[torch.LongTensor]): Position IDs.
        past_key_values (Optional[Any]): Cached key-value states.
        inputs_embeds (Optional[torch.FloatTensor]): Input embeddings.
        labels (Optional[torch.LongTensor]): Target labels for loss computation.
        use_cache (Optional[bool]): Whether to use KV cache.
        output_attentions (Optional[bool]): Whether to return attentions.
        output_hidden_states (Optional[bool]): Whether to return hidden states.
        return_dict (Optional[bool]): Whether to return a ModelOutput object.
        **kwargs: Additional keyword arguments passed to base model.

    Returns:
        Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
            Classification output with logits pooled from the last token.
    """
    return_dict = (
        return_dict
        if return_dict is not None
        else getattr(self.config, "use_return_dict", True)
    )

    transformer_outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )

    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)

    if input_ids is not None:
      batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
      batch_size = inputs_embeds.shape[0]
    else:
      batch_size = hidden_states.shape[0]

    pad_token_id = getattr(self.config, "pad_token_id", None)
    if pad_token_id is None and batch_size != 1 and input_ids is not None:
      # If no pad token defined and batch size > 1, check if attention_mask is provided
      if attention_mask is not None:
        sequence_lengths = attention_mask.sum(dim=-1) - 1
        sequence_lengths = sequence_lengths.to(logits.device)
      else:
        sequence_lengths = -1
    elif pad_token_id is None:
      sequence_lengths = -1
    else:
      if input_ids is not None:
        sequence_lengths = (
            torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
        )
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(logits.device)
      elif attention_mask is not None:
        sequence_lengths = attention_mask.sum(dim=-1) - 1
        sequence_lengths = sequence_lengths.to(logits.device)
      else:
        sequence_lengths = -1

    if isinstance(sequence_lengths, int) and sequence_lengths == -1:
      pooled_logits = logits[:, -1, :]
    else:
      pooled_logits = logits[
          torch.arange(batch_size, device=logits.device), sequence_lengths
      ]

    loss = None
    if labels is not None:
      labels = labels.to(logits.device)
      problem_type = getattr(self.config, "problem_type", None)
      if problem_type is None:
        if self.num_labels == 1:
          problem_type = "regression"
        elif self.num_labels > 1 and (
            labels.dtype == torch.long or labels.dtype == torch.int
        ):
          problem_type = "single_label_classification"
        else:
          problem_type = "multi_label_classification"

      if problem_type == "regression":
        loss_fct = MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(pooled_logits, labels)
      elif problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            pooled_logits.view(-1, self.num_labels), labels.view(-1)
        )
      elif problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)

    if not return_dict or SequenceClassifierOutputWithPast is None:
      output = (pooled_logits,) + transformer_outputs[1:]
      return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=getattr(transformer_outputs, "past_key_values", None),
        hidden_states=getattr(transformer_outputs, "hidden_states", None),
        attentions=getattr(transformer_outputs, "attentions", None),
    )


def register_gemma4_for_sequence_classification(
    model_name_or_path: Optional[str] = None,
) -> bool:
  """Registers Gemma4ForSequenceClassification with AutoModelForSequenceClassification.

  Only registers for Gemma 4 configurations and never interferes with other model
  families (Gemma 2, Gemma 3, Llama, Mistral, etc.) or upstream implementations.

  Args:
      model_name_or_path (Optional[str]): Model name, repository ID, or path.
        If provided, registration only proceeds if the model name indicates
        Gemma 4. If None, registers Gemma4Config into the AutoModel mapping.

  Returns:
      bool: True if registration succeeded or was already registered, False otherwise.
  """
  if AutoModelForSequenceClassification is None:
    return False

  # If a specific model was provided, only register if it's a Gemma 4 model
  if model_name_or_path is not None:
    name_lower = str(model_name_or_path).lower()
    if "gemma-4" not in name_lower and "gemma4" not in name_lower:
      return False

  config_cls = Gemma4Config
  if config_cls is None:
    return False

  try:
    # Check if already registered in model mapping (e.g. by an upstream transformers release)
    if hasattr(AutoModelForSequenceClassification, "_model_mapping"):
      mapping = AutoModelForSequenceClassification._model_mapping
      if config_cls in mapping:
        return True

    AutoModelForSequenceClassification.register(
        config_cls, Gemma4ForSequenceClassification
    )
    return True
  except Exception:
    return False

