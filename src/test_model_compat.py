"""Tests for :mod:`src.model_compat`.

These cover the four places a model family used to leak into the pipeline:
padding, turn termination, chat wrapping and LoRA target modules - plus the
autorater's Yes/No token lookup, which fails silently rather than loudly when
it gets it wrong.

The tokenizers here are hand-written doubles rather than ``MagicMock``s on
purpose: every bug this module exists to prevent is a bug about *attributes
being absent*, and a ``MagicMock`` answers every attribute.
"""

import unittest

from src import model_compat


class FakeTokenizer:
  """A tokenizer double with only the attributes it is given.

  Attributes not passed to ``__init__`` are genuinely absent, so
  ``getattr(tok, "unk_token_id", None)`` behaves the way it would for a real
  tokenizer that lacks one.
  """

  def __init__(self, vocab=None, chat_template=None, template_fn=None,
               added_tokens=None, **attrs):
    self._vocab = vocab or {}
    self._template_fn = template_fn
    if chat_template is not None:
      self.chat_template = chat_template
    if added_tokens is not None:
      self.additional_special_tokens = list(added_tokens)
    for key, value in attrs.items():
      setattr(self, key, value)

  def convert_tokens_to_ids(self, token):
    if token in self._vocab:
      return self._vocab[token]
    return getattr(self, "unk_token_id", None)

  def encode(self, text, add_special_tokens=True):
    del add_special_tokens
    return [self._vocab.get(text, getattr(self, "unk_token_id", 0))]

  def apply_chat_template(self, messages, tokenize=False,
                          add_generation_prompt=False):
    del tokenize
    if self._template_fn is None:
      raise ValueError("no template")
    return self._template_fn(messages[-1]["content"], add_generation_prompt)


def _gemma_template(content, add_generation_prompt):
  out = f"<bos><start_of_turn>user\n{content}<end_of_turn>\n"
  if add_generation_prompt:
    out += "<start_of_turn>model\n"
  return out


def _chatml_template(content, add_generation_prompt):
  out = f"<|im_start|>user\n{content}<|im_end|>\n"
  if add_generation_prompt:
    out += "<|im_start|>assistant\n"
  return out


def _mistral_template(content, add_generation_prompt):
  del add_generation_prompt
  return f"<s>[INST] {content} [/INST]"


class FakeModule:
  """Stands in for a ``torch.nn.Module`` for target-module inference."""

  def __init__(self, names):
    self._names = names

  def named_modules(self):
    return [(name, object()) for name in self._names]


class ConfigureTokenizerPaddingTest(unittest.TestCase):

  def test_native_pad_token_is_left_alone(self):
    tok = FakeTokenizer(pad_token="<pad>", pad_token_id=0,
                        eos_token="<eos>", eos_token_id=1)
    report = model_compat.configure_tokenizer_padding(tok)
    self.assertEqual(report.source, "native")
    self.assertEqual(tok.pad_token_id, 0)
    self.assertFalse(report.collides_with_eos)

  def test_unk_is_preferred_over_eos(self):
    # Mistral's shape: no pad token, but a real <unk> at id 0. Padding with
    # the EOS would move sequence-classification pooling off the terminator.
    tok = FakeTokenizer(pad_token=None, pad_token_id=None,
                        unk_token="<unk>", unk_token_id=0,
                        eos_token="</s>", eos_token_id=2)
    report = model_compat.configure_tokenizer_padding(tok)
    self.assertEqual(report.source, "unk")
    self.assertEqual(tok.pad_token, "<unk>")
    self.assertFalse(report.collides_with_eos)

  def test_eos_is_the_last_resort_and_is_flagged(self):
    tok = FakeTokenizer(pad_token=None, pad_token_id=None,
                        eos_token="<|endoftext|>", eos_token_id=7)
    report = model_compat.configure_tokenizer_padding(tok)
    self.assertEqual(report.source, "eos")
    self.assertEqual(tok.pad_token, "<|endoftext|>")
    self.assertTrue(report.collides_with_eos)

  def test_unk_equal_to_eos_is_not_used(self):
    tok = FakeTokenizer(pad_token=None, pad_token_id=None,
                        unk_token="<eos>", unk_token_id=1,
                        eos_token="<eos>", eos_token_id=1)
    report = model_compat.configure_tokenizer_padding(tok)
    self.assertEqual(report.source, "eos")
    self.assertTrue(report.collides_with_eos)

  def test_padding_side_is_applied(self):
    tok = FakeTokenizer(pad_token="<pad>", pad_token_id=0,
                        eos_token="<eos>", eos_token_id=1)
    model_compat.configure_tokenizer_padding(tok, padding_side="left")
    self.assertEqual(tok.padding_side, "left")

  def test_tokenizer_with_nothing_usable_degrades_quietly(self):
    tok = FakeTokenizer(pad_token=None, pad_token_id=None)
    report = model_compat.configure_tokenizer_padding(tok)
    self.assertEqual(report.source, "none")
    self.assertIsNone(report.pad_token)


class ResolveTurnEndTokenTest(unittest.TestCase):

  def test_gemma_template(self):
    tok = FakeTokenizer(chat_template="x", template_fn=_gemma_template,
                        eos_token="<eos>", bos_token="<bos>")
    self.assertEqual(model_compat.resolve_turn_end_token(tok),
                     "<end_of_turn>")

  def test_chatml_template(self):
    tok = FakeTokenizer(chat_template="x", template_fn=_chatml_template,
                        eos_token="<|endoftext|>")
    self.assertEqual(model_compat.resolve_turn_end_token(tok), "<|im_end|>")

  def test_mistral_template_falls_through_to_eos(self):
    # Mistral's template has no dedicated turn terminator, so the EOS is the
    # correct answer rather than a compromise.
    tok = FakeTokenizer(chat_template="x", template_fn=_mistral_template,
                        eos_token="</s>", bos_token="<s>")
    self.assertEqual(model_compat.resolve_turn_end_token(tok), "</s>")

  def test_added_tokens_are_used_when_no_template_exists(self):
    tok = FakeTokenizer(added_tokens=["<|im_end|>"], eos_token="<eos>")
    self.assertEqual(model_compat.resolve_turn_end_token(tok), "<|im_end|>")

  def test_bare_tokenizer_returns_eos(self):
    tok = FakeTokenizer(eos_token="</s>")
    self.assertEqual(model_compat.resolve_turn_end_token(tok), "</s>")

  def test_no_tokenizer_returns_none(self):
    self.assertIsNone(model_compat.resolve_turn_end_token(None))


class ChatWrapUserTest(unittest.TestCase):

  def test_leading_bos_is_dropped(self):
    tok = FakeTokenizer(chat_template="x", template_fn=_gemma_template,
                        bos_token="<bos>")
    wrapped = model_compat.chat_wrap_user(tok, "hi")
    self.assertFalse(wrapped.startswith("<bos>"))
    self.assertEqual(
        wrapped, "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"
    )

  def test_chatml_is_not_gemma_shaped(self):
    tok = FakeTokenizer(chat_template="x", template_fn=_chatml_template)
    wrapped = model_compat.chat_wrap_user(tok, "hi")
    self.assertEqual(
        wrapped, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )

  def test_base_model_without_template_gets_raw_text(self):
    tok = FakeTokenizer(eos_token="</s>")
    self.assertEqual(model_compat.chat_wrap_user(tok, "hi"), "hi")

  def test_non_string_template_output_is_rejected(self):
    tok = FakeTokenizer(chat_template="x",
                        template_fn=lambda c, g: [1, 2, 3])
    self.assertEqual(model_compat.chat_wrap_user(tok, "hi"), "hi")

  def test_template_error_falls_back_to_raw_text(self):
    tok = FakeTokenizer(chat_template="x")  # apply_chat_template raises.
    self.assertEqual(model_compat.chat_wrap_user(tok, "hi"), "hi")


class StripTerminalTokensTest(unittest.TestCase):

  def test_known_terminators_are_removed(self):
    self.assertEqual(
        model_compat.strip_terminal_tokens("answer<end_of_turn>"), "answer"
    )
    self.assertEqual(
        model_compat.strip_terminal_tokens("answer<|im_end|>"), "answer"
    )
    self.assertEqual(model_compat.strip_terminal_tokens("answer</s>"),
                     "answer")

  def test_stacked_terminators_are_all_removed(self):
    self.assertEqual(
        model_compat.strip_terminal_tokens("answer<end_of_turn>\n<eos>"),
        "answer",
    )

  def test_unknown_family_terminator_comes_from_the_tokenizer(self):
    tok = FakeTokenizer(eos_token="<|acme_stop|>")
    self.assertEqual(
        model_compat.strip_terminal_tokens("answer<|acme_stop|>", tok),
        "answer",
    )

  def test_plain_text_is_untouched_apart_from_whitespace(self):
    self.assertEqual(model_compat.strip_terminal_tokens("answer  \n"),
                     "answer")

  def test_empty_input(self):
    self.assertEqual(model_compat.strip_terminal_tokens(""), "")


class LoraTargetModuleTest(unittest.TestCase):

  def test_inference_finds_standard_projections(self):
    model = FakeModule([
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
    ])
    self.assertEqual(
        model_compat.infer_lora_target_modules(model),
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

  def test_inference_finds_fused_projections(self):
    model = FakeModule(["transformer.h.0.attn.c_attn"])
    self.assertEqual(model_compat.infer_lora_target_modules(model),
                     ["c_attn"])

  def test_inference_returns_empty_when_nothing_matches(self):
    model = FakeModule(["embedding", "norm"])
    self.assertEqual(model_compat.infer_lora_target_modules(model), [])

  def test_explicit_modules_skip_peft_inference(self):
    cfg = type("Cfg", (), {"target_modules": None})()
    calls = []

    def fake_get_peft_model(model, config):
      calls.append(config.target_modules)
      return "wrapped"

    out = model_compat.safe_get_peft_model(
        FakeModule([]), cfg, fake_get_peft_model,
        explicit_target_modules=["q_proj", "v_proj"],
    )
    self.assertEqual(out, "wrapped")
    self.assertEqual(calls, [["q_proj", "v_proj"]])

  def test_peft_is_tried_first(self):
    cfg = type("Cfg", (), {"target_modules": None})()
    out = model_compat.safe_get_peft_model(
        FakeModule([]), cfg, lambda m, c: "wrapped"
    )
    self.assertEqual(out, "wrapped")
    self.assertIsNone(cfg.target_modules)

  def test_fallback_when_peft_cannot_infer(self):
    cfg = type("Cfg", (), {"target_modules": None})()
    model = FakeModule(["m.0.self_attn.q_proj", "m.0.self_attn.v_proj"])
    attempts = []

    def fake_get_peft_model(_, config):
      attempts.append(config.target_modules)
      if config.target_modules is None:
        raise ValueError("Please specify `target_modules` in `peft_config`")
      return "wrapped"

    out = model_compat.safe_get_peft_model(model, cfg, fake_get_peft_model)
    self.assertEqual(out, "wrapped")
    self.assertEqual(cfg.target_modules, ["q_proj", "v_proj"])
    self.assertEqual(len(attempts), 2)

  def test_unrelated_value_error_is_not_swallowed(self):
    cfg = type("Cfg", (), {"target_modules": None})()

    def fake_get_peft_model(_, __):
      raise ValueError("something else entirely")

    with self.assertRaisesRegex(ValueError, "something else entirely"):
      model_compat.safe_get_peft_model(FakeModule([]), cfg,
                                       fake_get_peft_model)

  def test_actionable_error_when_nothing_can_be_inferred(self):
    cfg = type("Cfg", (), {"target_modules": None})()

    def fake_get_peft_model(_, __):
      raise ValueError("Please specify `target_modules`")

    with self.assertRaisesRegex(ValueError, "--lora_target_modules"):
      model_compat.safe_get_peft_model(FakeModule(["norm"]), cfg,
                                       fake_get_peft_model)


class ParseTargetModulesTest(unittest.TestCase):

  def test_comma_separated(self):
    self.assertEqual(
        model_compat.parse_target_modules("q_proj, k_proj ,v_proj"),
        ["q_proj", "k_proj", "v_proj"],
    )

  def test_empty_means_let_peft_decide(self):
    self.assertIsNone(model_compat.parse_target_modules(""))
    self.assertIsNone(model_compat.parse_target_modules(None))
    self.assertIsNone(model_compat.parse_target_modules(" , , "))


class ResolveYesNoTokenIdsTest(unittest.TestCase):

  def test_plain_vocabulary(self):
    tok = FakeTokenizer(vocab={"Yes": 10, "No": 11}, unk_token_id=3)
    self.assertEqual(model_compat.resolve_yes_no_token_ids(tok), (10, 11))

  def test_sentencepiece_prefix_variant(self):
    # The bare spellings resolve to the unk id, which is exactly the silent
    # failure this function exists to prevent: both lookups would return 3,
    # the logit difference would be identically zero and every sample would
    # score 0.5.
    tok = FakeTokenizer(vocab={"\u2581Yes": 20, "\u2581No": 21},
                        unk_token_id=3)
    self.assertEqual(model_compat.resolve_yes_no_token_ids(tok), (20, 21))

  def test_byte_level_bpe_prefix_variant(self):
    tok = FakeTokenizer(vocab={"\u0120Yes": 30, "\u0120No": 31},
                        unk_token_id=3)
    self.assertEqual(model_compat.resolve_yes_no_token_ids(tok), (30, 31))

  def test_prefixes_are_not_mixed_across_variants(self):
    # "Yes" exists bare but "No" only with a prefix; taking one of each would
    # compare logits from two different spellings.
    tok = FakeTokenizer(vocab={"Yes": 10, "\u2581Yes": 20, "\u2581No": 21},
                        unk_token_id=3)
    self.assertEqual(model_compat.resolve_yes_no_token_ids(tok), (20, 21))

  def test_raises_with_actionable_message_when_unresolvable(self):
    tok = FakeTokenizer(vocab={}, unk_token_id=3)
    with self.assertRaisesRegex(ValueError, "use_gemini"):
      model_compat.resolve_yes_no_token_ids(tok)


if __name__ == "__main__":
  unittest.main()
