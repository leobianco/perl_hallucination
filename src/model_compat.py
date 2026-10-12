"""Model-family-agnostic compatibility helpers.

Everything in this module exists so that the SFT -> RM -> PE-RL pipeline can be
pointed at an arbitrary Hugging Face causal LM (Gemma, Qwen, Mistral, Llama,
Phi, ...) without any of them being named in the code.

The pipeline is deliberately *completion-style*, not *chat-style*: the task
processors build plain-text prompts and the policy continues them. This module
therefore does not try to impose a chat template on training. It only covers
the four places where a model family genuinely leaks through:

1. Padding. Some tokenizers ship a dedicated pad token, some do not. The
   fallback choice matters more than it looks: see
   :func:`configure_tokenizer_padding`.
2. Turn termination. ``<end_of_turn>`` is Gemma's; ChatML models use
   ``<|im_end|>``, Llama 3 uses ``<|eot_id|>``, Mistral uses ``</s>``. See
   :func:`resolve_turn_end_token`.
3. Chat wrapping, needed only by the SCOPE/SSFO "no context" ablation, which
   asks the model a bare question outside of the task template. See
   :func:`chat_wrap_user`.
4. LoRA target modules, which PEFT infers from the architecture and cannot
   infer for architectures it has not been taught. See
   :func:`safe_get_peft_model`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


#: Turn terminators used by the model families we are likely to meet, most
#: specific first. Only consulted when the tokenizer's own chat template does
#: not reveal one; see :func:`resolve_turn_end_token`.
_KNOWN_TURN_END_TOKENS: Sequence[str] = (
    "<end_of_turn>",  # Gemma 2/3/4
    "<|im_end|>",  # ChatML: Qwen 2/2.5/3, Yi, many finetunes
    "<|eot_id|>",  # Llama 3.x
    "<|end|>",  # Phi-3/4
    "<end_of_utterance>",  # SmolLM/Idefics
)

#: Terminal delimiters stripped from a completion before a new one is
#: appended. A superset of :data:`_KNOWN_TURN_END_TOKENS` because a completion
#: may also end on a plain document terminator.
_STRIPPABLE_TERMINAL_TOKENS: Sequence[str] = tuple(_KNOWN_TURN_END_TOKENS) + (
    "<eos>",
    "</s>",
    "<|endoftext|>",
    "<|end_of_text|>",
)

#: Attention/MLP projection names, in the order PEFT itself prefers them.
#: Used only as a last-resort fallback when PEFT cannot infer target modules
#: for an architecture; see :func:`infer_lora_target_modules`.
_CANDIDATE_LORA_TARGETS: Sequence[str] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query_key_value",  # fused (Falcon, GPT-NeoX)
    "c_attn",  # fused (GPT-2 style)
)


class PaddingReport:
  """What :func:`configure_tokenizer_padding` decided, and whether to worry.

  Attributes:
    pad_token: The token now used for padding.
    source: Where it came from - ``"native"``, ``"unk"``, ``"eos"`` or
      ``"added"``.
    collides_with_eos: True when the pad token *is* the EOS token. This is not
      fatal but it silently changes sequence-classification pooling, so
      callers that care (the reward model) warn on it.
  """

  def __init__(self, pad_token: Optional[str], source: str,
               collides_with_eos: bool):
    self.pad_token = pad_token
    self.source = source
    self.collides_with_eos = collides_with_eos

  def __repr__(self) -> str:
    return (
        f"PaddingReport(pad_token={self.pad_token!r}, source={self.source!r}, "
        f"collides_with_eos={self.collides_with_eos})"
    )


def configure_tokenizer_padding(
    tokenizer: Any,
    padding_side: Optional[str] = None,
) -> PaddingReport:
  """Gives ``tokenizer`` a pad token, preferring one that is not the EOS.

  Gemma ships a dedicated ``<pad>``; Qwen pads with ``<|endoftext|>`` while
  ending turns on ``<|im_end|>``; Mistral ships no pad token at all.

  The preference order is native pad -> ``unk`` -> ``eos``, and it is not
  arbitrary. Every ``*ForSequenceClassification`` head - including
  :class:`~src.models.gemma4_sequence_classification.Gemma4ForSequenceClassification`
  - pools the hidden state at the last **non-pad** position. If the pad token
  and the EOS token are the same id, that search walks back *past* the
  sequence's own terminator and pools one token early. The reward model then
  scores a truncated view of the completion, and the whole ``token_mismatch``
  investigation becomes unmeasurable because the terminal token is never the
  pooled position. Falling back to ``unk`` first avoids that wherever the
  tokenizer offers one (Mistral does: ``<unk>``, id 0).

  Args:
    tokenizer: A Hugging Face tokenizer, mutated in place.
    padding_side: ``"left"``, ``"right"`` or None to leave it alone.

  Returns:
    A :class:`PaddingReport` describing the outcome.
  """
  if padding_side is not None:
    tokenizer.padding_side = padding_side

  eos_id = getattr(tokenizer, "eos_token_id", None)

  if getattr(tokenizer, "pad_token_id", None) is not None:
    source = "native"
  else:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_id is not None and unk_id != eos_id:
      tokenizer.pad_token = unk_token
      # Some tokenizers do not derive the id from the string assignment.
      if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = unk_id
      source = "unk"
    elif eos_id is not None:
      tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
      if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = eos_id
      source = "eos"
    else:
      logger.warning(
          "Tokenizer has neither a pad, unk nor eos token; padding is left "
          "unconfigured and batched inference will probably fail."
      )
      return PaddingReport(None, "none", False)

  pad_id = getattr(tokenizer, "pad_token_id", None)
  collides = pad_id is not None and eos_id is not None and pad_id == eos_id
  return PaddingReport(
      getattr(tokenizer, "pad_token", None), source, collides
  )


def warn_on_pad_eos_collision(report: PaddingReport, context: str) -> None:
  """Emits a single loud warning when pad and EOS share an id.

  Split out from :func:`configure_tokenizer_padding` so that only the call
  sites where it actually matters - anything that pools on the last non-pad
  token - are noisy about it.

  Args:
    report: The report returned by :func:`configure_tokenizer_padding`.
    context: Human-readable description of the call site, e.g.
      ``"reward model training"``.
  """
  if not report.collides_with_eos:
    return
  logger.warning(
      "[%s] The pad token and the EOS token are the same id (%r). "
      "Sequence-classification pooling selects the last non-pad position, so "
      "it will land one token *before* the sequence terminator. Reward "
      "scores stay self-consistent (training and PE-RL scoring both do it) "
      "but the terminal token is invisible to the reward model.",
      context,
      report.pad_token,
  )


def _added_token_strings(tokenizer: Any) -> List[str]:
  """Returns the tokenizer's added/special token strings, best effort."""
  tokens: List[str] = []
  vocab = getattr(tokenizer, "added_tokens_encoder", None)
  if isinstance(vocab, dict):
    tokens.extend(str(t) for t in vocab)
  extra = getattr(tokenizer, "additional_special_tokens", None)
  if isinstance(extra, (list, tuple)):
    tokens.extend(str(t) for t in extra)
  for attr in ("eos_token", "bos_token", "pad_token", "unk_token"):
    value = getattr(tokenizer, attr, None)
    if value:
      tokens.append(str(value))
  return tokens


def resolve_turn_end_token(tokenizer: Any) -> Optional[str]:
  """Returns the token that closes an assistant turn for this tokenizer.

  Resolution order, most authoritative first:

  1. Render the tokenizer's own chat template and read the trailing token out
     of it. This is the only source that cannot be wrong.
  2. Match the tokenizer's added-token inventory against
     :data:`_KNOWN_TURN_END_TOKENS`.
  3. Fall back to ``eos_token``, which is correct for models that do not
     distinguish "end of turn" from "end of document" (Mistral's ``</s>``).

  Args:
    tokenizer: A Hugging Face tokenizer.

  Returns:
    The turn terminator string, or None if the tokenizer exposes nothing
    usable.
  """
  rendered = _render_chat_template(tokenizer, "x")
  if rendered:
    for token in _KNOWN_TURN_END_TOKENS:
      if token in rendered:
        return token
    # The template may use a terminator we have never seen. Take the last
    # angle/pipe delimited special token it emits.
    matches = re.findall(r"<\|?[^<>|]+\|?>", rendered)
    if matches:
      bos = str(getattr(tokenizer, "bos_token", "") or "")
      for candidate in reversed(matches):
        if candidate and candidate != bos:
          return candidate

  added = _added_token_strings(tokenizer)
  for token in _KNOWN_TURN_END_TOKENS:
    if token in added:
      return token

  eos = getattr(tokenizer, "eos_token", None)
  return str(eos) if eos else None


def _render_chat_template(tokenizer: Any, content: str) -> Optional[str]:
  """Renders a one-turn chat template, or returns None if unavailable."""
  if tokenizer is None:
    return None
  if not getattr(tokenizer, "chat_template", None):
    return None
  apply = getattr(tokenizer, "apply_chat_template", None)
  if not callable(apply):
    return None
  try:
    rendered = apply(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
  except Exception as e:  # pylint: disable=broad-except
    logger.debug("apply_chat_template failed, falling back to raw text: %s", e)
    return None
  # A tokenizer configured with `tokenize=True` semantics, or a test double,
  # can hand back something that is not text. Treating that as a prompt would
  # be far worse than not templating at all.
  if not isinstance(rendered, str):
    logger.debug(
        "apply_chat_template returned %s, not str; falling back to raw text.",
        type(rendered).__name__,
    )
    return None
  return rendered


def chat_wrap_user(tokenizer: Any, text: str) -> str:
  """Wraps ``text`` as a user turn plus an assistant generation prompt.

  Used by the SCOPE and SSFO "no context" ablations, which pose a bare
  question to the model outside of the task's own plain-text template and so
  need the model's native conversational framing to get a sensible answer.

  A leading BOS is removed. Gemma's, Llama's and Mistral's templates all emit
  one, but the result here is consumed as an ordinary prompt string and is
  re-tokenized downstream with ``add_special_tokens=True``; leaving the
  template's BOS in place would give the model two of them, which shifts the
  position embeddings of the entire prompt.

  Falls back to returning ``text`` unchanged when the tokenizer carries no
  chat template - which is the right answer for a base (non-instruct) model.

  Args:
    tokenizer: A Hugging Face tokenizer, or None.
    text: The user message.

  Returns:
    The wrapped prompt, or ``text`` if no chat template is available.
  """
  rendered = _render_chat_template(tokenizer, text)
  if rendered is None:
    return text
  bos = str(getattr(tokenizer, "bos_token", "") or "")
  if bos and rendered.startswith(bos):
    rendered = rendered[len(bos):]
  return rendered


def strip_terminal_tokens(text: str, tokenizer: Any = None) -> str:
  """Removes trailing turn/document terminators from ``text``.

  Args:
    text: The completion to clean.
    tokenizer: Optional tokenizer, whose own EOS and turn-end tokens are
      added to the strip list so that families we do not know about are
      still handled.

  Returns:
    ``text`` with trailing whitespace and terminal tokens removed.
  """
  if not text:
    return text

  candidates = list(_STRIPPABLE_TERMINAL_TOKENS)
  if tokenizer is not None:
    for token in (
        getattr(tokenizer, "eos_token", None),
        resolve_turn_end_token(tokenizer),
    ):
      if token and str(token) not in candidates:
        candidates.append(str(token))

  cleaned = text.rstrip()
  changed = True
  while changed:
    changed = False
    for token in candidates:
      if cleaned.endswith(token):
        cleaned = cleaned[: -len(token)].rstrip()
        changed = True
  return cleaned


def infer_lora_target_modules(model: Any) -> List[str]:
  """Finds attention projection module names by inspecting ``model``.

  Only used when PEFT cannot infer them itself, which happens whenever the
  architecture postdates the installed PEFT release. Scanning the module tree
  is architecture-independent by construction.

  Args:
    model: A ``torch.nn.Module``.

  Returns:
    The candidate projection names actually present, in a deterministic
    order. Empty if none matched.
  """
  present: List[str] = []
  try:
    names = [name for name, _ in model.named_modules()]
  except Exception:  # pylint: disable=broad-except
    return present

  leaf_names = {name.rsplit(".", 1)[-1] for name in names}
  for candidate in _CANDIDATE_LORA_TARGETS:
    if candidate in leaf_names:
      present.append(candidate)
  return present


def safe_get_peft_model(
    model: Any,
    lora_config: Any,
    get_peft_model_fn: Any,
    explicit_target_modules: Optional[Sequence[str]] = None,
) -> Any:
  """Wraps ``model`` in a PEFT adapter, inferring targets if PEFT cannot.

  ``LoraConfig(target_modules=None)`` asks PEFT to look the architecture up in
  its built-in table. That table is a static mapping, so a model family newer
  than the installed PEFT raises ``ValueError: Please specify target_modules``.
  Rather than pin a per-family table here - which would be exactly the kind of
  hardcoding this module exists to remove - we let PEFT try first and only
  fall back to inspecting the module tree.

  Args:
    model: The base model to adapt.
    lora_config: A ``peft.LoraConfig``. Mutated only on the fallback path.
    get_peft_model_fn: ``peft.get_peft_model``, injected so this module does
      not import PEFT (it must stay importable on a CPU dev box).
    explicit_target_modules: User override. When given, PEFT inference is
      skipped entirely.

  Returns:
    The PEFT-wrapped model.

  Raises:
    ValueError: If PEFT cannot infer the targets and no candidate projection
      modules could be found either.
  """
  if explicit_target_modules:
    lora_config.target_modules = list(explicit_target_modules)
    logger.info(
        "Applying LoRA to explicitly requested modules: %s",
        lora_config.target_modules,
    )
    return get_peft_model_fn(model, lora_config)

  try:
    return get_peft_model_fn(model, lora_config)
  except ValueError as e:
    if "target_modules" not in str(e):
      raise
    inferred = infer_lora_target_modules(model)
    if not inferred:
      raise ValueError(
          "PEFT could not infer LoRA target modules for "
          f"{type(model).__name__}, and no standard attention projections "
          f"({', '.join(_CANDIDATE_LORA_TARGETS)}) were found in the module "
          "tree. Pass --lora_target_modules explicitly."
      ) from e
    logger.warning(
        "PEFT could not infer LoRA target modules for %s; falling back to "
        "the projections found in the module tree: %s. Pass "
        "--lora_target_modules to pin this explicitly.",
        type(model).__name__,
        inferred,
    )
    lora_config.target_modules = inferred
    return get_peft_model_fn(model, lora_config)


def parse_target_modules(value: Optional[str]) -> Optional[List[str]]:
  """Parses a comma-separated ``--lora_target_modules`` value.

  Args:
    value: e.g. ``"q_proj,k_proj,v_proj,o_proj"``, or None/empty.

  Returns:
    The parsed list, or None to mean "let PEFT decide".
  """
  if not value:
    return None
  modules = [m.strip() for m in str(value).split(",") if m.strip()]
  return modules or None


def _single_token_id(tokenizer: Any, token: str) -> Optional[int]:
  """Returns the id of ``token`` if the vocabulary holds it verbatim."""
  convert = getattr(tokenizer, "convert_tokens_to_ids", None)
  if not callable(convert):
    return None
  try:
    token_id = convert(token)
  except Exception:  # pylint: disable=broad-except
    return None
  if token_id is None:
    return None
  if not isinstance(token_id, int):
    return None
  unk_id = getattr(tokenizer, "unk_token_id", None)
  if unk_id is not None and token_id == unk_id:
    return None
  return token_id


def resolve_yes_no_token_ids(tokenizer: Any) -> Tuple[int, int]:
  """Finds the vocabulary ids the local autorater reads its logits at.

  The judge prompt ends immediately before the answer, and the pipeline
  compares ``logits[..., yes_id]`` with ``logits[..., no_id]`` at the final
  position. Both ids therefore have to be real, distinct tokens.

  ``convert_tokens_to_ids("Yes")`` alone is not enough. SentencePiece
  vocabularies store word-initial tokens with a ``U+2581`` prefix and
  byte-level BPE vocabularies with a ``U+0120`` prefix, and asking for the
  bare string returns the **unknown-token id** rather than raising. Both
  lookups then return the same id, ``logit_no - logit_yes`` is identically
  zero, and every sample scores exactly 0.5 - an autorater that looks like it
  ran and discriminates nothing.

  Candidate prefixes are tried in order and the first one that yields two
  distinct valid ids wins, so "Yes" and "No" are always taken from the same
  spelling variant.

  Args:
    tokenizer: A Hugging Face tokenizer.

  Returns:
    ``(yes_token_id, no_token_id)``.

  Raises:
    ValueError: If no variant resolves to two distinct tokens, which means
      this tokenizer cannot be used for single-token Yes/No scoring.
  """
  # "" first: the prompt ends with a newline, after which neither
  # SentencePiece nor byte-level BPE emits a leading-space token.
  for prefix in ("", "\u2581", "\u0120", " "):
    yes_id = _single_token_id(tokenizer, f"{prefix}Yes")
    no_id = _single_token_id(tokenizer, f"{prefix}No")
    if yes_id is not None and no_id is not None and yes_id != no_id:
      if prefix:
        logger.info(
            "Autorater Yes/No tokens resolved with prefix %r (ids %d/%d).",
            prefix,
            yes_id,
            no_id,
        )
      return yes_id, no_id

  # Last resort: first token of the encoded word.
  encode = getattr(tokenizer, "encode", None)
  if callable(encode):
    try:
      yes_ids = encode("Yes", add_special_tokens=False)
      no_ids = encode("No", add_special_tokens=False)
      if yes_ids and no_ids and yes_ids[0] != no_ids[0]:
        logger.warning(
            "Autorater Yes/No tokens fell back to the first token of the "
            "encoded word (ids %d/%d). Check that the judge model really "
            "emits these as single tokens.",
            yes_ids[0],
            no_ids[0],
        )
        return yes_ids[0], no_ids[0]
    except Exception:  # pylint: disable=broad-except
      pass

  raise ValueError(
      "Could not resolve distinct single-token ids for 'Yes' and 'No' in "
      f"{type(tokenizer).__name__}. The local autorater scores by comparing "
      "the logits of those two tokens, so it cannot be used with this "
      "tokenizer; run the autorater with --use_gemini instead."
  )
