"""Size-aware launcher settings: which DeepSpeed profile, and how much batch.

Every campaign before this module ran one model family at one size, so a
single ``scripts/deepspeed_config.yaml`` (ZeRO Stage 2) and one hand-tuned
PE-RL batch geometry were enough. ``--base-model`` made the size a free
variable, and the two settings that depend on it were left behind: a 7B policy
launched with the 4B geometry and ``auto_find_batch_size=False`` OOMs in the
PE-RL stage, hours into a campaign, with no runtime escape hatch.

This module is the single place that answers "how big is this campaign, and
what does that imply". It derives everything from one number - the parameter
count the checkpoint's *name* advertises - and deliberately nothing else: no
Hub round-trip, no torch import, no GPU query. The orchestrator runs
hermetically under plain ``python3``, has to be testable without a GPU, and
has to produce the same command in a dry run as in a real one.

The number is a proxy for memory, not a fact about the architecture. It is
calibrated against what is known to work rather than computed: the project's
default ``google/gemma-4-E4B-it`` reads as 4 and runs fine under Stage 2 with
the historical geometry, so the thresholds are set above it.

Stage names match :data:`src.orchestrator.config.VALID_STAGES`.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
#: Let the size decide. The default, and what the wizard offers.
AUTO = "auto"
#: Parameters replicated per GPU, optimizer state and gradients sharded.
ZERO2 = "zero2"
#: Parameters sharded too. Buys headroom, costs an all-gather per forward.
ZERO3 = "zero3"
#: Stage 3 with parameters and optimizer state paged to host RAM.
ZERO3_OFFLOAD = "zero3_offload"

#: Profile name -> ``accelerate launch --config_file`` argument.
PROFILE_CONFIGS: Dict[str, str] = {
    ZERO2: "scripts/deepspeed_config.yaml",
    ZERO3: "scripts/deepspeed_config_zero3.yaml",
    ZERO3_OFFLOAD: "scripts/deepspeed_config_zero3_offload.yaml",
}

#: Accepted values of ``CampaignConfig.deepspeed_profile``.
VALID_PROFILES = (AUTO,) + tuple(PROFILE_CONFIGS)

#: Data-parallel processes the accelerate configs launch. Mirrors
#: ``num_processes`` in every file of :data:`PROFILE_CONFIGS`; used only to
#: check the RLOO generation-batch divisibility rule below, never to launch
#: anything. Keep the two in step if the GPU count of the box changes.
NUM_PROCESSES = 2

# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------
#: Billions of parameters at or above which a model counts as "large" and
#: gets the cheap memory savings: gradient checkpointing in every stage, and
#: the halved PE-RL micro-batch (with doubled accumulation, so the effective
#: batch is unchanged). Both are local, well-understood trades of compute for
#: memory, and neither changes the distributed execution model.
#:
#: 6 sits between the 4B models the campaigns were tuned on and the 7B ones
#: they are being retargeted at, so nothing below 6B changes behaviour.
CHECKPOINT_THRESHOLD_B = 6.0

#: Billions of parameters at or above which the PE-RL stage moves to ZeRO
#: Stage 3.
#:
#: This was 6.0, which put Mistral-7B on Stage 3 and was a mistake worth
#: recording. The arithmetic on a 2x80 GB box: a 7B policy and a 7B reward
#: model are ~29 GB of bf16 weights, and with LoRA the gradients and optimiser
#: state are a rounding error, so Stage 2 lands around 35-38 GB of 80 - some
#: 42 GB spare. Stage 3 would cut that to ~27 GB while costing 20-30% in
#: all-gather latency. Worse, the saving is partly fictional: the reward model
#: is never passed to `deepspeed.initialize`, so it is not sharded at all, and
#: `reward_fn` in src/pipelines.py all-gathers it in full for every scored
#: batch regardless. Buying 8 GB of a 42 GB surplus for a quarter of the
#: throughput is a bad trade, and Stage 3 is where this codebase's sharp edges
#: live (the SFT adapter merge, the reward gather, generation gathering).
#:
#: 12 is where two resident models stop fitting comfortably: 4 GB per billion
#: parameters for the pair, so ~48 GB of weights, leaving roughly 30 GB for
#: activations and the rollout KV cache. It is an engineering estimate, not a
#: measurement - watch `nvidia-smi` on the first campaign that crosses it.
ZERO3_THRESHOLD_B = 12.0

#: Above this, parameters do not fit even sharded across the box, so the only
#: remaining move is to page them to host RAM. Deliberately far from anything
#: this project runs today: offloading a LoRA campaign is nearly always the
#: wrong trade (see the header of scripts/deepspeed_config_zero3_offload.yaml).
OFFLOAD_THRESHOLD_B = 30.0

# --------------------------------------------------------------------------
# Size estimation
# --------------------------------------------------------------------------
#: Matches the size token in a checkpoint name: ``-7B-``, ``-4B``, ``_1b_``,
#: and Gemma's ``-E4B-`` (the leading ``E`` reads "effective").
#:
#: Anchored on a separator at both ends on purpose. Without the left anchor
#: ``Mixtral-8x7B`` would read as 7, understating it by a factor of six; it is
#: handled by :data:`MODEL_SIZE_OVERRIDES` instead. Without the right anchor
#: an arbitrary ``...v0.3b...`` suffix could match.
_SIZE_RE = re.compile(r"(?:^|[-_./])[eE]?(\d+(?:\.\d+)?)[bB](?=[-_./]|$)")

#: Checkpoints whose name does not state a usable size, keyed by lowercased
#: repo id. Values are *total* parameters in billions, which is what governs
#: resident weight memory - for a mixture of experts that is the full count,
#: not the active one, because every expert is still loaded.
MODEL_SIZE_OVERRIDES: Dict[str, float] = {
    "mistralai/mixtral-8x7b-instruct-v0.1": 46.7,
    "mistralai/mixtral-8x7b-v0.1": 46.7,
    "mistralai/mixtral-8x22b-instruct-v0.1": 141.0,
}


def estimate_parameters_b(repo_id: Optional[str]) -> Optional[float]:
  """Estimates a checkpoint's parameter count, in billions, from its name.

  Args:
    repo_id: Hugging Face repo id or local path, e.g.
      ``mistralai/Mistral-7B-Instruct-v0.3``.

  Returns:
    The advertised size in billions of parameters, or None when the name does
    not state one. None is not an error: it means "no opinion", and every
    caller treats it as "keep the historical behaviour" rather than guessing,
    because guessing large costs a slow campaign and guessing small costs an
    OOM six hours in - but silently changing what a named model did yesterday
    is worse than both.
  """
  if not repo_id:
    return None
  text = str(repo_id).strip()
  override = MODEL_SIZE_OVERRIDES.get(text.lower())
  if override is not None:
    return override
  # Last match wins: a fine-tune is named ``<user>/npov_PERL_Mistral-7B_...``,
  # so the size sits after the prefix, never before it.
  sizes = [float(m) for m in _SIZE_RE.findall(text)]
  return sizes[-1] if sizes else None


def resident_parameters_b(
    stage: str,
    policy_model: str,
    reward_model: Optional[str] = None,
) -> Optional[float]:
  """Parameters a single GPU must hold weights for, during ``stage``.

  Args:
    stage: 'sft', 'rm' or 'perl'.
    policy_model: Base model of the policy (``CampaignConfig.base_model``).
    reward_model: Base model of the reward model, when it differs.

  Returns:
    Billions of parameters, or None when no size could be estimated.

  The PE-RL stage keeps the policy and the reward model resident at the same
  time, so it is sized on the *larger* of the two rather than on the policy
  alone. It is deliberately not their sum: the sum would have flipped the
  existing 4B campaigns (4 + 4 = 8) onto a launcher they were never measured
  under, changing the behaviour of runs that are known to work. The
  two-models-at-once cost is accounted for in the thresholds themselves -
  see :data:`ZERO3_THRESHOLD_B`, which is set from the arithmetic for a
  *pair* of resident models, not a single one.
  """
  if stage == "rm":
    candidates = [reward_model or policy_model]
  elif stage == "perl":
    candidates = [policy_model, reward_model or policy_model]
  else:
    candidates = [policy_model]
  sizes = [
      size
      for size in (estimate_parameters_b(model) for model in candidates)
      if size is not None
  ]
  return max(sizes) if sizes else None


def is_large(
    stage: str,
    policy_model: str,
    reward_model: Optional[str] = None,
    threshold: float = CHECKPOINT_THRESHOLD_B,
) -> bool:
  """Whether ``stage`` needs the cheap memory-saving settings.

  This governs gradient checkpointing and the PE-RL batch geometry, which is
  why it defaults to :data:`CHECKPOINT_THRESHOLD_B` and not to
  :data:`ZERO3_THRESHOLD_B`. The two used to be one constant; they were split
  because the point at which trading compute for activation memory starts to
  pay (~6B) is far below the point at which it is worth changing the
  distributed execution model (~12B). Profile selection is
  :func:`select_profile`'s business, not this function's.

  Args:
    stage: 'sft', 'rm' or 'perl'.
    policy_model: Base model of the policy.
    reward_model: Base model of the reward model, when it differs.
    threshold: Billions of parameters at or above which the answer is True.

  Returns:
    True when the estimated resident size reaches ``threshold``. An
    unestimatable model is False - see :func:`estimate_parameters_b`.
  """
  size = resident_parameters_b(stage, policy_model, reward_model)
  return size is not None and size >= threshold


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------
def select_profile(
    stage: str,
    policy_model: str,
    reward_model: Optional[str] = None,
) -> str:
  """Chooses the DeepSpeed profile for one stage of a campaign.

  Args:
    stage: 'sft', 'rm' or 'perl'.
    policy_model: Base model of the policy.
    reward_model: Base model of the reward model, when it differs.

  Returns:
    One of :data:`ZERO2`, :data:`ZERO3`, :data:`ZERO3_OFFLOAD`.

  Stage 3 is applied to PE-RL only. SFT and RM hold exactly one model and
  train a LoRA adapter on it: at 7B that is ~14.5 GB of bf16 weights on an 80
  GB card, which Stage 2 handles with room to spare, and sharding it would
  buy nothing but all-gather latency on every forward pass.
  """
  size = resident_parameters_b(stage, policy_model, reward_model)
  if size is None:
    return ZERO2
  if size >= OFFLOAD_THRESHOLD_B:
    return ZERO3_OFFLOAD
  if stage == "perl" and size >= ZERO3_THRESHOLD_B:
    return ZERO3
  return ZERO2


def deepspeed_config_path(
    stage: str,
    policy_model: str,
    reward_model: Optional[str] = None,
    profile: str = AUTO,
) -> str:
  """Resolves the ``accelerate launch --config_file`` argument for a stage.

  Args:
    stage: 'sft', 'rm' or 'perl'.
    policy_model: Base model of the policy.
    reward_model: Base model of the reward model, when it differs.
    profile: ``auto``, a profile name, or a literal path to an accelerate
      config. A path is passed through untouched, which is the escape hatch
      for a configuration this module does not know how to name.

  Returns:
    A path suitable for ``--config_file=``.
  """
  chosen = (profile or AUTO).strip()
  if chosen == AUTO:
    chosen = select_profile(stage, policy_model, reward_model)
  if chosen in PROFILE_CONFIGS:
    return PROFILE_CONFIGS[chosen]
  return chosen


def describe(
    stage: str,
    policy_model: str,
    reward_model: Optional[str] = None,
    profile: str = AUTO,
) -> str:
  """One log line explaining the choice, for the campaign transcript.

  Args:
    stage: 'sft', 'rm' or 'perl'.
    policy_model: Base model of the policy.
    reward_model: Base model of the reward model, when it differs.
    profile: The configured profile; see :func:`deepspeed_config_path`.

  Returns:
    Human-readable summary, e.g.
    ``perl: ~7B resident -> zero3, auto-selected``.

  Square brackets are deliberately avoided: this string is routed to the
  live dashboard, whose rich-markup stripper would swallow a ``[auto]`` tag
  whole and leave a confusing double space behind.
  """
  size = resident_parameters_b(stage, policy_model, reward_model)
  size_str = f"~{size:g}B resident" if size is not None else "size unknown"
  path = deepspeed_config_path(stage, policy_model, reward_model, profile)
  name = next(
      (key for key, value in PROFILE_CONFIGS.items() if value == path), path
  )
  chosen = (profile or AUTO).strip()
  how = "auto-selected" if chosen == AUTO else "forced"
  return f"{stage}: {size_str} -> {name}, {how} ({path})"


# --------------------------------------------------------------------------
# Memory-saving training flags
# --------------------------------------------------------------------------
def memory_flags(
    stage: str,
    policy_model: str,
    reward_model: Optional[str] = None,
) -> Dict[str, str]:
  """Extra training flags a large model needs, as ``flag -> value``.

  Args:
    stage: 'sft', 'rm' or 'perl'.
    policy_model: Base model of the policy.
    reward_model: Base model of the reward model, when it differs.

  Returns:
    A mapping without the leading ``--``. Empty for a small model, so a 4B
    campaign emits byte-identical commands to the ones it emitted before this
    module existed.

  Gradient checkpointing recomputes activations in the backward pass instead
  of storing them. It is the cheapest large-model saving available here - it
  is bounded by depth rather than by batch, so it scales with exactly the
  thing that grows - and it costs roughly 20-30% of step time. Under LoRA it
  needs one extra call, ``enable_input_require_grads``; see ``_attach_lora``
  in ``src/pipelines.py`` for why (nothing upstream of the first adapter
  requires grad, so reentrant checkpointing would otherwise drop the graph).
  """
  if not is_large(stage, policy_model, reward_model):
    return {}
  return {"gradient_checkpointing": "True"}


#: Flags :func:`scale_perl_geometry` reads or rewrites. The stages scan a
#: sweep's ``command`` list for exactly these, so a flag missing here is
#: silently left at its 4B-tuned value.
PERL_GEOMETRY_KEYS = (
    "num_generations",
    "steps_per_generation",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
)


def scale_perl_geometry(
    geometry: Dict[str, str],
    policy_model: str,
    reward_model: Optional[str] = None,
) -> Dict[str, str]:
  """Halves the PE-RL per-device batch for a large policy, keeping the maths.

  Args:
    geometry: The baseline geometry, e.g.
      :data:`src.orchestrator.model_manager.PERL_BATCH_GEOMETRY`.
    policy_model: Base model of the policy.
    reward_model: Base model of the reward model, when it differs.

  Returns:
    A new dict. Unchanged for a small model, or when halving would break one
    of the invariants below.

  ``per_device_train_batch_size`` is halved and
  ``gradient_accumulation_steps`` doubled, so the effective batch - and
  therefore the optimisation problem the sweep searched - is identical; only
  the peak activation memory of a single micro-step falls. The evaluation
  batch is halved outright, having no such constraint.

  Two invariants are checked rather than assumed, because violating either
  produces a confusing runtime failure hours later:

    * TRL requires the generation batch
      (``per_device_train_batch_size * NUM_PROCESSES * steps_per_generation``)
      to be divisible by ``num_generations``.
    * ``per_device_train_batch_size`` cannot go below 1.

  A geometry that fails them is returned untouched with a warning: refusing
  to start would be worse than running the configuration that was explicitly
  written down, and the warning is what tells the operator to intervene.
  """
  scaled = dict(geometry)
  if not is_large("perl", policy_model, reward_model):
    return scaled

  def _int(key: str) -> Optional[int]:
    try:
      return int(str(geometry[key]).strip())
    except (KeyError, TypeError, ValueError):
      return None

  train_batch = _int("per_device_train_batch_size")
  accumulation = _int("gradient_accumulation_steps")
  if train_batch is None or accumulation is None:
    logger.warning(
        "PE-RL geometry is not numeric, leaving it alone: %s", geometry
    )
    return scaled
  if train_batch < 2:
    logger.warning(
        "PE-RL per_device_train_batch_size is already %d; a large policy "
        "cannot be given more headroom by halving it. Reduce "
        "--max_completion_length or --num_generations instead.",
        train_batch,
    )
    return scaled

  halved = train_batch // 2
  generations = _int("num_generations")
  steps_per_generation = _int("steps_per_generation")
  if generations and steps_per_generation:
    generation_batch = halved * NUM_PROCESSES * steps_per_generation
    if generation_batch % generations:
      logger.warning(
          "Halving per_device_train_batch_size to %d would make the "
          "generation batch (%d) indivisible by num_generations (%d); "
          "leaving the geometry alone.",
          halved,
          generation_batch,
          generations,
      )
      return scaled

  scaled["per_device_train_batch_size"] = str(halved)
  scaled["gradient_accumulation_steps"] = str(accumulation * 2)
  eval_batch = _int("per_device_eval_batch_size")
  if eval_batch and eval_batch > 1:
    scaled["per_device_eval_batch_size"] = str(eval_batch // 2)
  logger.info(
      "Large policy detected: PE-RL micro-batch %s -> %s, accumulation "
      "%s -> %s (effective batch unchanged).",
      train_batch,
      scaled["per_device_train_batch_size"],
      accumulation,
      scaled["gradient_accumulation_steps"],
  )
  return scaled
