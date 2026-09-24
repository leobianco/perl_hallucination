"""Configuration dataclasses for the Auto-PERL scientific campaign orchestrator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime
import math
import os
from typing import Any, Dict, List, Optional
import yaml

from src.orchestrator import accel
from src.orchestrator import flavors


VALID_TASKS = [
    "npov",
    "bosch",
    "ragtruth",
    "ragtruth-qa",
    "ragtruth-summarization",
]

#: Every stage a campaign may run, in the only order they can legally run in:
#: each one consumes what the previous produced. ``autorater`` is the
#: exception - it depends on nothing the campaign trains, only on the
#: human-labelled set - and leads precisely because of that, so a judge that
#: cannot separate the labels is discovered before the GPUs warm up.
VALID_STAGES: List[str] = ["autorater", "sft", "rm", "perl", "eval"]

#: Rough per-trial wall-clock estimates (minutes) for each sweep stage. These
#: feed both the wizard's ETA preview and the sweep timeout budgets, so the two
#: can never drift apart.
MINUTES_PER_TRIAL: Dict[str, float] = {"sft": 12.0, "rm": 8.0, "perl": 35.0}

#: Headroom applied on top of the estimate when sizing a sweep timeout. The
#: timeout is a safety net against a wedged agent, not a scheduling target, so
#: it must sit comfortably above the expected duration.
TIMEOUT_SAFETY_FACTOR = 1.5

#: No sweep is ever given a smaller budget than this.
MIN_TIMEOUT_MINUTES = 240


def sweep_timeout_minutes(stage: str, max_runs: int) -> int:
  """Sizes the wall-clock budget of a sweep from its trial count.

  A fixed budget silently truncates large sweeps: the agent is killed, the
  best run *so far* is promoted, and the campaign reports success with fewer
  trials than requested. Scaling with ``max_runs`` keeps the timeout doing its
  real job (catching a wedged agent) without capping the search.

  Args:
    stage: Sweep stage name ('sft', 'rm' or 'perl').
    max_runs: Number of trials the sweep is asked to run.

  Returns:
    The timeout in minutes.
  """
  per_trial = MINUTES_PER_TRIAL.get(stage, 10.0)
  estimate = max(0, int(max_runs)) * per_trial * TIMEOUT_SAFETY_FACTOR
  return int(max(MIN_TIMEOUT_MINUTES, math.ceil(estimate)))


#: How a trial's score is read out of its W&B history.
#:
#: ``final``
#:   The last value the trial logged - what ``run.summary`` holds. One point,
#:   so it is only trustworthy for a metric that is itself stable, e.g. an
#:   evaluation computed over a whole held-out set.
#: ``final_window``
#:   The mean of the trial's last ``selection_window`` logged points. This is
#:   the converged *level*, which is what a human reads off a smoothed W&B
#:   curve. Correct when the metric is noisy per step: the PE-RL training
#:   reward is a mean over a handful of sampled generations, so scoring a
#:   trial on any single step - its peak or its last - ranks configurations
#:   by which one drew a lucky batch.
#: ``best``
#:   The extremum over every logged step, i.e. early-stopping semantics.
#:   Correct when the deployed checkpoint is itself the best-step one, which
#:   is what ``--load_best_model_at_end`` gives us for SFT and RM.
SELECTION_STRATEGIES = ("final", "final_window", "best")

#: Which checkpoint of the materialization run becomes the *default* of the
#: published Hub repository, i.e. the one a bare ``from_pretrained(repo_id)``
#: returns.
#:
#: Both checkpoints are always published: the default one at the repository
#: root and the other under the ``best/`` or ``last/`` subfolder, reachable
#: as ``from_pretrained(repo_id, subfolder=...)`` or ``repo_id:subfolder``.
#: This knob therefore only chooses the default, never what is kept.
#:
#: ``best``  - ``--load_best_model_at_end True``; root holds the best step.
#: ``final`` - no in-run early stopping; root holds the last step.
CHECKPOINT_POLICIES = ("best", "final")


@dataclass
class SweepStageConfig:
  """Configuration for a W&B hyperparameter sweep stage (SFT, RM, or PE-RL)."""

  enabled: bool = True
  sweep_config_path: str = ""
  max_runs: int = 30
  timeout_minutes: int = 240
  metric: str = "eval/loss"
  goal: str = "minimize"  # "minimize" or "maximize"
  parameter_overrides: Dict[str, Any] = field(default_factory=dict)
  #: How trials are ranked against each other; see SELECTION_STRATEGIES.
  #: ``final`` is the conservative default so that a stage configured by hand
  #: keeps the historical behaviour; :meth:`CampaignConfig.create_default`
  #: opts SFT and RM into ``best``.
  selection_strategy: str = "final"
  #: Trailing logged points averaged under ``selection_strategy`` =
  #: ``final_window``. **Read by no other strategy.**
  #:
  #: None means "not applicable", which is what a stage ranking on ``best``
  #: or ``final`` serializes. It deliberately does not default to a number:
  #: a config dump showing ``selection_window: 10`` next to
  #: ``selection_strategy: best`` reads like the stage averages 10 points,
  #: when ``best`` takes a single peak and never averages anything.
  #:
  #: Where it *is* used it trades noise for lag: too small and the score is
  #: still one lucky batch, too large and it drags in the early, untrained
  #: part of a short run. Size it against the trial's step count - a PE-RL
  #: trial at ``num_train_epochs=0.2`` logs a few dozen steps, so ~10
  #: averages the last third without reaching back into the warmup.
  selection_window: Optional[int] = None

  #: Which checkpoint the published Hub repo serves by default; see
  #: CHECKPOINT_POLICIES. The other one is always published alongside it
  #: under a subfolder, so this is purely a choice of default - but it should
  #: still agree with ``selection_strategy``, otherwise the metric quoted in
  #: the report belongs to a checkpoint that is not the repo's default.
  checkpoint_policy: str = "best"
  #: Eval/save cadence (in optimizer steps) of the materialization run. It
  #: decides how finely the best checkpoint can be located, whether that
  #: checkpoint ends up at the root or in the ``best/`` subfolder. None keeps
  #: the stage's historical cadence.
  #: It should mirror the sweep YAML's ``--eval_steps``, otherwise trials are
  #: ranked at a resolution the winner's retraining cannot reproduce.
  materialization_eval_steps: Optional[int] = None
  # Checkpoint paths (used specifically for PE-RL stage)
  sft_model_path: Optional[str] = None  # "auto" or explicit HF repo ID / local path
  reward_model_path: Optional[str] = None  # "auto" or explicit HF repo ID / local path


@dataclass
class EvalStageConfig:
  """Configuration for the final model evaluation and Gemini autorater stage."""

  enabled: bool = True
  evaluator_model: str = "gemini-2.5-flash"
  use_gemini: bool = True
  # Seed used to subsample the test set and to draw few-shot examples.
  # Defaults to the value hard-coded in ``scripts/evaluator.sh`` (12345) so
  # that orchestrated runs score the *same* subset as the baselines already
  # recorded in BASELINES_*.md. The training seed (``CampaignConfig.seed``)
  # is deliberately independent.
  seed: int = 12345
  max_eval_samples: int = 1000
  eval_batch_size: int = 32
  max_workers: int = 32
  threshold: float = 0.1025
  # Per-phase wall-clock budget. Generation of 1000 completions plus Gemini
  # autorating comfortably fits in three hours; beyond that something is
  # wedged (a stalled Vertex call, a hung vLLM worker).
  timeout_minutes: int = 180

  evaluator_num_fewshot: int = 2
  #: ROC-AUC below which the ``autorater`` stage flags the judge as too weak
  #: to trust. Advisory only: it is logged and reported, and the campaign
  #: carries on. Stopping would throw away a run over a number the user may
  #: well have expected, and the hallucination rates are still computable -
  #: they just have a larger error bar than the report would otherwise
  #: suggest.
  min_autorater_auc: float = 0.85
  #: Independent autorater calls per sample. The judge jitters between calls
  #: even at temperature 0, so k > 1 takes the median and reports the spread.
  #: Costs k times the API budget; 1 keeps the historical single call.
  autorater_num_samples: int = 1
  writer_num_fewshot: int = 0
  max_tokens: int = 250
  #: Reference decoding temperature. Used as the fallback for command
  #: builders, as the single temperature when ``temperature_grid`` is empty,
  #: and as the temperature the headline number is quoted at. 0.0 (greedy) is
  #: the deterministic, zero-sampling-variance reading of a policy, which is
  #: the one number that cannot be accused of being a lucky draw.
  temperature: float = 0.0
  #: Decoding temperatures *every* evaluated policy is scored at: the SFT
  #: baseline and each PE-RL branch alike, so all checkpoints are compared on
  #: identical footing at every point.
  #:
  #: Scoring each PE-RL policy only at its own rollout temperature is a
  #: one-point estimate of a curve, taken at the point most favourable to it:
  #: RL concentrates the policy, so a sharpened policy sampled at T looks like
  #: the SFT model sampled colder. The grid exposes that. 0.0 is greedy (where
  #: temperature scaling cannot help, so a win there is a win of the mode
  #: itself); 0.7 and 1.0 are the recommended chat temperatures of Qwen and
  #: Gemma respectively; 1.25 is a deliberately hot stress regime.
  #:
  #: Costs one generation + autorating pass per (policy, temperature).
  #: An empty list falls back to the single ``temperature``, which is what
  #: every campaign run before the grid existed did.
  temperature_grid: List[float] = field(
      default_factory=lambda: [0.0, 0.7, 1.0, 1.25]
  )
  #: Also score each PE-RL branch (and the SFT baseline) at the rollout
  #: temperature its winning trial was trained with, when that temperature
  #: is not already on ``temperature_grid``.
  #:
  #: ``sweep_perl.yaml`` searches rollout temperature over the grid minus
  #: greedy, so for new campaigns this adds nothing. It exists for policies
  #: trained under an older sweep grid (0.3/0.6): a reviewer is entitled to
  #: ask for the number at the temperature a policy was optimised for, and
  #: this guarantees it is there. Additive only - it never replaces the grid.
  match_perl_rollout_temperature: bool = True
  top_p: float = 1.0
  top_k: int = 0
  #: Context window vLLM sizes its KV cache for during generation. None keeps
  #: vLLM's own default, which is whatever the checkpoint *declares* - and a
  #: declaration is not a promise that the cache fits: ``Qwen3-4B-Instruct-2507``
  #: advertises 262144 tokens, roughly 38 GB of KV cache for a single sequence,
  #: so the engine either refuses to start or serves one request at a time.
  #:
  #: Set it to something the evaluation actually needs (longest prompt plus
  #: ``max_tokens``). Too small is not a quiet truncation - vLLM rejects a
  #: prompt that does not fit - so the failure stays visible.
  max_model_len: Optional[int] = None
  compute_bertscore: bool = True
  bertscore_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  compute_perplexity: bool = True
  #: Language model used to score the fluency (conditional perplexity) of a
  #: completion. None means "the campaign's ``base_model``", which is what
  #: ``scripts/evaluator.sh`` has always done (``FLUENCY_MODEL=${BASE_MODEL}``)
  #: and the only default that stays correct when the policy changes family.
  #: Pinning a literal here made ``--base-model Qwen/...`` still pull the
  #: gated Gemma repo and report a Qwen policy's fluency under a Gemma LM.
  #: Set it explicitly only to hold the fluency reference fixed across a
  #: cross-family comparison.
  fluency_model: Optional[str] = None
  wandb_project: str = "new_perl_eval"
  log_to_wandb: bool = True
  overwrite_scores: bool = False
  scores_checkpoint_path: Optional[str] = None

  #: Grade every completion against the reward-hacking rubric (fluency,
  #: non-repetition, non-extractiveness) in addition to the hallucination
  #: judge.
  #:
  #: On by default. The failure it detects is specific to what this campaign
  #: does: RLOO against a faithfulness reward model pays a policy to stop
  #: composing and start quoting, and the hallucination judge scores a
  #: verbatim copy of the context as perfectly faithful. A campaign can
  #: therefore report a large hallucination-rate win that is entirely a
  #: degenerate policy. A guard that must be switched on is off exactly when
  #: it is needed, so it defaults on; the cost is one extra judge call per
  #: sample.
  run_reward_hacking_autorater: bool = True
  #: Judge for the rubric. None reuses ``evaluator_model``, which keeps the
  #: two judges on the same model unless there is a reason to split them.
  reward_hacking_model: Optional[str] = None
  #: Demonstration *pairs* shown to the rubric judge. Each pair is one SFT
  #: response graded top of the scale next to a deliberately degenerated
  #: version of the same response graded bottom of the scale, so the judge
  #: sees both ends anchored on identical content. 0 runs the judge zero-shot.
  reward_hacking_num_fewshot: int = 2
  #: Aggregate quality score (in [0, 1]) below which a completion counts
  #: towards ``reward_hacking_rate``.
  #:
  #: UNCALIBRATED. Unlike ``threshold``, which the autorater stage fits
  #: against labelled data, no labelled reward-hacking set exists, so 0.6 is a
  #: judgement call: on the 1-5 scale it is the midpoint between "3 across the
  #: board" and "4 across the board". Read ``reward_hacking_quality`` and the
  #: per-dimension means as the primary signal and treat the rate as a
  #: convenience summary whose absolute level means little - though its
  #: *delta* between two policies scored by the same judge is still
  #: informative.
  reward_hacking_threshold: float = 0.6


@dataclass
class RobustnessConfig:
  """Knobs controlling how hard the campaign tries to survive the night.

  The defaults assume an unattended run on a GCP VM: transient W&B/HF errors
  are retried, a crashed materialization gets one more chance, and a silent
  subprocess is reported long before the stage timeout would kill it.
  """

  #: Attempts (not retries) for W&B API and Hugging Face Hub calls.
  api_attempts: int = 4
  #: Attempts for the (expensive) retraining of a winning configuration.
  #: A permanent failure - a bad flag, a missing dataset - is not retried,
  #: see :func:`src.orchestrator.retry.is_retryable`.
  materialize_attempts: int = 2
  #: Attempts for one evaluation phase (generation or scoring).
  eval_attempts: int = 2
  #: Initial backoff; doubles per attempt up to ``max_delay_s``.
  retry_base_delay_s: float = 15.0
  max_delay_s: float = 300.0
  #: Warn when a subprocess produces no output for this long. Not fatal -
  #: some phases are legitimately quiet - but it is the only early sign of a
  #: wedged trial in an overnight run.
  stall_warning_minutes: float = 30.0
  #: Mirror every streamed line into ``logs/campaign/{campaign_id}.log``.
  log_to_file: bool = True
  log_dir: str = "logs/campaign"


@dataclass
class ReportingConfig:
  """Configuration for artifact and metrics reporting."""

  generate_markdown: bool = True
  reports_dir: str = "reports"
  publish_wandb_report: bool = True
  render_console_summary: bool = True


@dataclass
class CampaignConfig:
  """Top-level campaign configuration coordinating all experimental stages."""

  name: str = ""
  task_name: str = "npov"
  seed: int = 130104
  user: str = "leobianco"
  project: str = "new_perl"
  wandb_entity: Optional[str] = None
  base_model: str = "google/gemma-4-E4B-it"
  #: Base model for the *reward* model, when it should differ from the policy.
  #:
  #: None means "use ``base_model``", which is what almost every campaign
  #: wants and, crucially, what makes changing ``base_model`` sufficient. The
  #: reward model used to be pinned only in ``scripts/sweep_rm.yaml``, so a
  #: campaign retargeted at a new policy silently kept scoring its rollouts
  #: with a reward model built on the old one - a mismatch that produces
  #: plausible-looking reward curves and no error at all.
  #:
  #: Set it explicitly to run the deliberately-asymmetric configurations, e.g.
  #: a small reward model against a larger policy.
  reward_base_model: Optional[str] = None

  #: Which ``accelerate``/DeepSpeed launcher profile the training stages use.
  #:
  #: ``auto`` (the default) derives it from the size the base model's name
  #: advertises: see :mod:`src.orchestrator.accel`. In practice that means
  #: ZeRO Stage 2 everywhere for the 4B models this project was built on -
  #: byte-identical commands to the ones that ran before the knob existed -
  #: and Stage 3 for the PE-RL stage only once the policy reaches 6B, which
  #: is the stage and the size at which two resident models stop fitting
  #: comfortably on an 80 GB card.
  #:
  #: Set it to a profile name (``zero2``, ``zero3``, ``zero3_offload``) to
  #: force one, or to a literal path to an accelerate config file to use
  #: something this project does not have a name for.
  deepspeed_profile: str = accel.AUTO

  #: ``autorater`` comes first and is not a training stage: it measures the
  #: Gemini judge against the human-labelled set and fits the decision
  #: threshold the final evaluation then scores with. It runs before any GPU
  #: time is spent precisely because a judge that cannot separate the labels
  #: invalidates every number downstream of it, and that is worth finding
  #: out in the first ten minutes rather than the last.
  stages: List[str] = field(
      default_factory=lambda: ["autorater", "sft", "rm", "perl", "eval"]
  )
  #: Which reward-model training sets this campaign builds branches for; see
  #: :mod:`src.orchestrator.flavors`.
  #:
  #: Lives here rather than on ``rm`` because it reshapes the whole DAG: one
  #: entry per flavor means one RM sweep *and* one PE-RL sweep per flavor,
  #: with SFT shared upstream and evaluation scoring every branch in one
  #: pass. Putting it under ``rm`` would have made a campaign-topology knob
  #: look like a per-sweep detail.
  #:
  #: ``["organic"]`` reproduces the behaviour of every campaign that ran
  #: before flavors were selectable, down to the stage ids in the state file.
  rm_dataset_flavors: List[str] = field(default_factory=lambda: [flavors.ORGANIC])
  sft: SweepStageConfig = field(default_factory=SweepStageConfig)
  rm: SweepStageConfig = field(default_factory=SweepStageConfig)
  perl: SweepStageConfig = field(default_factory=SweepStageConfig)
  eval: EvalStageConfig = field(default_factory=EvalStageConfig)
  reporting: ReportingConfig = field(default_factory=ReportingConfig)
  robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
  dry_run: bool = False
  no_tui: bool = False
  #: Power the VM off once the campaign has nothing left to do, the way
  #: ``SHUTDOWN=true`` does in ``scripts/perl.sh``. Off by default: the cost
  #: of a wrong ``true`` is a machine that disappears under an interactive
  #: session, while the cost of a wrong ``false`` is only idle GPU time.
  #:
  #: It fires on ``COMPLETED`` *and* ``FAILED`` - a campaign that dies at
  #: hour two is the most expensive one to leave running - but never on a
  #: stop the user asked for, and never in a dry run. See
  #: :mod:`src.orchestrator.shutdown`.
  shutdown_when_done: bool = False
  #: Cancellable countdown before the machine actually goes down. Long
  #: enough to read the outcome and hit Ctrl-C, short enough that an
  #: unattended overnight run is not still billing at breakfast. 0 powers
  #: off immediately.
  shutdown_grace_seconds: int = 60
  state_file: Optional[str] = None

  def __post_init__(self):
    if not self.name:
      timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
      self.name = f"{self.task_name}_campaign_{timestamp}"
    if not self.state_file:
      self.state_file = f"./checkpoints/{self.task_name}/{self.name}_state.json"

  def resolved_reward_base_model(self) -> str:
    """The base model the reward model is trained on.

    Single source of truth for the RM stage and for anything that reports on
    it, so that the default ("same as the policy") cannot be re-derived
    differently in two places.

    Returns:
      ``reward_base_model`` when set, otherwise ``base_model``.
    """
    return self.reward_base_model or self.base_model

  def resolved_fluency_model(self) -> str:
    """The language model that scores completion fluency.

    Same contract as :meth:`resolved_reward_base_model`: an unset value
    follows the policy, so moving a campaign to another model family is one
    edit rather than three.

    Returns:
      ``eval.fluency_model`` when set, otherwise ``base_model``.
    """
    return self.eval.fluency_model or self.base_model

  def deepspeed_config_for(self, stage: str) -> str:
    """The ``accelerate launch --config_file`` argument for one stage.

    Same contract as :meth:`resolved_reward_base_model`: one place decides,
    so the sweep and the winner's retraining cannot end up on different
    launchers. They must agree - a trial that fitted under Stage 3 and a
    retrain that does not fit under Stage 2 is a campaign that dies after the
    expensive part is already paid for.

    Args:
      stage: 'sft', 'rm' or 'perl'.

    Returns:
      A path to an accelerate config file.
    """
    return accel.deepspeed_config_path(
        stage,
        self.base_model,
        self.reward_base_model,
        profile=self.deepspeed_profile,
    )

  def memory_flags_for(self, stage: str) -> Dict[str, str]:
    """Extra training flags a large base model needs, as ``flag -> value``.

    Args:
      stage: 'sft', 'rm' or 'perl'.

    Returns:
      A mapping without the leading ``--``; empty below
      :data:`src.orchestrator.accel.CHECKPOINT_THRESHOLD_B`.
    """
    return accel.memory_flags(stage, self.base_model, self.reward_base_model)

  def describe_launcher(self, stage: str) -> str:
    """One line naming the launcher choice and why, for the campaign log.

    Args:
      stage: 'sft', 'rm' or 'perl'.

    Returns:
      Human-readable summary; see :func:`src.orchestrator.accel.describe`.
    """
    return accel.describe(
        stage,
        self.base_model,
        self.reward_base_model,
        profile=self.deepspeed_profile,
    )

  def validate(self) -> None:
    """Validates the campaign configuration values.

    Raises:
      ValueError: On an unknown task, stage, selection strategy or checkpoint
        policy.
    """
    if self.task_name not in VALID_TASKS:
      raise ValueError(
          f"Invalid task_name '{self.task_name}'. Must be one of {VALID_TASKS}"
      )
    for stage_name in self.stages:
      if stage_name not in VALID_STAGES:
        raise ValueError(
            f"Invalid stage '{stage_name}'. Must be one of "
            f"{list(VALID_STAGES)}"
        )

    # A profile name is checked, a path is not: the path is only read by
    # `accelerate launch`, minutes later, and a typo there is reported
    # clearly. A typo in a *name* would instead fall through
    # `deepspeed_config_path` as if it were a path, and the campaign would
    # die on "config file not found" with no hint that 'zero_3' was meant to
    # be 'zero3'.
    profile = (self.deepspeed_profile or "").strip()
    if not profile:
      raise ValueError(
          "deepspeed_profile must not be empty; use 'auto' to let the base "
          f"model's size decide, or one of {list(accel.PROFILE_CONFIGS)}."
      )
    looks_like_path = "/" in profile or profile.endswith((".yaml", ".yml"))
    if profile not in accel.VALID_PROFILES and not looks_like_path:
      raise ValueError(
          f"Unknown deepspeed_profile '{profile}'. Must be one of "
          f"{list(accel.VALID_PROFILES)}, or a path to an accelerate "
          "config file."
      )

    # Caught here rather than at dataset-load time: a typo'd flavor would
    # otherwise surface as a 404 from the Hub an hour into the campaign,
    # after SFT had already been paid for.
    selected_flavors = flavors.normalize_flavors(self.rm_dataset_flavors)
    if not selected_flavors:
      raise ValueError(
          "rm_dataset_flavors must name at least one reward-model dataset; "
          f"valid values are {list(flavors.RM_DATASET_FLAVORS)}."
      )
    unknown = [f for f in selected_flavors if f not in flavors.RM_DATASET_FLAVORS]
    if unknown:
      raise ValueError(
          f"Unknown rm_dataset_flavors {unknown}. Must be chosen from "
          f"{list(flavors.RM_DATASET_FLAVORS)}. Note that 'synthetic_llm' is "
          "deliberately not selectable: its training split is assembled from "
          "two other datasets and needs the num_*_hallus_to_keep knobs."
      )
    # A fan-out is defined by its reward models. Without an RM sweep every
    # branch would fall back to the single `perl.reward_model_path`
    # override, so the campaign would train N identical policies and report
    # them as a comparison between datasets.
    if len(selected_flavors) > 1 and "rm" not in self.stages:
      raise ValueError(
          f"rm_dataset_flavors names {len(selected_flavors)} datasets "
          f"({', '.join(selected_flavors)}) but the campaign does not run "
          "the 'rm' stage, so there is nothing to train on them. Add 'rm' "
          "to stages, or select a single dataset and point "
          "perl.reward_model_path at an existing reward model."
      )
    for stage_name in ("sft", "rm", "perl"):
      stage_cfg: SweepStageConfig = getattr(self, stage_name)
      if stage_cfg.selection_strategy not in SELECTION_STRATEGIES:
        raise ValueError(
            f"Invalid selection_strategy "
            f"'{stage_cfg.selection_strategy}' for the {stage_name.upper()} "
            f"stage. Must be one of {list(SELECTION_STRATEGIES)}."
        )
      if stage_cfg.selection_strategy == "final_window" and not (
          isinstance(stage_cfg.selection_window, int)
          and stage_cfg.selection_window >= 1
      ):
        # Caught here rather than defaulted at query time: a window of 0 or
        # None is not a smaller window, it is a different strategy, and a
        # campaign that silently ranked on one point while its config said
        # otherwise would be unreproducible from the config alone.
        raise ValueError(
            f"selection_window must be an int >= 1 for the "
            f"{stage_name.upper()} stage when selection_strategy is "
            f"'final_window', got {stage_cfg.selection_window!r}. Use "
            "selection_strategy='final' to rank on a single point."
        )

      if stage_cfg.checkpoint_policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            f"Invalid checkpoint_policy '{stage_cfg.checkpoint_policy}' for "
            f"the {stage_name.upper()} stage. Must be one of "
            f"{list(CHECKPOINT_POLICIES)}."
        )
      # No combination of the two is rejected any more: the materialization
      # run publishes both the best and the final checkpoint (one at the
      # repository root, the other under a named subfolder), so ranking on
      # one criterion can no longer leave the corresponding model unpushed.

    if self.shutdown_when_done and self.shutdown_grace_seconds < 0:
      # A negative countdown is almost certainly a typo for "no countdown",
      # but guessing which would silently remove the only chance the user
      # has to cancel a shutdown they did not mean to arm.
      raise ValueError(
          "shutdown_grace_seconds must be >= 0, got "
          f"{self.shutdown_grace_seconds}. Use 0 to power off immediately."
      )

    # Checked here rather than at generation time: a bad grid entry would
    # otherwise surface as a vLLM error hours in, after SFT and PE-RL had
    # already been paid for.
    for value in list(self.eval.temperature_grid or []) + [
        self.eval.temperature
    ]:
      if (
          isinstance(value, bool)
          or not isinstance(value, (int, float))
          or not math.isfinite(float(value))
          or float(value) < 0.0
      ):
        raise ValueError(
            "eval.temperature and every eval.temperature_grid entry must be "
            f"a finite number >= 0, got {value!r}."
        )

  def to_dict(self) -> Dict[str, Any]:
    """Converts the config dataclass to a nested dictionary."""
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> CampaignConfig:
    """Creates a CampaignConfig from a dictionary with nested dataclass handling."""
    data = dict(data)

    # Support top-level 'campaign' block as documented in implementation plan
    if "campaign" in data and isinstance(data["campaign"], dict):
      campaign_block = data.pop("campaign")
      for k, v in campaign_block.items():
        if k == "task":
          data.setdefault("task_name", v)
        else:
          data.setdefault(k, v)

    if "task" in data and "task_name" not in data:
      data["task_name"] = data.pop("task")

    # Support both '<stage>' and '<stage>_stage' key notations
    sft_data = data.pop("sft_stage", None) or data.pop("sft", {})
    rm_data = data.pop("rm_stage", None) or data.pop("rm", {})
    perl_data = data.pop("perl_stage", None) or data.pop("perl", {})
    eval_data = data.pop("eval_stage", None) or data.pop("eval", {})
    rep_data = data.pop("reporting_stage", None) or data.pop("reporting", {})
    rob_data = data.pop("robustness", {})

    return cls(
        sft=SweepStageConfig(**sft_data),
        rm=SweepStageConfig(**rm_data),
        perl=SweepStageConfig(**perl_data),
        eval=EvalStageConfig(**eval_data),
        reporting=ReportingConfig(**rep_data),
        robustness=RobustnessConfig(**rob_data),
        **data,
    )

  def to_yaml(self, path: str) -> None:
    """Serializes the configuration to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

  @classmethod
  def from_yaml(cls, path: str) -> CampaignConfig:
    """Loads a CampaignConfig from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f)
    return cls.from_dict(data)

  @classmethod
  def create_default(
      cls,
      task_name: str = "npov",
      user: str = "leobianco",
      wandb_entity: Optional[str] = None,
      seed: int = 130104,
      sft_runs: int = 30,
      rm_runs: int = 30,
      perl_runs: int = 10,
      dry_run: bool = False,
      rm_dataset_flavors: Optional[List[str]] = None,
  ) -> CampaignConfig:
    """Creates a standard production-ready CampaignConfig for the given task."""
    return cls(
        task_name=task_name,
        user=user,
        wandb_entity=wandb_entity,
        seed=seed,
        dry_run=dry_run,
        rm_dataset_flavors=flavors.normalize_flavors(rm_dataset_flavors)
        or [flavors.ORGANIC],
        sft=SweepStageConfig(
            enabled=True,
            sweep_config_path="scripts/sweep_sft.yaml",
            max_runs=sft_runs,
            timeout_minutes=sweep_timeout_minutes("sft", sft_runs),
            metric="eval/loss",
            goal="minimize",
            # A LoRA SFT run on a small dataset routinely bottoms out early
            # and then climbs back as it memorises. Its *last* eval loss
            # therefore says more about how long the run was than about how
            # good its configuration is - and the checkpoint we publish is
            # the best-step one anyway (--load_best_model_at_end).
            selection_strategy="best",
            checkpoint_policy="best",
            # Mirrors --eval_steps in scripts/sweep_sft.yaml. Without this
            # the retraining only evaluated per epoch, so the winner's peak
            # (found among every-10-step evals) was unreachable.
            materialization_eval_steps=10,
        ),
        rm=SweepStageConfig(
            enabled=True,
            sweep_config_path="scripts/sweep_rm.yaml",
            max_runs=rm_runs,
            timeout_minutes=sweep_timeout_minutes("rm", rm_runs),
            metric="eval/roc_auc",
            goal="maximize",
            # Same reasoning as SFT; up to 15 epochs makes late overfitting
            # the rule rather than the exception here.
            selection_strategy="best",
            checkpoint_policy="best",
            # Mirrors --eval_steps in scripts/sweep_rm.yaml.
            materialization_eval_steps=50,
        ),
        perl=SweepStageConfig(
            enabled=True,
            sweep_config_path="scripts/sweep_perl.yaml",
            max_runs=perl_runs,
            timeout_minutes=sweep_timeout_minutes("perl", perl_runs),
            metric="train/rewards/reward_fn/mean",
            goal="maximize",
            # Deliberately NOT "best". The PE-RL objective is a *training*
            # reward logged every step over 8 generations; its peak almost
            # always lands early, before the policy stabilises, so ranking on
            # it would select the luckiest batch rather than the best
            # configuration. The converged level is the honest signal.
            #
            # But plain "final" is only one step of that same noisy signal:
            # it does not measure the converged level, it measures wherever
            # the trial happened to stop. Two configurations then swap places
            # on a single batch of 8 sampled generations, which is how a
            # winner that maximises nothing visible gets published. The
            # window averages the tail, which is the level a human reads off
            # the smoothed curve in the W&B UI.
            selection_strategy="final_window",
            # ~10 of the few dozen steps a 0.2-epoch trial logs: the last
            # third, without reaching back into the warmup.
            selection_window=10,

            # The last checkpoint is the one a bare
            # ``from_pretrained(repo_id)`` returns, for the same reason: a
            # peak-reward adapter is one lucky batch away from being a
            # collapsed policy, and the evaluation stage should default to
            # the stable model. The best-reward checkpoint is still published
            # under the "best/" subfolder if you want to compare them.
            checkpoint_policy="final",
            # Mirrors --eval_steps in scripts/sweep_perl.yaml, so the "best/"
            # companion is located at the same resolution the sweep used.
            materialization_eval_steps=50,
            sft_model_path="auto",
            reward_model_path="auto",
        ),
        eval=EvalStageConfig(
            enabled=True,
            evaluator_model="gemini-2.5-flash",
            max_eval_samples=1000,
        ),
        reporting=ReportingConfig(
            generate_markdown=True,
            reports_dir="reports",
            publish_wandb_report=True,
        ),
    )
