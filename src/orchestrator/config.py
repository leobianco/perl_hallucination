"""Configuration dataclasses for the Auto-PERL scientific campaign orchestrator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime
import math
import os
from typing import Any, Dict, List, Optional
import yaml


VALID_TASKS = [
    "npov",
    "bosch",
    "ragtruth",
    "ragtruth-qa",
    "ragtruth-summarization",
]

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
  #: Independent autorater calls per sample. The judge jitters between calls
  #: even at temperature 0, so k > 1 takes the median and reports the spread.
  #: Costs k times the API budget; 1 keeps the historical single call.
  autorater_num_samples: int = 1
  writer_num_fewshot: int = 0
  max_tokens: int = 250
  temperature: float = 0.0
  top_p: float = 1.0
  top_k: int = 0
  compute_bertscore: bool = True
  bertscore_model: str = "sentence-transformers/all-MiniLM-L6-v2"
  compute_perplexity: bool = True
  fluency_model: str = "google/gemma-4-E4B-it"
  wandb_project: str = "new_perl_eval"
  log_to_wandb: bool = True
  overwrite_scores: bool = False
  scores_checkpoint_path: Optional[str] = None


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
  stages: List[str] = field(
      default_factory=lambda: ["sft", "rm", "perl", "eval"]
  )
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
      if stage_name not in ["sft", "rm", "perl", "eval"]:
        raise ValueError(
            f"Invalid stage '{stage_name}'. Must be one of ['sft', 'rm',"
            " 'perl', 'eval']"
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
  ) -> CampaignConfig:
    """Creates a standard production-ready CampaignConfig for the given task."""
    return cls(
        task_name=task_name,
        user=user,
        wandb_entity=wandb_entity,
        seed=seed,
        dry_run=dry_run,
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
