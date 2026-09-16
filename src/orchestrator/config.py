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
#:   The last value the trial logged - what ``run.summary`` holds. Correct
#:   when the metric is noisy per step and only its *converged* level is
#:   meaningful (the PE-RL training reward).
#: ``best``
#:   The extremum over every logged step, i.e. early-stopping semantics.
#:   Correct when the deployed checkpoint is itself the best-step one, which
#:   is what ``--load_best_model_at_end`` gives us for SFT and RM.
SELECTION_STRATEGIES = ("final", "best")

#: Which checkpoint of the materialization run is published.
#:
#: ``best``  - ``--load_best_model_at_end True`` (early stopping in-run).
#: ``final`` - the last checkpoint, i.e. no in-run early stopping.
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
  #: Which checkpoint of the materialization run reaches the Hub; see
  #: CHECKPOINT_POLICIES. Keep this consistent with ``selection_strategy``:
  #: ranking trials on their peak and then publishing their last checkpoint
  #: (or the reverse) selects on one criterion and ships another.
  checkpoint_policy: str = "best"
  #: Eval/save cadence (in optimizer steps) of the materialization run. Only
  #: meaningful under ``checkpoint_policy="best"``: it decides how finely the
  #: peak can be recovered. None keeps the stage's historical cadence.
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
  base_model: str = "google/gemma-4-E2B-it"
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
        policy, or on a selection/checkpoint combination that would rank
        trials on one criterion and publish a model chosen by another.
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
      if stage_cfg.checkpoint_policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            f"Invalid checkpoint_policy '{stage_cfg.checkpoint_policy}' for "
            f"the {stage_name.upper()} stage. Must be one of "
            f"{list(CHECKPOINT_POLICIES)}."
        )
      # Ranking trials by their peak and then shipping the winner's last
      # checkpoint means the number in the report was never measured on the
      # model that was published. Catch it here rather than in the report.
      if (
          stage_cfg.selection_strategy == "best"
          and stage_cfg.checkpoint_policy == "final"
      ):
        raise ValueError(
            f"The {stage_name.upper()} stage ranks trials on their best step "
            "(selection_strategy='best') but publishes the last checkpoint "
            "(checkpoint_policy='final'). The reported metric would then "
            "belong to a checkpoint that was never pushed. Use "
            "checkpoint_policy='best', or rank on 'final' too."
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
            selection_strategy="final",
            checkpoint_policy="best",
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
