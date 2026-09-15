"""Configuration dataclasses for the Auto-PERL scientific campaign orchestrator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime
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
  # Checkpoint paths (used specifically for PE-RL stage)
  sft_model_path: Optional[str] = None  # "auto" or explicit HF repo ID / local path
  reward_model_path: Optional[str] = None  # "auto" or explicit HF repo ID / local path


@dataclass
class EvalStageConfig:
  """Configuration for the final model evaluation and Gemini autorater stage."""

  enabled: bool = True
  evaluator_model: str = "gemini-2.5-flash"
  use_gemini: bool = True
  max_eval_samples: int = 1000
  eval_batch_size: int = 32
  max_workers: int = 32
  threshold: float = 0.1025
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
  base_model: str = "google/gemma-4-E2B-it"
  stages: List[str] = field(
      default_factory=lambda: ["sft", "rm", "perl", "eval"]
  )
  sft: SweepStageConfig = field(default_factory=SweepStageConfig)
  rm: SweepStageConfig = field(default_factory=SweepStageConfig)
  perl: SweepStageConfig = field(default_factory=SweepStageConfig)
  eval: EvalStageConfig = field(default_factory=EvalStageConfig)
  reporting: ReportingConfig = field(default_factory=ReportingConfig)
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
    """Validates the campaign configuration values."""
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

    return cls(
        sft=SweepStageConfig(**sft_data),
        rm=SweepStageConfig(**rm_data),
        perl=SweepStageConfig(**perl_data),
        eval=EvalStageConfig(**eval_data),
        reporting=ReportingConfig(**rep_data),
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
        seed=seed,
        dry_run=dry_run,
        sft=SweepStageConfig(
            enabled=True,
            sweep_config_path="scripts/sweep_sft.yaml",
            max_runs=sft_runs,
            timeout_minutes=240,
            metric="eval/loss",
            goal="minimize",
        ),
        rm=SweepStageConfig(
            enabled=True,
            sweep_config_path="scripts/sweep_rm.yaml",
            max_runs=rm_runs,
            timeout_minutes=240,
            metric="eval/roc_auc",
            goal="maximize",
        ),
        perl=SweepStageConfig(
            enabled=True,
            sweep_config_path="scripts/sweep_perl.yaml",
            max_runs=perl_runs,
            timeout_minutes=360,
            metric="train/rewards/reward_fn/mean",
            goal="maximize",
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
