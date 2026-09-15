"""State management and atomic persistence for the Auto-PERL campaign orchestrator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime
from enum import Enum
import json
import os
import tempfile
from typing import Any, Dict, List, Optional


class StageStatus(str, Enum):
  """Lifecycle status of a campaign stage."""

  PENDING = "PENDING"
  RUNNING = "RUNNING"
  COMPLETED = "COMPLETED"
  FAILED = "FAILED"
  SKIPPED = "SKIPPED"


@dataclass
class StageResult:
  """Structured outcome and artifacts of an executed stage."""

  status: StageStatus = StageStatus.PENDING
  sweep_id: Optional[str] = None
  best_run_id: Optional[str] = None
  best_metric_val: Optional[float] = None
  best_params: Dict[str, Any] = field(default_factory=dict)
  model_repo_id: Optional[str] = None
  metrics: Dict[str, Any] = field(default_factory=dict)
  error_message: Optional[str] = None
  start_time: Optional[str] = None
  end_time: Optional[str] = None

  def to_dict(self) -> Dict[str, Any]:
    res = dataclasses.asdict(self)
    res["status"] = self.status.value
    return res

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> StageResult:
    data = dict(data)
    if "status" in data:
      data["status"] = StageStatus(data["status"])
    return cls(**data)


@dataclass
class CampaignState:
  """Atomic, serializable state of an entire experimental campaign."""

  campaign_id: str
  task_name: str
  status: str = "IN_PROGRESS"  # "IN_PROGRESS", "COMPLETED", "FAILED", "PAUSED"
  current_stage: Optional[str] = None
  stages: Dict[str, StageResult] = field(default_factory=dict)
  stages_order: List[str] = field(default_factory=list)
  config_dict: Optional[Dict[str, Any]] = None
  created_at: str = field(
      default_factory=lambda: datetime.datetime.now().isoformat()
  )
  updated_at: str = field(
      default_factory=lambda: datetime.datetime.now().isoformat()
  )

  def mark_stage_running(self, stage_name: str) -> None:
    """Marks a stage as active and running."""
    self.current_stage = stage_name
    self.updated_at = datetime.datetime.now().isoformat()
    if stage_name not in self.stages:
      self.stages[stage_name] = StageResult()
    self.stages[stage_name].status = StageStatus.RUNNING
    self.stages[stage_name].start_time = datetime.datetime.now().isoformat()

  def record_stage_result(self, stage_name: str, result: StageResult) -> None:
    """Updates stage outcome and timestamps.

    The ``start_time`` recorded by :meth:`mark_stage_running` is preserved
    when the stage does not set one itself, so elapsed times survive in the
    state file and can be shown by the CLI.
    """
    previous = self.stages.get(stage_name)
    if result.start_time is None and previous is not None:
      result.start_time = previous.start_time
    result.end_time = datetime.datetime.now().isoformat()
    self.stages[stage_name] = result
    self.updated_at = datetime.datetime.now().isoformat()

  def mark_stage_failed(self, stage_name: str, error_message: str) -> None:
    """Flags a stage as failed *without* discarding its progress.

    Overwriting the whole ``StageResult`` on failure would drop the sweep id
    and any partial metrics, so the next ``resume`` would re-register the
    sweep and pay for every trial a second time.

    Args:
      stage_name: Stage that raised.
      error_message: Human readable cause, stored for the report.
    """
    result = self.stages.get(stage_name)
    if result is None:
      result = StageResult()
      self.stages[stage_name] = result
    result.status = StageStatus.FAILED
    result.error_message = error_message
    result.end_time = datetime.datetime.now().isoformat()
    self.updated_at = datetime.datetime.now().isoformat()

  def merge_stage_metrics(
      self, stage_name: str, metrics: Dict[str, Any]
  ) -> None:
    """Merges partial metrics into a stage result as they become available.

    Used by long multi-phase stages (evaluation) so an expensive, already
    scored model is not re-evaluated after an interruption.

    Args:
      stage_name: Stage owning the metrics.
      metrics: Metric keys to merge into the stored result.
    """
    result = self.stages.get(stage_name)
    if result is None:
      result = StageResult(status=StageStatus.RUNNING)
      self.stages[stage_name] = result
    result.metrics.update(metrics)
    self.updated_at = datetime.datetime.now().isoformat()

  def is_stage_completed(self, stage_name: str) -> bool:
    """Checks if a stage was successfully completed previously."""
    if stage_name not in self.stages:
      return False
    return self.stages[stage_name].status == StageStatus.COMPLETED

  def get_next_pending_stage(
      self, stage_order: List[str]
  ) -> Optional[str]:
    """Returns the first stage in stage_order that is not yet completed."""
    for stage_name in stage_order:
      if not self.is_stage_completed(stage_name):
        return stage_name
    return None

  def get_model_repo_id(self, stage_name: str) -> Optional[str]:
    """Retrieves the Hugging Face model repo ID produced by a stage."""
    if stage_name in self.stages:
      return self.stages[stage_name].model_repo_id
    return None

  def to_dict(self) -> Dict[str, Any]:
    """Serializes the state to a JSON-compatible dictionary."""
    return {
        "campaign_id": self.campaign_id,
        "task_name": self.task_name,
        "status": self.status,
        "current_stage": self.current_stage,
        "stages": {k: v.to_dict() for k, v in self.stages.items()},
        "stages_order": self.stages_order,
        "config_dict": self.config_dict,
        "created_at": self.created_at,
        "updated_at": self.updated_at,
    }

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> CampaignState:
    """Deserializes CampaignState from a dictionary."""
    stages_data = data.get("stages", {})
    stages = {k: StageResult.from_dict(v) for k, v in stages_data.items()}
    return cls(
        campaign_id=data["campaign_id"],
        task_name=data["task_name"],
        status=data.get("status", "IN_PROGRESS"),
        current_stage=data.get("current_stage"),
        stages=stages,
        stages_order=data.get("stages_order", []),
        config_dict=data.get("config_dict"),
        created_at=data.get(
            "created_at", datetime.datetime.now().isoformat()
        ),
        updated_at=data.get(
            "updated_at", datetime.datetime.now().isoformat()
        ),
    )

  def save(self, filepath: str) -> None:
    """Atomically writes state to a JSON file to prevent partial write corruption."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    dir_name = os.path.dirname(os.path.abspath(filepath))
    with tempfile.NamedTemporaryFile(
        "w", dir=dir_name, delete=False, encoding="utf-8"
    ) as tf:
      json.dump(self.to_dict(), tf, indent=2)
      temp_name = tf.name
    os.replace(temp_name, filepath)

  @classmethod
  def load(cls, filepath: str) -> CampaignState:
    """Loads CampaignState from an existing JSON file."""
    if not os.path.exists(filepath):
      raise FileNotFoundError(f"State file '{filepath}' does not exist.")
    with open(filepath, "r", encoding="utf-8") as f:
      data = json.load(f)
    return cls.from_dict(data)

  @classmethod
  def load_or_create(
      cls, filepath: str, campaign_id: str, task_name: str
  ) -> CampaignState:
    """Loads existing state or initializes a new one if file does not exist."""
    if os.path.exists(filepath):
      return cls.load(filepath)
    new_state = cls(campaign_id=campaign_id, task_name=task_name)
    new_state.save(filepath)
    return new_state
