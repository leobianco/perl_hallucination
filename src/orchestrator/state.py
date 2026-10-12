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
  sweep_name: Optional[str] = None
  best_run_id: Optional[str] = None
  best_metric_val: Optional[float] = None
  best_params: Dict[str, Any] = field(default_factory=dict)
  model_repo_id: Optional[str] = None
  metrics: Dict[str, Any] = field(default_factory=dict)
  error_message: Optional[str] = None
  start_time: Optional[str] = None
  end_time: Optional[str] = None
  #: Sweep trials finished so far, refreshed *while the stage runs*. The
  #: process that owns the dashboard knows this from its own log parser, but
  #: a second process (``status --watch`` in another tmux pane) can only see
  #: what reached the state file - without these it shows a frozen ``00/N``.
  trials_done: int = 0
  #: Trial budget, mirrored here so the counter is meaningful even if the
  #: config is later edited or unavailable.
  trials_total: int = 0
  #: How ``best_metric_val`` was read out of the winning trial: ``final``
  #: (its last logged value), ``final_window`` (the mean of its last
  #: ``selection_window`` points) or ``best`` (its peak over all eval steps).
  #: Without this, two campaigns of the same task report numbers that are
  #: not comparable and nothing on disk says why.
  selection_strategy: Optional[str] = None
  #: Step at which the winner's peak occurred, when that peak is strictly
  #: better than where the run ended. None under ``final`` selection, and
  #: also None when the run simply ended at its best point.
  selection_step: Optional[int] = None
  #: How many trailing points were actually averaged under ``final_window``.
  #: Recorded rather than recomputed from the config because a short trial
  #: yields fewer points than the window asked for, and the report must
  #: quote the number the score was really built from.
  selection_window: Optional[int] = None

  #: The winner's *last* logged value of the metric. Under ``best``
  #: selection the gap to ``best_metric_val`` is the honest measure of how
  #: much of the score is early stopping - a large gap on a noisy metric is
  #: a warning sign, not a result.
  final_metric_val: Optional[float] = None
  #: How the sweep ended, as established against W&B *after* the agent
  #: returned: ``complete`` (the whole budget finished), ``partial`` (fewer
  #: trials finished than were asked for - a crash, a timeout, or a user
  #: stop), or ``unknown`` (W&B could not be reached to find out).
  #:
  #: Without this a stage that lost half its trials to a dead VM is
  #: indistinguishable on disk from one that ran perfectly: both are
  #: COMPLETED, because a sweep that produced *some* trained models still
  #: yields a winner and still publishes a checkpoint.
  sweep_outcome: Optional[str] = None
  #: Human-readable notes about anything that degraded this stage but did
  #: not fail it. Surfaced by the dashboard and the campaign report so a
  #: shortened sweep cannot be mistaken for a clean one.
  warnings: List[str] = field(default_factory=list)

  def to_dict(self) -> Dict[str, Any]:
    res = dataclasses.asdict(self)
    res["status"] = self.status.value
    return res

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> StageResult:
    data = dict(data)
    if "status" in data:
      data["status"] = StageStatus(data["status"])
    # A campaign can outlive the code that started it: a state file written
    # by a newer build (or a hand-edited one) must not make `status` crash in
    # a watching pane. Unknown keys are dropped rather than fatal.
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CampaignState:
  """Atomic, serializable state of an entire experimental campaign."""

  campaign_id: str
  task_name: str
  #: One of ``IN_PROGRESS``, ``COMPLETED``, ``FAILED``, ``PAUSED``,
  #: ``STOPPED`` (``[s]`` or Ctrl-C) or ``ABORTED`` (``[x]``). The last three
  #: mean a human was at the keyboard, which is how
  #: :mod:`src.orchestrator.shutdown` decides not to power the machine off.
  status: str = "IN_PROGRESS"
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
    """Marks a stage as active and running.

    The failure message of the attempt that preceded this one is cleared.
    It describes a run that is over; leaving it in place makes the dashboard
    show a resumed, healthy stage with a red "Materialization ... was
    interrupted" next to it for the rest of the campaign. The degradation
    notes of that attempt go with it, for the same reason and because this
    attempt is about to re-establish them from W&B. Progress, sweep id and
    best-so-far are deliberately *kept* - those still hold.

    Args:
      stage_name: Stage that is (re)starting.
    """
    self.current_stage = stage_name
    self.updated_at = datetime.datetime.now().isoformat()
    if stage_name not in self.stages:
      self.stages[stage_name] = StageResult()
    self.stages[stage_name].status = StageStatus.RUNNING
    self.stages[stage_name].error_message = None
    self.stages[stage_name].warnings = []
    self.stages[stage_name].sweep_outcome = None
    self.stages[stage_name].start_time = datetime.datetime.now().isoformat()

  def update_stage_progress(
      self,
      stage_name: str,
      trials_done: Optional[int] = None,
      trials_total: Optional[int] = None,
      best_metric_val: Optional[float] = None,
  ) -> bool:
    """Records in-flight sweep progress for a running stage.

    This is what makes ``status``/``status --watch`` usable from a second
    tmux pane: that process has no access to the running campaign's in-memory
    log parser, so anything it is meant to display has to be on disk.

    The trial counters are frozen once ``sweep_outcome`` is set, i.e. once
    the stage has asked W&B how many trials really finished. Everything
    after that point is materialization output, which still flows through
    the same line sink; without the freeze its first line would overwrite
    the reconciled count with the stdout estimate again - and the estimate
    counts crashed trials as done, which is the whole reason the
    reconciliation exists. ``best_metric_val`` is deliberately still
    accepted: it is not part of the accounting.

    Args:
      stage_name: Stage currently running.
      trials_done: Trials that reached a terminal state, if known.
      trials_total: Trial budget, if known.
      best_metric_val: Best metric observed so far, if known.

    Returns:
      True when something actually changed, so callers can avoid rewriting
      the state file on every single line of agent output.
    """
    result = self.stages.get(stage_name)
    if result is None:
      result = StageResult(status=StageStatus.RUNNING)
      self.stages[stage_name] = result

    reconciled = result.sweep_outcome is not None
    changed = False
    if (
        not reconciled
        and trials_done is not None
        and int(trials_done) != result.trials_done
    ):
      result.trials_done = int(trials_done)
      changed = True
    if (
        not reconciled
        and trials_total
        and int(trials_total) != result.trials_total
    ):
      result.trials_total = int(trials_total)
      changed = True
    if best_metric_val is not None and best_metric_val != result.best_metric_val:
      result.best_metric_val = float(best_metric_val)
      changed = True

    if changed:
      self.updated_at = datetime.datetime.now().isoformat()
    return changed

  def record_stage_result(self, stage_name: str, result: StageResult) -> None:
    """Updates stage outcome and timestamps.

    The ``start_time`` recorded by :meth:`mark_stage_running` is preserved
    when the stage does not set one itself, so elapsed times survive in the
    state file and can be shown by the CLI. The live trial counters are
    carried over for the same reason: a stage that failed on trial 7 of 15
    should keep saying so instead of resetting to 0.

    Args:
      stage_name: Stage that finished.
      result: Its outcome.
    """
    previous = self.stages.get(stage_name)
    if result.start_time is None and previous is not None:
      result.start_time = previous.start_time
    if previous is not None:
      # The human-readable sweep name is recorded by `record_sweep_id` when
      # the sweep is registered, and stages return a fresh StageResult that
      # only carries the id. Without this the name is dropped on the last
      # line of every stage and the report falls back to a bare
      # `entity/project/ab12cd34`, which is exactly what the descriptive
      # names exist to avoid.
      if not result.sweep_name:
        result.sweep_name = previous.sweep_name
      if not result.sweep_id:
        result.sweep_id = previous.sweep_id
      # Stages report their outcome, not their progress; without this the
      # counters tracked during the sweep would be thrown away on the last
      # line of the stage.
      if not result.trials_done:
        result.trials_done = previous.trials_done
      if not result.trials_total:
        result.trials_total = previous.trials_total
      # Same reasoning for the leader board. A stage that failed part way
      # through, or whose post-sweep W&B query came back empty, still knows
      # the best score it saw - dropping it would blank the "Best" column of
      # a stage that visibly ran trials.
      if result.best_metric_val is None:
        result.best_metric_val = previous.best_metric_val
        # The provenance describes that very number, so it has to travel
        # with it; otherwise the report would label a carried-over score
        # with the current attempt's (unused) strategy.
        result.selection_strategy = previous.selection_strategy
        result.selection_step = previous.selection_step
        result.final_metric_val = previous.final_metric_val
      if not result.best_run_id:
        result.best_run_id = previous.best_run_id
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
