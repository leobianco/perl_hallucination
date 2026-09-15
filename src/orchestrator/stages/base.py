"""Abstract base class and shared context for all campaign stages."""

from __future__ import annotations

import abc
from dataclasses import dataclass
import glob
import json
import logging
import os
import re
from typing import Any, Callable, Dict, Optional, Set, Tuple

from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.state import CampaignState, StageResult, StageStatus
from src.orchestrator.sweep_controller import SweepController


logger = logging.getLogger(__name__)


@dataclass
class CampaignContext:
  """Shared runtime execution context passed across stages."""

  config: CampaignConfig
  state: CampaignState
  sweep_controller: SweepController
  model_manager: ModelManager
  #: Where the engine persists ``state``; lets stages checkpoint a sweep id
  #: as soon as it exists, so a crash mid-sweep is resumable.
  state_path: Optional[str] = None
  #: Predicate that is True only for a *hard* abort (``[x]`` / Ctrl-C twice).
  #: Stages use it for the post-sweep materialization, which must survive the
  #: ``[a]`` advance and ``[s]`` stop intents that cut the sweep agent short.
  abort_requested_callback: Optional[Callable[[], bool]] = None

  @property
  def sft_model_repo_id(self) -> Optional[str]:
    """Resolves the SFT model repo ID from state or configuration overrides."""
    if self.config.perl.sft_model_path and self.config.perl.sft_model_path != "auto":
      return self.config.perl.sft_model_path
    return self.state.get_model_repo_id("sft")

  @property
  def reward_model_repo_id(self) -> Optional[str]:
    """Resolves the Reward Model repo ID from state or configuration overrides."""
    if (
        self.config.perl.reward_model_path
        and self.config.perl.reward_model_path != "auto"
    ):
      return self.config.perl.reward_model_path
    return self.state.get_model_repo_id("rm")

  @property
  def perl_model_repo_id(self) -> Optional[str]:
    """Resolves the final PE-RL model repo ID from state."""
    return self.state.get_model_repo_id("perl")


class BaseStage(abc.ABC):
  """Abstract interface for a campaign execution stage."""

  def __init__(self, context: CampaignContext):
    self.context = context
    self.config = context.config
    self.state = context.state
    self.sweep_controller = context.sweep_controller
    self.model_manager = context.model_manager

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Stage identifier (e.g. 'sft', 'rm', 'perl', 'eval')."""

  @abc.abstractmethod
  def execute(
      self,
      live_line_callback: Optional[Callable[[str], None]] = None,
      stop_requested_callback: Optional[Callable[[], bool]] = None,
  ) -> StageResult:
    """Executes the stage logic and returns the resulting StageResult."""

  # --- Shared helpers -------------------------------------------------
  def tunable_keys(self, sweep_dict: Dict[str, Any]) -> Set[str]:
    """Returns the hyperparameter names this sweep actually searches over.

    Only these may be replayed on the materialization command line: the rest
    of a W&B run config is the training-argument and model-config dump, which
    the training scripts' argument parsers reject.

    Args:
      sweep_dict: The sweep configuration that was registered.

    Returns:
      The set of parameter names, or an empty set when the sweep declares
      none (the model manager then falls back to its own allowlist).
    """
    params = sweep_dict.get("parameters")
    if isinstance(params, dict):
      return {str(key) for key in params}
    return set()

  def materialization_stop_callback(self) -> Optional[Callable[[], bool]]:
    """Returns the predicate that may interrupt the winner's retraining.

    The predicate handed to the *sweep agent* is True for ``[a]`` advance and
    ``[s]`` stop, because both mean "seal the sweep now". Reusing it for the
    materialization made the retraining subprocess die on its very first
    poll, raising "Materialization ... was interrupted before the checkpoint
    could be pushed" - a non-retryable error that failed the whole campaign
    precisely when the user asked to *keep going* with the current best run.

    Only a hard abort may kill the retraining; when no abort predicate was
    supplied (e.g. a stage constructed directly in a test), materialization
    simply runs to completion.
    """
    return self.context.abort_requested_callback

  def remaining_runs(
      self,
      sweep_id: str,
      max_runs: int,
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> int:
    """Returns how many trials still need to run to reach ``max_runs``.

    ``wandb agent --count N`` bounds a single agent process, not the sweep.
    Passing the full budget to a resumed stage runs a second full budget on
    top of the trials already paid for: stopping at 6 of 10 and resuming
    would produce 16 trials, not 10.

    The count comes from W&B, which is the only party that knows the real
    total. When it cannot be established the budget is left untouched -
    overshooting costs GPU hours, but undershooting would silently shrink
    the search the user asked for.

    Args:
      sweep_id: The sweep being executed.
      max_runs: The configured trial budget for the whole stage.
      live_line_callback: Optional sink for an explanatory line.

    Returns:
      Trials left to run; 0 when the budget is already spent.
    """
    budget = max(0, int(max_runs or 0))
    done = self.sweep_controller.count_finished_runs(sweep_id)
    if done is None or done <= 0:
      return budget

    remaining = max(0, budget - int(done))
    if live_line_callback:
      if remaining:
        live_line_callback(
            f"Sweep already has {done}/{budget} finished trials; running the"
            f" remaining {remaining}."
        )
      else:
        live_line_callback(
            f"Sweep already has {done}/{budget} finished trials; skipping the"
            " agent and scoring the trials that exist."
        )
    return remaining

  def get_sweep_descriptor(self, sweep_dict: Dict[str, Any]) -> Tuple[str, str]:
    """Returns (model_short_name, stage_type_string) for this stage.

    Args:
      sweep_dict: Sweep configuration dictionary.

    Returns:
      Tuple of (short model name, stage type string).
    """
    del sweep_dict  # Default implementation does not inspect command args.
    model_short = self.config.base_model.rstrip("/").split("/")[-1]
    return model_short, self.name.upper()

  def determine_sweep_number(self, prefix: str) -> int:
    """Calculates the 1-based sweep number for this stage.

    Counts prior sweeps matching this prefix from local checkpoint
    state files and the Weights & Biases API.

    Args:
      prefix: Prefix string identifying the task, model, and stage.

    Returns:
      Next 1-based sweep number.
    """
    seen_numbers: Set[int] = set()
    state_file_path = (
        self.config.state_file
        or f"./checkpoints/{self.config.task_name}/state.json"
    )
    checkpoints_dir = os.path.dirname(os.path.abspath(state_file_path))

    if os.path.isdir(checkpoints_dir):
      for fpath in glob.glob(os.path.join(checkpoints_dir, "*_state.json")):
        fpath = os.path.abspath(fpath)
        if fpath == os.path.abspath(state_file_path):
          continue
        try:
          with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
          if data.get("campaign_id") == self.state.campaign_id:
            continue
          if data.get("task_name") != self.config.task_name:
            continue
          stage_data = data.get("stages", {}).get(self.name, {})
          sw_name = stage_data.get("sweep_name")
          if sw_name and sw_name.startswith(prefix):
            m = re.search(r"Sweep #(\d+)", sw_name)
            if m:
              seen_numbers.add(int(m.group(1)))
            else:
              seen_numbers.add(len(seen_numbers) + 1)
        except (OSError, json.JSONDecodeError):
          continue

    if not self.config.dry_run:
      try:
        import wandb  # pylint: disable=g-import-not-at-top

        api = wandb.Api()
        entity = self.sweep_controller.resolve_entity()
        proj = self.config.project
        sweeps = api.sweeps(f"{entity}/{proj}")
        for s in sweeps:
          s_name = getattr(s, "name", "") or ""
          if s_name.startswith(prefix):
            m = re.search(r"Sweep #(\d+)", s_name)
            if m:
              seen_numbers.add(int(m.group(1)))
            else:
              seen_numbers.add(len(seen_numbers) + 1)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("Could not query existing sweeps from W&B: %s", e)

    return max(seen_numbers, default=0) + 1

  def generate_sweep_name(self, sweep_dict: Dict[str, Any]) -> str:
    """Constructs a descriptive human-readable W&B sweep name.

    Format: [TASK NAME] [MODEL NAME & SIZE] [STAGE / DATA TYPE] [Sweep #X]
    Example: BOSCH gemma-4-E2B-it SFT Sweep #1
             BOSCH gemma-3-1b-it RM Organic Sweep #1
             BOSCH gemma-4-E2B-it PERL Organic Sweep #1

    Args:
      sweep_dict: Sweep configuration dictionary.

    Returns:
      Descriptive sweep name formatted string.
    """
    task_str = self.config.task_name.upper()
    model_short, stage_type_str = self.get_sweep_descriptor(sweep_dict)
    prefix = f"{task_str} {model_short} {stage_type_str}"
    sweep_num = self.determine_sweep_number(prefix)
    return f"{prefix} Sweep #{sweep_num}"

  def resolve_sweep_id(
      self,
      sweep_dict: Dict[str, Any],
      live_line_callback: Optional[Callable[[str], None]] = None,
  ) -> str:
    """Reuses the sweep of an interrupted attempt, or registers a new one.

    Registering a fresh sweep after a crash silently discards every trial the
    previous attempt paid for. When the persisted state still holds a sweep id
    for this stage and the stage never completed, the agent is pointed back at
    that sweep instead.

    The recorded sweep is verified first. A sweep deleted from the W&B UI
    between two runs would otherwise be handed to ``wandb agent``, which
    fails the whole campaign with a confusing "sweep not found" - even
    though there is nothing to salvage and a new sweep is exactly what the
    user wants. Verification is tri-state on purpose: when W&B cannot be
    reached the sweep is *kept*, because abandoning a live sweep on a
    transient error is far more expensive than a failed agent launch.

    Args:
      sweep_dict: Sweep configuration to register when there is nothing to
        resume.
      live_line_callback: Optional log sink.

    Returns:
      The sweep id to run the agent against.
    """
    previous = self.state.stages.get(self.name)
    if (
        previous is not None
        and previous.sweep_id
        and previous.status != StageStatus.COMPLETED
    ):
      name_hint = f" ({previous.sweep_name})" if previous.sweep_name else ""
      still_exists = self.sweep_controller.sweep_exists(previous.sweep_id)
      if still_exists is False:
        message = (
            f"Recorded {self.name.upper()} sweep {previous.sweep_id}"
            f"{name_hint} no longer exists on W&B (deleted?). "
            "Registering a new sweep."
        )
        logger.warning(message)
        if live_line_callback:
          live_line_callback(f"[WARNING] {message}")
        # Drop the stale pointer so a later crash does not resurrect it.
        previous.sweep_id = None
        previous.sweep_name = None
        if self.context.state_path:
          try:
            self.state.save(self.context.state_path)
          except OSError as exc:
            logger.warning("Could not persist cleared sweep id: %s", exc)
      else:
        if still_exists is None:
          logger.warning(
              "Could not confirm sweep %s exists; reusing it rather than "
              "risking the loss of completed trials.",
              previous.sweep_id,
          )
        message = (
            f"Resuming existing {self.name.upper()} sweep "
            f"{previous.sweep_id}{name_hint} instead of starting a new one."
        )
        logger.info(message)
        if live_line_callback:
          live_line_callback(message)
        return previous.sweep_id

    # If sweep_dict doesn't already have a descriptive name, generate one
    if not sweep_dict.get("name"):
      sweep_name = self.generate_sweep_name(sweep_dict)
      sweep_dict["name"] = sweep_name
    else:
      sweep_name = sweep_dict["name"]

    message = f"Registering sweep '{sweep_name}' on W&B..."
    logger.info(message)
    if live_line_callback:
      live_line_callback(message)

    sweep_id = self.sweep_controller.create_sweep(sweep_dict)
    self.record_sweep_id(sweep_id, sweep_name=sweep_name)
    return sweep_id

  def record_sweep_id(
      self, sweep_id: str, sweep_name: Optional[str] = None
  ) -> None:
    """Persists ``sweep_id`` and optional ``sweep_name`` immediately so a crash stays resumable."""
    result = self.state.stages.get(self.name)
    if result is None:
      result = StageResult(status=StageStatus.RUNNING)
      self.state.stages[self.name] = result
    result.sweep_id = sweep_id
    if sweep_name:
      result.sweep_name = sweep_name
    if self.context.state_path:
      try:
        self.state.save(self.context.state_path)
      except OSError as exc:
        logger.warning("Could not persist sweep id to state file: %s", exc)

