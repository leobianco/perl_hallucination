"""Abstract base class and shared context for all campaign stages."""

from __future__ import annotations

import abc
from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, Optional, Set

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
      message = (
          f"Resuming existing {self.name.upper()} sweep {previous.sweep_id} "
          "instead of starting a new one."
      )
      logger.info(message)
      if live_line_callback:
        live_line_callback(message)
      return previous.sweep_id

    sweep_id = self.sweep_controller.create_sweep(sweep_dict)
    self.record_sweep_id(sweep_id)
    return sweep_id

  def record_sweep_id(self, sweep_id: str) -> None:
    """Persists ``sweep_id`` immediately so a crash stays resumable."""
    result = self.state.stages.get(self.name)
    if result is None:
      result = StageResult(status=StageStatus.RUNNING)
      self.state.stages[self.name] = result
    result.sweep_id = sweep_id
    if self.context.state_path:
      try:
        self.state.save(self.context.state_path)
      except OSError as exc:
        logger.warning("Could not persist sweep id to state file: %s", exc)

