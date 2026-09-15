"""Abstract base class and shared context for all campaign stages."""

from __future__ import annotations

import abc
from dataclasses import dataclass
import logging
from typing import Callable, Optional
from src.orchestrator.config import CampaignConfig
from src.orchestrator.model_manager import ModelManager
from src.orchestrator.state import CampaignState, StageResult
from src.orchestrator.sweep_controller import SweepController

logger = logging.getLogger(__name__)


@dataclass
class CampaignContext:
  """Shared runtime execution context passed across stages."""

  config: CampaignConfig
  state: CampaignState
  sweep_controller: SweepController
  model_manager: ModelManager

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
