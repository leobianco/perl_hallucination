"""Auto-PERL Orchestrator Package.

A modular, zero-lockin scientific campaign orchestrator for Parameter-Efficient
Reinforcement Learning (PE-RL) experiments.
"""

from src.orchestrator.config import CampaignConfig
from src.orchestrator.config import EvalStageConfig
from src.orchestrator.config import SweepStageConfig
from src.orchestrator.engine import CampaignEngine
from src.orchestrator.state import CampaignState
from src.orchestrator.state import StageStatus

__all__ = [
    "CampaignConfig",
    "SweepStageConfig",
    "EvalStageConfig",
    "CampaignEngine",
    "CampaignState",
    "StageStatus",
]

