"""Token Mismatch Experiment Module.

Provides modular pipelines, runners, and utilities to explore and evaluate
terminal token mismatches between reward model training and PE-RL rollout scoring
(e.g., training RM on <eos> while scoring PE-RL rollouts on <end_of_turn>).

This module is completely self-contained and does not modify any core codebase files.
It can be safely deleted or modified without impacting other experiments.
"""

from src.token_mismatch.pipelines import (
    TokenMismatchPERLPipeline,
    TokenMismatchRewardModelPipeline,
    apply_terminal_token_formatting,
    resolve_terminal_token_string,
)

__all__ = [
    "TokenMismatchRewardModelPipeline",
    "TokenMismatchPERLPipeline",
    "apply_terminal_token_formatting",
    "resolve_terminal_token_string",
]
