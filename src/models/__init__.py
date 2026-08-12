"""Custom model architectures and extensions for new_perl."""

from src.models.gemma4_sequence_classification import (
    Gemma4ForSequenceClassification,
    register_gemma4_for_sequence_classification,
)

__all__ = [
    "Gemma4ForSequenceClassification",
    "register_gemma4_for_sequence_classification",
]
