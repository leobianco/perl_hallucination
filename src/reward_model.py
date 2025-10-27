"""
reward_model.py

This module implements the training and evaluation pipeline for a reward model using sequence classification with LoRA (Low-Rank Adaptation) and Hugging Face Transformers. It supports augmenting datasets with organic and structured hallucination samples, computes ROC-AUC metrics, and pushes the trained model to the Hugging Face Hub.

Functions:
    main(): Entry point for training and evaluating the reward model.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./reward_model.sh npov
"""

from src.pipelines import RewardModelPipeline


def main():
    """Run the reward model pipeline (refactored to use `pipelines.RewardModelPipeline`)."""

    RewardModelPipeline().run()


if __name__ == "__main__":
    main()
