"""Script for fine-tuning a model using Parameter-Efficient Reinforcement Learning (PE-RL).

Implementation via Hugging Face Transformers and TRL libraries.
Dispatches to PERLPipeline, which loads datasets, tokenizes data, initializes models
(policy and reward model), and sets up the RLOOTrainer for RL training.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./scripts/perl.sh npov
"""

from src.pipelines import PERLPipeline


def main():
    """Run the PERL pipeline."""
    PERLPipeline().run()


if __name__ == "__main__":
    main()
