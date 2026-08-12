"""Script for generating synthetic preference data for the SCOPE baseline.

Generates dispreferred completions using noisy decoding by mixing
logits/probabilities
between the fine-tuned SFT checkpoint and the base pre-trained model as
described in
Section 3 / Algorithm 1 of the SCOPE paper (https://arxiv.org/abs/2502.13674).

Usage:
    ./scripts/scope_data_generation.sh npov
"""

from src.pipelines import ScopeDataGenerationPipeline


def main():
  """Run the SCOPE data generation pipeline."""
  ScopeDataGenerationPipeline().run()


if __name__ == "__main__":
  main()
