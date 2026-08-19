"""This module provides the main script for Direct Preference Optimization (DPO) of language models using LoRA and TRL.

Usage:
    ./scripts/scope_dpo.sh npov
"""

from src.pipelines import DPOPipeline


def main():
  """Run the DPO preference tuning pipeline."""
  DPOPipeline().run()


if __name__ == "__main__":
  main()
