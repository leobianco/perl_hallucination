"""Script for generating synthetic preference data for the SSFO baseline.

Generates preference data by contrasting the SFT model's generations with
context (chosen)
and without context (rejected / hallucinated) as described in:
"SSFO: Self-Supervised Faithfulness Optimization for Retrieval-Augmented
Generation"
(https://arxiv.org/abs/2508.17225).

Usage:
    ./scripts/ssfo_data_generation.sh npov
"""

# Patch transformers heterogeneity configuration to allow global attribute access in vLLM
try:
  import transformers
  from transformers.configuration_utils import PretrainedConfig

  PretrainedConfig.allow_global_per_layer_attribute_access = True
  from transformers.integrations.heterogeneity.configuration_utils import (
      HeterogeneousPretrainedConfig,
  )

  HeterogeneousPretrainedConfig.allow_global_per_layer_attribute_access = True
except Exception:
  pass

from src.pipelines import SSFODataGenerationPipeline


def main():
  """Run the SSFO data generation pipeline."""
  SSFODataGenerationPipeline().run()


if __name__ == "__main__":
  main()
