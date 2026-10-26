"""Main script to dispatch and run various LLM evaluation pipelines (autorating, generation, or scoring) based on command-line arguments.

To generate completions on test set:
./scripts/evaluator.sh npov generate

To score these completions
./scripts/evaluator.sh npov score

To evaluate the quality of the autorater model:
./scripts/evaluator.sh npov autoratereval

To measure how often the reward-hacking rubric flags healthy references:
./scripts/evaluator.sh npov reward_hacking_autorater_eval
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

from src import reward_hacking_calibration
from src.pipelines import (
    EvaluationAutoraterPipeline,
    EvaluationGenerationPipeline,
    EvaluationRewardHackingAutoraterPipeline,
    EvaluationScoringPipeline,
)
from src.utils import EvalArguments
from transformers import HfArgumentParser, set_seed


def main():
  parser = HfArgumentParser(EvalArguments)
  script_args = parser.parse_args_into_dataclasses()[0]
  set_seed(script_args.seed)

  # Checked first: this mode must never fall through to the hallucination
  # autorater evaluation because --evaluate_evaluator was left on.
  if script_args.mode == reward_hacking_calibration.MODE:
    pipe = EvaluationRewardHackingAutoraterPipeline()
  elif script_args.mode == "autoratereval" or script_args.evaluate_evaluator:
    pipe = EvaluationAutoraterPipeline()
  elif script_args.mode == "generate":
    pipe = EvaluationGenerationPipeline()
  elif script_args.mode == "score":
    pipe = EvaluationScoringPipeline()
  elif script_args.dataset_with_completions is None:
    pipe = EvaluationGenerationPipeline()
  else:
    pipe = EvaluationScoringPipeline()

  pipe.run()


if __name__ == "__main__":
  main()
