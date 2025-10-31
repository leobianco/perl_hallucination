"""
Main script to dispatch and run various LLM evaluation pipelines (autorating, generation, or scoring) based on command-line arguments.

To generate completions on test set:
./scripts/evaluator.sh npov generate

To score these completions
./scripts/evaluator.sh npov score 

To evaluate the quality of the autorater model:
./scripts/evaluator.sh npov autoratereval
"""

from transformers import HfArgumentParser, set_seed

from src.pipelines import (
    EvaluationAutoraterPipeline,
    EvaluationGenerationPipeline,
    EvaluationScoringPipeline,
)

from src.utils import EvalArguments


def main():
    parser = HfArgumentParser(EvalArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    if script_args.evaluate_evaluator:
        pipe = EvaluationAutoraterPipeline()
    elif script_args.dataset_with_completions is None:
        pipe = EvaluationGenerationPipeline()
    else:
        pipe = EvaluationScoringPipeline()

    pipe.run()


if __name__ == "__main__":
    main()
