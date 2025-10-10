"""
Thin dispatcher for evaluator-related flows.

This module delegates to the pipeline implementations in
`scripts.pipelines`. It intentionally keeps the interface small and
lightweight so the heavy code (vLLM/gemini/plotting) lives in the pipeline
implementations.
"""

from transformers import HfArgumentParser, set_seed

from scripts.evaluator_args import ScriptArguments as EvalArgs
from scripts.pipelines import (
    EvaluationAutoraterPipeline,
    EvaluationGenerationPipeline,
    EvaluationScoringPipeline,
)


def main():
    parser = HfArgumentParser(EvalArgs)
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
