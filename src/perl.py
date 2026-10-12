"""Script for fine-tuning a model using Parameter-Efficient Reinforcement Learning (PE-RL).

Implementation via Hugging Face Transformers and TRL libraries.
Dispatches to PERLPipeline, which loads datasets, tokenizes data, initializes models
(policy and reward model), and sets up the RLOOTrainer for RL training.

With ``--continual_eval True``, this process instead becomes the GPU-free
coordinator of ``src.continual_eval.run_perl_with_continual_eval``, which
alternates training segments (subprocesses of this same script) and autorater
evaluations.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./scripts/perl.sh npov
"""

import os
import sys

from src.continual_eval import (
    run_perl_with_continual_eval,
    should_coordinate_continual_eval,
)


def main(argv=None):
  """Runs the PERL pipeline, or coordinates its continual evaluation."""
  cli_args = list(sys.argv[1:] if argv is None else argv)
  if should_coordinate_continual_eval(cli_args, os.environ):
    rc = run_perl_with_continual_eval(cli_args)
    if rc != 0:
      sys.exit(rc)
    return
  # Imported here so that the coordinator, which only launches subprocesses,
  # never loads the training stack (torch, TRL, vLLM).
  from src.pipelines import PERLPipeline  # pylint: disable=g-import-not-at-top

  PERLPipeline().run(*cli_args)


if __name__ == "__main__":
  main()
