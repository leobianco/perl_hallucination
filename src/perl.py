"""
Script for fine-tuning a model using Parameter-Efficient Reinforcement Learning (PE-RL). Implementation via Hugging Face Transformers and TRL libraries. Dispatches to PERLPipeline, which loads datasets, tokenizes data, initializes models (policy, reference policy, reward model), and sets up the RLOOTrainer for RL training. The script also disables automatic model compilation due to known issues with torch.compile in recent Transformers versions. Training logs are saved to disk after completion.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./scripts/perl.sh npov
"""

import torch

from src.pipelines import PERLPipeline


def no_compile(model, *args, **kwargs):
    """Disables torch.compile for the model to avoid unwanted compilation and related errors.

    Args:
        model (torch.nn.Module): The model to (not) compile.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        torch.nn.Module: The unmodified model.

    Notes:
        See: https://github.com/huggingface/transformers/issues/39191
        Transformers v4.53.0 introduced automatic compilation of forward passes, which can cause
        excessive recompilations and errors. This function disables compilation by overriding torch.compile.
    """

    print("[INFO] torch.compile() was called but it is disabled by the user")

    return model


torch.compile = no_compile


def main():
    """Run the PERL pipeline."""

    PERLPipeline().run()


if __name__ == "__main__":
    main()
