"""
This module provides the main script for supervised fine-tuning (SFT) of language models using LoRA (Low-Rank Adaptation) and the TRL library.

It loads datasets, configures LoRA and SFT training arguments, prepares tokenizers and data collators, and launches training for various prompt-based tasks (e.g., npov, bosch, ragtruth).

Functions:
    main: Entry point for parsing arguments, preparing data, configuring the model, and running SFT training.

Usage: call the associated shell script along with the corresponding task. E.g.:
    ./writer_sft.sh npov
"""

from src.pipelines import SFTPipeline


def main():
    """Run the SFT pipeline (refactored to use `pipelines.SFTPipeline`)."""

    SFTPipeline().run()


if __name__ == "__main__":
    main()
