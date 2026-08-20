"""Entry point for Reward Model (RM) training with configurable terminal token formatting.

New Token Mismatch Flags:
-------------------------
--rm_terminal_token (str, default: 'eos'):
    Specifies which terminal token to append to prompt+completion text during
    reward model training. The sequence classification head pools logits at this
    final token position.
    Options:
      - 'eos'         : Appends <eos> (SFT-style document termination).
      - 'end_of_turn' : Appends <end_of_turn> (dialogue turn closure).
      - 'none'        : No terminal token added; scores raw last word/punctuation.
      - '<custom>'    : Any custom token string.

Usage:
  python src/token_mismatch/reward_model.py --task_name npov --rm_terminal_token eos ...
  python src/token_mismatch/reward_model.py --task_name npov --rm_terminal_token end_of_turn ...
  python src/token_mismatch/reward_model.py --task_name npov --rm_terminal_token none ...
"""

from src.token_mismatch.pipelines import TokenMismatchRewardModelPipeline


def main():
  """Run the token mismatch reward model training pipeline."""
  TokenMismatchRewardModelPipeline().run()


if __name__ == "__main__":
  main()
