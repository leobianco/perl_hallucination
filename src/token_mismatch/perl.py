"""Entry point for Parameter-Efficient Reinforcement Learning (PE-RL) with configurable rollout scoring tokens.

New Token Mismatch Flags:
-------------------------
--scoring_terminal_token (str, default: 'end_of_turn'):
    Specifies which terminal token is formatted onto the policy rollout completion
    before feeding it into the Reward Model during RLOO training.
    Options:
      - 'end_of_turn' : Appends <end_of_turn> to completion (dialogue turn closure).
      - 'eos'         : Appends <eos> to completion (matches SFT-style RM training).
      - 'as_generated': Passes completion as returned by TRL decoding.
      - 'none'        : Strips special tokens, scoring the raw last word/punctuation.
      - '<custom>'    : Any custom token string.

--force_terminal_token_swap (bool, default: False):
    When True, explicitly strips any existing terminal delimiters (<end_of_turn>,
    <eos>, etc.) from the completion before appending --scoring_terminal_token.

Usage:
  python src/token_mismatch/perl.py --task_name npov --scoring_terminal_token end_of_turn ...
  python src/token_mismatch/perl.py --task_name npov --scoring_terminal_token eos --force_terminal_token_swap True ...
  python src/token_mismatch/perl.py --task_name npov --scoring_terminal_token none ...
"""

from src.token_mismatch.pipelines import TokenMismatchPERLPipeline


def main():
  """Run the token mismatch PERL training pipeline."""
  TokenMismatchPERLPipeline().run()


if __name__ == "__main__":
  main()
