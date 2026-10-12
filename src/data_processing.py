from argparse import ArgumentParser

from src.utils import get_task_processor


def main():
  parser = ArgumentParser()
  parser.add_argument("--task_name", type=str)
  parser.add_argument("--seed", type=int, default=12345)
  parser.add_argument("--hf_repo", type=str)
  parser.add_argument(
      "--synthetic_hallus_llm",
      default=False,
      type=lambda x: (str(x).lower() == "true"),
  )
  parser.add_argument(
      "--synthetic_hallus_struct",
      default=False,
      type=lambda x: (str(x).lower() == "true"),
  )
  parser.add_argument(
      "--num_synth_hallus",
      type=int,
      default=0,
      help=(
          "Number of non-hallucinated samples to use as basis (-1 to use all"
          " available samples)."
      ),
  )
  parser.add_argument("--gemini_api_key", type=str)
  parser.add_argument("--synth_llm_temperature", type=float, default=0.7)
  parser.add_argument("--synth_llm_num_fewshot", type=int, default=2)
  parser.add_argument(
      "--synth_struct_top_k",
      type=int,
      default=3,
      help="Maximum rank k of context sentences to erase for hallucination.",
  )
  parser.add_argument(
      "--synth_struct_hallu_threshold",
      type=float,
      default=0.40,
      help="Minimum ROUGE-1 score for context sentence to count as hallucination.",
  )
  parser.add_argument(
      "--synth_struct_irrelevant_threshold",
      type=float,
      default=0.15,
      help="Maximum ROUGE-1 score for context sentence to count as irrelevant.",
  )
  parser.add_argument(
      "--synth_struct_max_nonhall_per_entry",
      type=int,
      default=2,
      help="Maximum non-hallucinated variations per entry.",
  )
  parser.add_argument(
      "--synth_struct_balance_ratio",
      type=float,
      default=1.25,
      help="Maximum allowable ratio between majority and minority classes.",
  )
  parser.add_argument(
      "--drop_bad_quality",
      default=True,
      type=lambda x: (str(x).lower() == "true"),
      help=(
          "Drop annotator-flagged rows (quality != 'good', i.e."
          " 'incorrect_refusal' and 'truncated'). RAGTruth labels these as"
          " non-hallucinated, so keeping them injects refusals into the SFT"
          " targets."
      ),
  )
  parser.add_argument(
      "--split_frac_sft",
      type=float,
      default=0.40,
      help=(
          "Fraction of training source_ids reserved for SFT. Splits are made"
          " over source_ids, never rows, because one RAGTruth source yields up"
          " to six responses sharing a single prompt."
      ),
  )
  parser.add_argument(
      "--split_frac_perl",
      type=float,
      default=0.25,
      help=(
          "Fraction of training source_ids reserved for PE-RL rollout prompts."
          " The remainder (1 - sft - perl) goes to the reward model. The three"
          " blocks are disjoint."
      ),
  )
  parser.add_argument(
      "--sft_val_fraction",
      type=float,
      default=0.15,
      help="Fraction of SFT-block sources held out for SFT validation.",
  )
  parser.add_argument(
      "--perl_num_test_prompts",
      type=int,
      default=50,
      help="Number of held-out test prompts kept for PE-RL evaluation.",
  )
  parser.add_argument(
      "--test_dev_fraction",
      type=float,
      default=1.0 / 3.0,
      help=(
          "Fraction of the official test SOURCES reserved as the 'dev' pool"
          " used for every model-selection decision (reward-model eval,"
          " autorater threshold, PE-RL eval prompts). The remaining sources"
          " form the 'final' pool, read only by the reported evaluation set."
          " Without this split the number you report is the best of many"
          " noisy estimates on the rows you tuned against, not a held-out"
          " score. Must be strictly between 0 and 1."
          " RAGTruth ships exactly 150 held-out test SOURCE passages per"
          " subtask (900 test responses = 150 sources x 6 generator LLMs),"
          " and every consumer dedupes to one prompt per source, so 150 is"
          " the hard ceiling on unique evaluation prompts. The 1/3 default"
          " therefore yields 50 dev (tuning) and 100 final (reported)"
          " prompts."
      ),
  )
  parser.add_argument(
      "--synth_struct_max_responses_per_source",
      type=int,
      default=2,
      help=(
          "Max responses per source fed to synthetic generation. All responses"
          " of a source share one context, so tampering each yields"
          " near-duplicates."
      ),
  )
  parser.add_argument(
      "--synth_struct_max_hallu_per_entry",
      type=int,
      default=1,
      help=(
          "Max hallucinated variants generated per entry (caps"
          " synth_struct_top_k)."
      ),
  )
  parser.add_argument(
      "--synth_perturb_fraction",
      type=float,
      default=0.5,
      help=(
          "Fraction of reward-model sources assigned to the hallucinated"
          " synthesis arm; the rest form the faithful arm. The arms are"
          " disjoint by source_id so no context is seen under both labels."
      ),
  )
  parser.add_argument(
      "--subtask",
      type=str,
      default=None,
      help=(
          "Optional subtask override (e.g. 'qa' or 'summarization' for"
          " ragtruth)."
      ),
  )
  args = parser.parse_args()

  if args.subtask and args.task_name in (
      "ragtruth",
      "ragtruth-qa",
      "ragtruth-summarization",
  ):
    sub = args.subtask.lower().strip()
    if "sum" in sub:
      args.task_name = "ragtruth-summarization"
    elif "data" in sub:
      raise ValueError(
          "Data-to-text subtask has been dropped from RAGTruth in this codebase. "
          "Supported task names are 'ragtruth-qa' and 'ragtruth-summarization'."
      )
    else:
      args.task_name = "ragtruth-qa"

  if "data" in (args.task_name or "").lower():
    raise ValueError(
        "Data-to-text subtask has been dropped from RAGTruth in this codebase. "
        "Supported task names are 'ragtruth-qa' and 'ragtruth-summarization'."
    )

  processor_cls = get_task_processor(args.task_name)
  processor = processor_cls(args)
  processor.run()


if __name__ == "__main__":
  main()
