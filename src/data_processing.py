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
  parser.add_argument("--num_synth_hallus", type=int, default=0)
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
      default=0.20,
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
  args = parser.parse_args()

  processor_cls = get_task_processor(args.task_name)
  processor = processor_cls(args)
  processor.run()


if __name__ == "__main__":
  main()
