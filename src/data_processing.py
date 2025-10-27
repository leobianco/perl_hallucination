from argparse import ArgumentParser

from utils import get_task_processor


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
    args = parser.parse_args()

    processor_cls = get_task_processor(args.task_name)
    processor = processor_cls(args)
    processor.run()


if __name__ == "__main__":
    main()
