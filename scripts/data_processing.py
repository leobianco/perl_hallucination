from argparse import ArgumentParser

import sys
import os

print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)


from data.bosch_task_processor import BoschTaskProcessor
from data.npov_task_processor import NPOVTaskProcessor
from data.ragtruth_task_processor import RagtruthTaskProcessor


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

    task_map = {
        "npov": NPOVTaskProcessor,
        "bosch": BoschTaskProcessor,
        "ragtruth": RagtruthTaskProcessor,
    }

    processor_cls = task_map.get(args.task_name)
    if processor_cls is None:
        raise ValueError(f"Unknown task: {args.task_name}")

    processor = processor_cls(args)
    processor.run()
