"""Triage a scored evaluation run: what did the judge actually see?

A jump in the flagged rate (e.g. 10% -> 71%) has three possible sources:

  1. The completions changed (different adapter, different decoding, empty or
     degenerate text).
  2. The judge prompt changed (different few-shot draw, different task
     processor, wrong column picked up).
  3. The judge itself changed (model version, endpoint).

This script attacks (1) and (2) directly by loading the scored dataset and
cross-tabulating the stored scores against properties of the completion, then
printing the exact prompt the judge received for a few rows of each class.

Usage:
    python3 scripts/triage_eval_scores.py <hf_dataset_with_completions> \
        --task npov --split test --examples 3

The repo ID is printed by the scoring run as
"Loading dataset with completions from ...", and is also recorded in
logs/<run>/..._summary.json as `provenance_dataset_with_completions`.
"""

import argparse
import collections
import re
import statistics
from typing import List, Optional

from datasets import load_dataset

from src.utils import get_task_processor


def looks_truncated(text: str) -> bool:
  """Heuristic: generation stopped mid-sentence (likely hit max_tokens)."""
  stripped = (text or "").rstrip()
  return bool(stripped) and stripped[-1] not in ".!?\"')]}"


def repetition_ratio(text: str, n: int = 4) -> float:
  """Fraction of repeated n-grams: 0.0 is clean, high values mean looping."""
  words = (text or "").split()
  if len(words) < n + 1:
    return 0.0
  grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
  counts = collections.Counter(grams)
  repeated = sum(c - 1 for c in counts.values() if c > 1)
  return repeated / len(grams)


def pick_completion(row: dict) -> str:
  for key in ("completion", "response", "generation", "output"):
    value = row.get(key)
    if isinstance(value, str):
      return value
  return ""


# Surface artifacts that a human reader mentally skips but a judge reads as
# "extra content that was not in the provided arguments". These are the usual
# cause of a judge flagging text that looks fine to you.
_ARTIFACTS = {
    "chat_template_tokens": re.compile(
        r"<(start|end)_of_turn>|<\|.*?\|>|\[/?INST\]"
    ),
    "markdown_emphasis": re.compile(r"\*\*|__|^#{1,6}\s", re.MULTILINE),
    "bullet_list": re.compile(r"^\s*[-*•]\s+|^\s*\d+\.\s+", re.MULTILINE),
    "assistant_preamble": re.compile(
        r"^\s*(sure|certainly|here(?:'s| is)|okay|of course|below is)\b",
        re.IGNORECASE,
    ),
    "restates_task": re.compile(
        r"neutral point-of-view answer|user query:|arguments provided:",
        re.IGNORECASE,
    ),
    "leading_whitespace": re.compile(r"\A\s"),
}


def artifact_flags(text: str) -> dict:
  return {
      name: bool(pattern.search(text or ""))
      for name, pattern in _ARTIFACTS.items()
  }


def summarize(name: str, values: List[float]) -> None:
  if not values:
    print(f"  {name:<22}: (none)")
    return
  print(
      f"  {name:<22}: mean={statistics.fmean(values):.3f}"
      f" median={statistics.median(values):.3f}"
      f" min={min(values):.3f} max={max(values):.3f}"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", help="HF repo ID of the scored completions.")
  parser.add_argument("--split", default="test")
  parser.add_argument("--task", default=None, help="Task name, e.g. npov.")
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument("--examples", type=int, default=3)
  args = parser.parse_args()

  ds = load_dataset(args.dataset, split=args.split)
  print(f"Dataset : {args.dataset} ({len(ds)} rows)")
  print(f"Columns : {ds.column_names}")

  if "scores" not in ds.column_names:
    print("\n[WARNING] No 'scores' column: this dataset was never scored, or")
    print("the scores live only in a local checkpoint. Nothing to cross-tab.")
    return

  rows = [dict(row) for row in ds]
  scores: List[Optional[float]] = []
  for row in rows:
    raw = row.get("scores")
    try:
      scores.append(None if raw is None else float(raw))
    except (TypeError, ValueError):
      scores.append(None)

  scored = [(r, s) for r, s in zip(rows, scores) if s is not None]
  flagged = [(r, s) for r, s in scored if s < args.threshold]
  clean = [(r, s) for r, s in scored if s >= args.threshold]
  print(
      f"\nFlagged (score < {args.threshold}): {len(flagged)}/{len(scored)} ="
      f" {100.0 * len(flagged) / max(1, len(scored)):.1f}%"
  )

  print("\n--- Completion health, flagged vs clean ---")
  artifact_rates = {}
  for label, bucket in (("FLAGGED", flagged), ("CLEAN", clean)):
    texts = [pick_completion(r) for r, _ in bucket]
    if not texts:
      continue
    n_empty = sum(1 for t in texts if not t.strip())
    n_trunc = sum(1 for t in texts if looks_truncated(t))
    print(f"\n{label} (n={len(texts)})")
    print(
        f"  empty completions     : {n_empty}"
        f" ({100.0 * n_empty / len(texts):.1f}%)"
    )
    print(
        f"  looks truncated       : {n_trunc}"
        f" ({100.0 * n_trunc / len(texts):.1f}%)"
    )
    summarize("word count", [float(len(t.split())) for t in texts])
    summarize("4-gram repetition", [repetition_ratio(t) for t in texts])
    counts = collections.Counter()
    for text in texts:
      for name, present in artifact_flags(text).items():
        counts[name] += int(present)
    artifact_rates[label] = {
        name: 100.0 * counts[name] / len(texts) for name in _ARTIFACTS
    }

  if artifact_rates:
    print("\n--- Surface artifacts (% of bucket) ---")
    print(f"  {'artifact':<22} {'FLAGGED':>9} {'CLEAN':>9}  {'gap':>8}")
    for name in _ARTIFACTS:
      flagged_pct = artifact_rates.get("FLAGGED", {}).get(name, float("nan"))
      clean_pct = artifact_rates.get("CLEAN", {}).get(name, float("nan"))
      print(
          f"  {name:<22} {flagged_pct:>8.1f}% {clean_pct:>8.1f}%"
          f"  {flagged_pct - clean_pct:>+7.1f}"
      )
    print(
        "\n  A large positive gap means the judge is reacting to formatting,"
        "\n  not to faithfulness. Compare against the reference texts with"
        "\n  scripts/judge_control_run.py before trusting the flagged rate."
    )

  # An empty or degenerate completion is flagged for reasons that have nothing
  # to do with faithfulness, so quantify how much of the flagged mass that is.
  degenerate = [
      1
      for r, _ in flagged
      if not pick_completion(r).strip() or repetition_ratio(pick_completion(r)) > 0.3
  ]
  if flagged:
    print(
        f"\nFlagged rows that are empty or looping: {len(degenerate)}"
        f" ({100.0 * len(degenerate) / len(flagged):.1f}% of flagged)"
    )

  if args.task:
    print("\n--- Exact prompt the judge received ---")
    try:
      processor_cls = get_task_processor(args.task)
      prompt_fn = processor_cls.get_evaluator_prompt()
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"  could not build evaluator prompt: {str(e)[:200]}")
      return
    for label, bucket in (("FLAGGED", flagged), ("CLEAN", clean)):
      for row, score in bucket[: args.examples]:
        try:
          built = prompt_fn(dict(row), fewshot_examples=None)
          text = (
              built.get("evaluator_prompt")
              if isinstance(built, dict)
              else str(built)
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          text = f"(prompt build failed: {str(e)[:120]})"
        print(f"\n===== {label} score={score!r} =====")
        print(re.sub(r"\n{3,}", "\n\n", str(text))[:1200])


if __name__ == "__main__":
  main()
