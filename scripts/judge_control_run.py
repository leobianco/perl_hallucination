"""Control experiment: score known-good reference texts with the same judge.

The judge's operating characteristics (TPR/FPR) were measured on human-written
reference texts. If it now flags 70%+ of model generations, either the
generations really are that bad, or the judge is being pushed outside the
regime it was calibrated in (surface-form shift: chat scaffolding, markdown,
preamble, truncation).

This script separates those two hypotheses by running the *same* judge, with
the *same* prompt template and few-shot examples, over the *reference*
responses of the same rows (`use_true_label=True`). Those texts are known
good, so:

    flag rate on references ~= judge FPR (a few %)   -> judge is fine,
                                                        investigate generations
    flag rate on references >> judge FPR             -> the prompt/judge setup
                                                        is broken, the
                                                        generation numbers are
                                                        not interpretable

Usage (on the VM, same env as scripts/evaluator.sh):

    python3 scripts/judge_control_run.py <completions_repo_id> \
        --task npov --n 100 [--also-generations]

Note: the request shape and the logprob parsing are duplicated from
`EvaluationPipeline` on purpose, so this diagnostic does not drag in torch /
vLLM and cannot be perturbed by pipeline changes under investigation.
"""

import argparse
import concurrent.futures
import math
import os
import random
from typing import Optional

from datasets import load_dataset
from google import genai
from google.genai import types

from src.utils import get_task_processor

SYSTEM_INSTRUCTION = (
    "You are an evaluator. Output strictly 'Yes' or 'No' with no"
    " conversational preamble or markdown."
)


def make_client() -> genai.Client:
  """Same resolution order as evaluator.sh (Vertex by default)."""
  use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() in (
      "true",
      "1",
  ) or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
  if use_vertex:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )
  return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def score_prompt(
    client: genai.Client, model: str, prompt: str, seed: int
) -> Optional[float]:
  """Returns P(No) i.e. P(faithful), or None when unresolved."""
  config = types.GenerateContentConfig(
      temperature=0,
      max_output_tokens=64,
      seed=seed,
      system_instruction=SYSTEM_INSTRUCTION,
      response_mime_type="application/json",
      response_schema=types.Schema(type=types.Type.STRING, enum=["No", "Yes"]),
      response_logprobs=True,
      logprobs=5,
      thinking_config=types.ThinkingConfig(thinking_budget=0),
  )
  try:
    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"  [call failed] {str(e)[:120]}")
    return None
  if not response or not response.candidates:
    return None
  candidate = response.candidates[0]
  result = getattr(candidate, "logprobs_result", None)
  for step in (getattr(result, "top_candidates", None) or []) if result else []:
    logp_no = logp_yes = None
    for cand in getattr(step, "candidates", None) or []:
      token = (getattr(cand, "token", "") or "").strip().strip("\"'`").lower()
      # The SDK field is `log_probability`; reading `log_prob` yields None and
      # silently collapses every score to a hard 0/1.
      value = getattr(cand, "log_probability", None)
      if not isinstance(value, (int, float)):
        continue
      if token == "no":
        logp_no = float(value)
      elif token == "yes":
        logp_yes = float(value)
    if logp_no is not None and logp_yes is not None:
      p_no, p_yes = math.exp(logp_no), math.exp(logp_yes)
      if p_no + p_yes > 0:
        return p_no / (p_no + p_yes)
    if logp_no is not None:
      return min(1.0, math.exp(logp_no))
    if logp_yes is not None:
      return max(0.0, 1.0 - math.exp(logp_yes))
  text = ((response.text or "").strip().strip("\"'` \n\r\t").lower())
  if text.startswith("no"):
    return 1.0
  if text.startswith("yes"):
    return 0.0
  return None


def run(client, model, prompts, seed, workers, label) -> None:
  scores = [None] * len(prompts)
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {
        pool.submit(score_prompt, client, model, p, seed): i
        for i, p in enumerate(prompts)
    }
    for future in concurrent.futures.as_completed(futures):
      scores[futures[future]] = future.result()
  valid = [s for s in scores if s is not None]
  if not valid:
    print(f"\n{label}: no scores returned.")
    return
  flagged = sum(1 for s in valid if s < 0.5)
  continuous = sum(1 for s in valid if s not in (0.0, 1.0))
  print(f"\n{label}")
  print(f"  scored          : {len(valid)}/{len(prompts)}")
  print(
      f"  flagged (Yes)   : {flagged} ="
      f" {100.0 * flagged / len(valid):.1f}%"
  )
  print(f"  continuous      : {continuous}/{len(valid)}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dataset", help="HF repo ID with completions.")
  parser.add_argument("--split", default="test")
  parser.add_argument("--task", default="npov")
  parser.add_argument("--model", default=os.environ.get("EVALUATOR_MODEL", "gemini-2.5-flash"))
  parser.add_argument("--n", type=int, default=100)
  parser.add_argument("--seed", type=int, default=12345)
  parser.add_argument("--workers", type=int, default=16)
  parser.add_argument(
      "--also-generations",
      action="store_true",
      help="Score the generations too, as a same-run comparison.",
  )
  args = parser.parse_args()

  ds = load_dataset(args.dataset, split=args.split)
  rows = [dict(r) for r in ds]
  rng = random.Random(args.seed)
  rng.shuffle(rows)
  rows = rows[: args.n]
  print(f"Dataset: {args.dataset} | sampled {len(rows)} rows | model {args.model}")

  prompt_fn = get_task_processor(args.task).get_evaluator_prompt()
  client = make_client()

  # No few-shot examples here on purpose: this isolates the judge's reaction to
  # the *text*, without the extra variable of which few-shot rows were drawn.
  ref_prompts, gen_prompts = [], []
  for row in rows:
    ref = prompt_fn(dict(row), fewshot_examples=None, use_true_label=True)
    gen = prompt_fn(dict(row), fewshot_examples=None, use_true_label=False)
    ref_prompts.append(ref["evaluator_prompt"])
    gen_prompts.append(gen["evaluator_prompt"])

  run(client, args.model, ref_prompts, args.seed, args.workers,
      "REFERENCE texts (human-written, known good)")
  if args.also_generations:
    run(client, args.model, gen_prompts, args.seed, args.workers,
        "MODEL generations (same rows, same judge)")

  print(
      "\nInterpretation: if the reference flag rate is far above the judge's"
      " measured FPR, the judge setup is at fault and the generation numbers"
      " are not interpretable. If it is low, the judge is behaving and the"
      " difference is genuinely in the generations."
  )


if __name__ == "__main__":
  main()
