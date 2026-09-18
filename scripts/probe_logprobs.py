"""Diagnostic probe: does the autorater request shape return logprobs?

Replicates the exact `generate_content` config used by
`EvaluationPipeline.gemini_score_dataset.score_once` and prints the raw
response metadata, so we can tell whether continuous probabilities are
reachable at all on this model/endpoint.

It tries three request shapes:
  1. response_schema + response_logprobs  (what the evaluator sends today)
  2. response_logprobs only               (no controlled decoding)
  3. response_schema only                 (baseline)

If shape 1 returns no logprobs but shape 2 does, controlled decoding is
suppressing them and the evaluator must drop the schema.

Usage (on the VM, from the repo root, with the same env as evaluator.sh):

    python3 -m scripts.probe_logprobs
    # or
    python3 scripts/probe_logprobs.py

Environment (same variables the evaluator uses):
    GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=... GOOGLE_CLOUD_LOCATION=...
    or GEMINI_API_KEY=...
    EVALUATOR_MODEL (optional, defaults to gemini-2.5-flash)
"""

import os

from google import genai
from google.genai import types

MODEL = os.environ.get("EVALUATOR_MODEL", "gemini-2.5-flash")
PROMPT = (
    "Context: The Eiffel Tower is located in Paris, France.\n"
    "Statement: The Eiffel Tower is located in Berlin.\n"
    "Is the statement fully supported by the context? Answer Yes or No."
)
SYSTEM_INSTRUCTION = (
    "You are an evaluator. Output strictly 'Yes' or 'No' with no"
    " conversational preamble or markdown."
)


def make_client() -> genai.Client:
  """Mirrors `EvaluationPipeline.create_gemini_client`."""
  use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
      "true",
      "1",
  ) or bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
  if use_vertex:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    print(f"Client: Vertex AI (project={project}, location={location})")
    return genai.Client(vertexai=True, project=project, location=location)
  if os.environ.get("GEMINI_API_KEY"):
    print("Client: AI Studio API key")
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
  print("Client: default credentials")
  return genai.Client()


def call(client: genai.Client, schema_on: bool, logprobs_on: bool):
  """Issues one request with the requested capability combination."""
  config_kwargs = {
      "temperature": 0,
      "max_output_tokens": 64,
      "seed": 12345,
      "system_instruction": SYSTEM_INSTRUCTION,
  }
  if schema_on:
    config_kwargs["response_mime_type"] = "application/json"
    config_kwargs["response_schema"] = types.Schema(
        type=types.Type.STRING, enum=["No", "Yes"]
    )
  if logprobs_on:
    config_kwargs["response_logprobs"] = True
    config_kwargs["logprobs"] = 5
  try:
    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return client.models.generate_content(
      model=MODEL,
      contents=PROMPT,
      config=types.GenerateContentConfig(**config_kwargs),
  )


def dump(tag: str, response) -> bool:
  """Prints the scoring-relevant metadata. Returns True if logprobs arrived."""
  print(f"\n=== {tag} ===")
  if not response or not response.candidates:
    print("  no candidates (empty or blocked)")
    return False
  candidate = response.candidates[0]
  print("  text        :", repr((response.text or "")[:60]))
  print("  avg_logprobs:", getattr(candidate, "avg_logprobs", "MISSING"))
  result = getattr(candidate, "logprobs_result", None)
  if result is None:
    print("  logprobs_result: None  <-- no continuous score possible")
    return False
  top = getattr(result, "top_candidates", None) or []
  chosen = getattr(result, "chosen_candidates", None) or []
  print(f"  logprobs_result: top_candidates={len(top)} chosen={len(chosen)}")
  saw_value = False
  for i, step in enumerate(top[:4]):
    tokens = []
    for cand in getattr(step, "candidates", None) or []:
      # This is the field the scorer must read; `log_prob` does not exist.
      value = getattr(cand, "log_probability", None)
      saw_value = saw_value or isinstance(value, (int, float))
      tokens.append((getattr(cand, "token", None), value))
    print(f"    step {i}: {tokens}")
  return saw_value


def main() -> None:
  print(f"Model: {MODEL}")
  client = make_client()
  verdict = {}
  for schema_on, logprobs_on in ((True, True), (False, True), (True, False)):
    tag = f"schema={schema_on} logprobs={logprobs_on}"
    try:
      verdict[tag] = dump(tag, call(client, schema_on, logprobs_on))
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"\n=== {tag} ===\n  EXCEPTION: {str(e)[:300]}")
      verdict[tag] = False

  print("\n================ VERDICT ================")
  current = verdict.get("schema=True logprobs=True")
  no_schema = verdict.get("schema=False logprobs=True")
  if current:
    print("Logprobs ARE returned with the current evaluator request shape.")
    print("Continuous autorater scores should work after the parser fix.")
  elif no_schema:
    print("Logprobs are suppressed by `response_schema` (controlled decoding).")
    print("Fix: drop response_schema/response_mime_type when logprobs are on.")
  else:
    print("No logprobs from this model/endpoint in any shape.")
    print("Fix: use self-consistency (k samples, temperature > 0, mean) or a")
    print("model that supports response_logprobs.")


if __name__ == "__main__":
  main()
