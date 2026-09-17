"""Helpers for reading the evaluation stage's multi-model metric map.

The eval stage scores several policies in one go, so its metrics are namespaced
(``sft/hallucination_rate``, ``perl/hallucination_rate``,
``delta/hallucination_rate``). These helpers keep every consumer - the markdown
report, the console scorecard, the live dashboard - reading that map the same
way, including metrics written by older single-model runs.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

#: Evaluated policies in report order, with their display names.
TARGETS: Tuple[Tuple[str, str], ...] = (("sft", "SFT"), ("perl", "PE-RL"))

#: Preferred order for the comparison table; anything else is appended.
METRIC_ORDER: Tuple[str, ...] = (
    "hallucination_rate",
    "faithfulness_rate",
    "autorater_accuracy",
    "bertscore_f1",
    "perplexity",
    "num_samples",
)

#: Summary keys with this prefix describe *which model* produced the
#: completions (adapters, stacking, directories) rather than how good they
#: are. They are kept in the metric map for auditing but never tabulated.
PROVENANCE_PREFIX: str = "provenance_"

#: Prefixes of summary keys that audit *how the evaluation ran* rather than
#: how the model performed. Tabulating all of them would bury the four numbers
#: the campaign is actually about.
AUDIT_PREFIXES: Tuple[str, ...] = (PROVENANCE_PREFIX, "autorater_")

#: Audit keys that earn a row anyway: `autorater_accuracy` is a real metric,
#: and the dropped count and judge spread are needed to know whether a delta
#: between two policies means anything.
TABULATED_AUDIT_KEYS: FrozenSet[str] = frozenset({
    "autorater_accuracy",
    "autorater_n_dropped",
    "autorater_spread_mean",
})


def headline_metric(
    metrics: Dict[str, Any], name: str = "hallucination_rate"
) -> Optional[float]:
  """Returns the headline value of ``name`` for the campaign's best policy.

  PE-RL is the campaign's product, so it wins when present; SFT is the
  fallback. Bare and ``eval/``-prefixed keys are still honored so reports from
  earlier single-model runs keep rendering.

  Args:
    metrics: The eval stage metric map.
    name: Metric to look up.

  Returns:
    The metric value, or None when absent.
  """
  for key in (f"perl/{name}", f"sft/{name}", name, f"eval/{name}"):
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      return float(value)
  return None


def present_targets(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
  """Lists the (label, display name) pairs that actually have metrics."""
  return [
      (label, title)
      for label, title in TARGETS
      if any(str(key).startswith(f"{label}/") for key in metrics)
  ]


def metric_names(metrics: Dict[str, Any]) -> List[str]:
  """Returns the bare metric names present, in a stable display order."""
  bare = set()
  for key in metrics:
    text = str(key)
    for label, _ in TARGETS:
      if text.startswith(f"{label}/"):
        name = text[len(label) + 1:]
        # `provenance_*` entries record which adapters produced the
        # completions, and most `autorater_*` entries record how the judge
        # behaved. They are audit trail, not measurements, so they belong in
        # the summary file rather than in the comparison table.
        if name in TABULATED_AUDIT_KEYS or not name.startswith(AUDIT_PREFIXES):
          bare.add(name)
  ordered = [name for name in METRIC_ORDER if name in bare]
  ordered.extend(sorted(bare - set(ordered)))
  return ordered


def comparison_rows(
    metrics: Dict[str, Any],
) -> List[Tuple[str, List[Optional[Any]], Optional[float]]]:
  """Pivots the namespaced metrics into per-metric comparison rows.

  Args:
    metrics: The eval stage metric map.

  Returns:
    One ``(metric_name, [value per present target], delta)`` tuple per metric.
    ``delta`` is the signed PE-RL improvement over SFT (positive is always
    better) and is None when the comparison is unavailable.
  """
  targets = present_targets(metrics)
  rows = []
  for name in metric_names(metrics):
    values = [metrics.get(f"{label}/{name}") for label, _ in targets]
    delta = metrics.get(f"delta/{name}")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
      delta = None
    rows.append((name, values, delta))
  return rows


def format_value(value: Any) -> str:
  """Formats a metric value for display."""
  if value is None:
    return "-"
  if isinstance(value, bool):
    return str(value)
  if isinstance(value, float):
    return f"{value:.4f}"
  return str(value)
