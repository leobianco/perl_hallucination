"""Helpers for reading the evaluation stage's multi-model metric map.

The eval stage scores several policies in one go, so its metrics are namespaced
(``sft/hallucination_rate``, ``perl/hallucination_rate``,
``delta/hallucination_rate``). These helpers keep every consumer - the markdown
report, the console scorecard, the live dashboard - reading that map the same
way, including metrics written by older single-model runs.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from src.orchestrator import flavors

#: Bookkeeping entries that are identical by construction; a delta on them is
#: pure noise in the report.
NO_DELTA: FrozenSet[str] = frozenset({"num_samples", "num_examples", "seed"})

#: Metrics where a *lower* value is better. Used both to sign the SFT->PE-RL
#: delta and to decide which branch of a fan-out campaign is the headline.
LOWER_IS_BETTER: FrozenSet[str] = frozenset({
    "hallucination_rate",
    "perplexity",
    "eval_loss",
    "loss",
})

#: Baseline policy every campaign produces, and the display names of the two
#: stage kinds that get evaluated. A branched campaign contributes one
#: ``perl:<flavor>`` label per reward-model dataset flavor on top of these.
BASELINE_LABEL: str = "sft"
KIND_TITLES: Dict[str, str] = {"sft": "SFT", "perl": "PE-RL"}

#: Evaluated policies in report order, with their display names. Retained for
#: callers that only ever deal with unbranched campaigns; prefer
#: :func:`present_targets`, which also sees the branches.
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


def delta_label(target_label: str) -> str:
  """Returns the namespace holding ``target_label``'s improvement over SFT.

  Args:
    target_label: ``perl`` or ``perl:synthetic_struct``.

  Returns:
    ``delta`` or ``delta:synthetic_struct``. The flavor suffix rides along
    unchanged so that a two-branch campaign gets two independent delta
    namespaces rather than one that silently holds whichever branch was
    scored last.
  """
  _, flavor = flavors.split_stage_id(target_label)
  return "delta" if not flavor else f"delta{flavors.SEPARATOR}{flavor}"


def target_title(target_label: str) -> str:
  """Returns the display name of an evaluated policy."""
  kind, flavor = flavors.split_stage_id(target_label)
  base = KIND_TITLES.get(kind, kind.upper())
  return f"{base} ({flavors.flavor_title(flavor)})" if flavor else base


def _target_sort_key(target_label: str) -> Tuple[int, int, str]:
  """Orders policies as SFT first, then PE-RL branches in flavor order."""
  kind, flavor = flavors.split_stage_id(target_label)
  kind_rank = 0 if kind == BASELINE_LABEL else 1
  if flavor in flavors.RM_DATASET_FLAVORS:
    flavor_rank = flavors.RM_DATASET_FLAVORS.index(flavor)
  else:
    flavor_rank = len(flavors.RM_DATASET_FLAVORS)
  return (kind_rank, flavor_rank, flavor or "")


def target_labels(metrics: Dict[str, Any]) -> List[str]:
  """Lists the policy namespaces present in ``metrics``, in report order.

  Discovered from the keys rather than from a fixed list because the report
  is often rendered from a state file alone, with no campaign config to say
  how many branches the run had.

  Args:
    metrics: The eval stage metric map.

  Returns:
    Labels such as ``['sft', 'perl']`` or
    ``['sft', 'perl:organic', 'perl:synthetic_struct']``.
  """
  found = set()
  for key in metrics:
    text = str(key)
    if "/" not in text:
      continue
    label = text.split("/", 1)[0]
    kind, _ = flavors.split_stage_id(label)
    if kind in KIND_TITLES:
      found.add(label)
  return sorted(found, key=_target_sort_key)


def headline_metric(
    metrics: Dict[str, Any], name: str = "hallucination_rate"
) -> Optional[float]:
  """Returns the headline value of ``name`` for the campaign's best policy.

  PE-RL is the campaign's product, so it wins when present; SFT is the
  fallback. When the campaign branched there is no single PE-RL policy, so
  the *best* branch is reported - anything else would make the headline
  depend on which flavor happened to be listed first. Bare and
  ``eval/``-prefixed keys are still honored so reports from earlier
  single-model runs keep rendering.

  Args:
    metrics: The eval stage metric map.
    name: Metric to look up.

  Returns:
    The metric value, or None when absent.
  """
  policies = [
      label for label in target_labels(metrics) if label != BASELINE_LABEL
  ]
  values = []
  for label in policies:
    value = metrics.get(f"{label}/{name}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      values.append(float(value))
  if values:
    return min(values) if name in LOWER_IS_BETTER else max(values)

  for key in (f"{BASELINE_LABEL}/{name}", name, f"eval/{name}"):
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
      return float(value)
  return None


def present_targets(metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
  """Lists the (label, display name) pairs that actually have metrics."""
  return [(label, target_title(label)) for label in target_labels(metrics)]


def metric_names(metrics: Dict[str, Any]) -> List[str]:
  """Returns the bare metric names present, in a stable display order."""
  bare = set()
  labels = target_labels(metrics)
  for key in metrics:
    text = str(key)
    for label in labels:
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
) -> List[Tuple[str, List[Optional[Any]], List[Optional[float]]]]:
  """Pivots the namespaced metrics into per-metric comparison rows.

  Args:
    metrics: The eval stage metric map.

  Returns:
    One ``(metric_name, values, deltas)`` tuple per metric. ``values`` is
    aligned with :func:`present_targets`. ``deltas`` holds one entry per
    *non-baseline* policy, in the same relative order, each the signed
    improvement over SFT (positive is always better) or None when the
    comparison is unavailable.
  """
  targets = target_labels(metrics)
  policies = [label for label in targets if label != BASELINE_LABEL]
  rows = []
  for name in metric_names(metrics):
    values = [metrics.get(f"{label}/{name}") for label in targets]
    deltas: List[Optional[float]] = []
    for label in policies:
      delta = metrics.get(f"{delta_label(label)}/{name}")
      if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        delta = None
      deltas.append(delta)
    rows.append((name, values, deltas))
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


# --- Autorater calibration ----------------------------------------------

#: Metric namespace the ``autorater`` stage writes under.
AUTORATER_PREFIX = "autorater"

#: Calibration values the report shows, in order, with their labels and the
#: format each is worth reading at. The fitted threshold is deliberately
#: unrounded: a judge at temperature 0 saturates, so a threshold of
#: 0.9999887757936129 prints as "1.00000" at five decimals, which is both
#: wrong and unusable as a configuration value.
AUTORATER_REPORT_ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("roc_auc", "ROC-AUC", "{:.4f}"),
    ("best_threshold", "Fitted threshold", "{!r}"),
    ("balanced_accuracy", "Balanced accuracy", "{:.4f}"),
    ("accuracy_at_best_threshold", "Accuracy", "{:.4f}"),
    ("precision_at_best_threshold", "Precision", "{:.4f}"),
    ("tpr_at_best_threshold", "TPR (recall)", "{:.4f}"),
    ("fpr_at_best_threshold", "FPR", "{:.4f}"),
    ("scored_samples", "Scored samples", "{}"),
)


def autorater_calibration(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  """Strips the ``autorater/`` namespace off a calibration metric map.

  Args:
    metrics: The stage's recorded metrics, or None.

  Returns:
    The calibration values by bare name, empty when there are none.
  """
  prefix = f"{AUTORATER_PREFIX}/"
  return {
      str(key)[len(prefix):]: value
      for key, value in (metrics or {}).items()
      if str(key).startswith(prefix)
  }


def format_autorater_rows(
    calibration: Dict[str, Any]
) -> List[Tuple[str, str]]:
  """Renders the calibration as (label, formatted value) pairs.

  Args:
    calibration: Bare-named calibration values, from
      :func:`autorater_calibration`.

  Returns:
    One pair per value actually present, in report order. Absent values are
    skipped rather than shown as a dash: an older state file simply has
    fewer of them, and a row of dashes reads like a failure.
  """
  rows: List[Tuple[str, str]] = []
  for key, label, fmt in AUTORATER_REPORT_ROWS:
    if key not in calibration:
      continue
    value = calibration[key]
    try:
      rows.append((label, fmt.format(value)))
    except (TypeError, ValueError):
      rows.append((label, str(value)))
  return rows
