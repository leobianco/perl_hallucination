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
NO_DELTA: FrozenSet[str] = frozenset({
    "num_samples",
    "num_examples",
    "seed",
    # Not a measurement but the condition the measurement was taken under.
    # A "delta" on it would read as if decoding had improved by 0.6.
    "decoding_temperature",
})

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

#: Separates a stage id from the decoding temperature it was scored at, as in
#: ``sft@t0.6``. PE-RL labels are deliberately left bare: a branch is scored at
#: exactly one temperature (its own rollout temperature), so tagging it would
#: rename every existing ``perl/`` key for no new information. Only the extra
#: SFT baselines - the ones evaluated a second time to match a policy - carry
#: the marker, which is also what keeps a single-temperature campaign's metric
#: keys byte-identical to those of every campaign run before this feature.
TEMPERATURE_MARKER: str = "@t"

#: Per-target metric recording the temperature its completions were sampled
#: at. Written for every target so that the report states the decoding regime
#: outright, and so that :func:`baseline_label_for` can pair a policy with the
#: baseline measured under the same regime without consulting the config.
DECODING_TEMPERATURE_KEY: str = "decoding_temperature"

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
    "decoding_temperature",
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


def format_temperature(value: float) -> str:
  """Renders a temperature as the canonical token used in labels and titles.

  Args:
    value: The sampling temperature.

  Returns:
    A short decimal string, at least one place after the point, with no
    trailing zeros beyond that: 0.6, 1.0, 0.25. Used both to build labels and
    to compare them, so that two temperatures that print the same are treated
    as the same regime regardless of float round-tripping.
  """
  text = f"{float(value):.4f}".rstrip("0")
  if text.endswith("."):
    text += "0"
  return text


def make_target_label(stage_id: str, temperature: Optional[float]) -> str:
  """Tags ``stage_id`` with the temperature it is being scored at.

  Args:
    stage_id: The bare stage id, e.g. ``sft``.
    temperature: The decoding temperature, or None to leave the label bare.

  Returns:
    ``sft@t0.6``, or ``stage_id`` unchanged when ``temperature`` is None.
  """
  if temperature is None:
    return stage_id
  return f"{stage_id}{TEMPERATURE_MARKER}{format_temperature(temperature)}"


def split_target_label(label: str) -> Tuple[str, Optional[float]]:
  """Splits a target label into its stage id and temperature tag.

  Args:
    label: A target label, tagged or not.

  Returns:
    ``('sft', 0.6)`` for ``sft@t0.6``; ``('perl:organic', None)`` for an
    untagged label. An unparseable tag is returned as no tag rather than
    raising: a metric map is sometimes hand-edited, and a malformed label
    should cost its temperature row, not the whole report.
  """
  if TEMPERATURE_MARKER not in label:
    return label, None
  stage_id, _, tag = label.partition(TEMPERATURE_MARKER)
  try:
    return stage_id, float(tag)
  except ValueError:
    return label, None


def target_kind(label: str) -> str:
  """Returns the stage kind of a target label, ignoring flavor and tag."""
  stage_id, _ = split_target_label(label)
  kind, _ = flavors.split_stage_id(stage_id)
  return kind


def is_baseline(label: str) -> bool:
  """True when ``label`` is an SFT row rather than a trained policy.

  Every SFT row is a baseline, including the extra ones evaluated at a
  policy's rollout temperature. Getting this wrong gives ``sft@t0.6`` a delta
  column against itself.

  Args:
    label: A target label.

  Returns:
    Whether the label names the campaign's baseline.
  """
  return target_kind(label) == BASELINE_LABEL


def baseline_label_for(
    metrics: Dict[str, Any], label: str
) -> Optional[str]:
  """Returns the baseline ``label`` should be compared against.

  A policy is compared against the SFT row sampled at the *same* temperature,
  so that the difference between them is the weights and nothing else. When
  no matching baseline was evaluated - an older state file, or a campaign run
  with temperature matching switched off - the plain ``sft`` row is used and
  the comparison silently spans two decoding regimes, which is exactly what
  the extra baselines exist to avoid.

  Args:
    metrics: The eval stage metric map.
    label: The policy to find a baseline for.

  Returns:
    The baseline's label, or None when the campaign has no SFT row at all.
  """
  baselines = [name for name in target_labels(metrics) if is_baseline(name)]
  if not baselines:
    return None
  own = metrics.get(f"{label}/{DECODING_TEMPERATURE_KEY}")
  if isinstance(own, (int, float)) and not isinstance(own, bool):
    wanted = format_temperature(own)
    for name in baselines:
      theirs = metrics.get(f"{name}/{DECODING_TEMPERATURE_KEY}")
      if isinstance(theirs, (int, float)) and not isinstance(theirs, bool):
        if format_temperature(theirs) == wanted:
          return name
  return BASELINE_LABEL if BASELINE_LABEL in baselines else baselines[0]


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
  stage_id, _ = split_target_label(target_label)
  _, flavor = flavors.split_stage_id(stage_id)
  return "delta" if not flavor else f"delta{flavors.SEPARATOR}{flavor}"


def target_title(target_label: str) -> str:
  """Returns the display name of an evaluated policy."""
  stage_id, temperature = split_target_label(target_label)
  kind, flavor = flavors.split_stage_id(stage_id)
  base = KIND_TITLES.get(kind, kind.upper())
  qualifiers = []
  if flavor:
    qualifiers.append(flavors.flavor_title(flavor))
  if temperature is not None:
    qualifiers.append(f"T={format_temperature(temperature)}")
  return f"{base} ({', '.join(qualifiers)})" if qualifiers else base


def _target_sort_key(target_label: str) -> Tuple[int, float, int, str]:
  """Orders SFT rows first (untagged, then by temperature), then PE-RL.

  Args:
    target_label: The label to rank.

  Returns:
    A sort key placing every baseline ahead of every policy, the untagged
    baseline ahead of its warmer siblings, and the branches in flavor order.
  """
  stage_id, temperature = split_target_label(target_label)
  kind, flavor = flavors.split_stage_id(stage_id)
  kind_rank = 0 if kind == BASELINE_LABEL else 1
  # The untagged row is the campaign's nominal baseline and leads the table;
  # -1.0 sorts it ahead of any real temperature, including 0.0.
  temperature_rank = -1.0 if temperature is None else float(temperature)
  if flavor in flavors.RM_DATASET_FLAVORS:
    flavor_rank = flavors.RM_DATASET_FLAVORS.index(flavor)
  else:
    flavor_rank = len(flavors.RM_DATASET_FLAVORS)
  return (kind_rank, temperature_rank, flavor_rank, flavor or "")


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
    if target_kind(label) in KIND_TITLES:
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
      label for label in target_labels(metrics) if not is_baseline(label)
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
  policies = [label for label in targets if not is_baseline(label)]
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


def format_row_value(name: str, value: Any) -> str:
  """Formats a value for the comparison table, given which metric it is.

  Everything in that table is a measurement rendered to four decimals, except
  the decoding temperature, which is a *setting*. Printing it as ``0.3000``
  invites the reader to treat it as something that was measured.

  Args:
    name: The bare metric name.
    value: The value to render.

  Returns:
    The formatted value.
  """
  if name == DECODING_TEMPERATURE_KEY and isinstance(value, (int, float)):
    if not isinstance(value, bool):
      return format_temperature(value)
  return format_value(value)


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
