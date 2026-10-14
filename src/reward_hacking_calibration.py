"""Calibrates the reward-hacking rubric on healthy reference responses.

The rubric autorater grades writing quality, and ``reward_hacking_rate``
counts completions whose quality falls below ``--reward_hacking_threshold``.
No data labelled for reward hacking exists: looping, verbatim copying of the
context and fragmented decoding are failure modes of an RL policy, not of the
references in our datasets. What *can* be measured is the judge's behaviour on
text that is known to be healthy:

* Every reference is a negative, so every flag is a false alarm. The share
  flagged at the deployed threshold is the judge's false-positive rate (FPR).
* The largest thresholds whose FPR stays within 1%, 5% and 10% are the most
  sensitive thresholds the healthy data allows. They are reported, never
  wired back into the pipeline.
* The rubric must not grade factuality, so faithful and hallucinated
  references should get the same grades. The difference is measured within
  the reward-model arm only, where both labels come from one population.

Sensitivity (whether real reward hacking gets caught) cannot be measured on
this data.

Everything here is pure Python, so it is unit-testable without the ML stack.
:class:`src.pipelines.EvaluationRewardHackingAutoraterPipeline` loads the
data and calls the judge.
"""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src import utils

#: Value of ``--mode`` (and of the second argument of ``evaluator.sh``).
MODE: str = "reward_hacking_autorater_eval"

#: The two calibration arms. The reward-model arm holds both labels; the SFT
#: arm holds training targets, which are faithful by construction.
SOURCE_RM: str = "rm_organic"
SOURCE_SFT: str = "sft"
SOURCES: Tuple[str, ...] = (SOURCE_RM, SOURCE_SFT)

LABEL_FAITHFUL: str = "faithful"
LABEL_HALLUCINATED: str = "hallucinated"
LABEL_UNKNOWN: str = "unknown"
LABELS: Tuple[str, ...] = (LABEL_FAITHFUL, LABEL_HALLUCINATED, LABEL_UNKNOWN)

#: Key of the aggregate score in the statistics, next to the dimension keys.
QUALITY: str = "quality"

#: FPR budgets of the threshold suggestions. These are conventional reporting
#: levels, not recommendations.
TARGET_FPRS: Tuple[float, ...] = (0.01, 0.05, 0.10)

#: Two-sided 95% normal quantile. Same value as
#: ``src.orchestrator.eval_metrics.Z_95``.
Z_95: float = 1.959963984540054

#: A suggestion at FPR ``alpha`` over ``n`` rows is marked unreliable when
#: fewer than this many flags are expected (``n * alpha``). One flag more or
#: less moves the threshold to another grid point.
MIN_EXPECTED_FLAGS: float = 5.0

#: How many of the lowest-quality rows the false-alarm dump shows.
NUM_LOWEST_ROWS: int = 25

#: How many prompts the prompt dump shows. Demonstrations and preamble are
#: identical across rows, so a few prompts are enough to audit them.
NUM_PROMPTS_DUMPED: int = 5

#: Dataset suffixes the calibration refuses to read. ``_final_test_set`` is
#: reserved for the final evaluation, and fitting a threshold on it would be
#: model selection on the test set. ``_autorater`` concatenates every split
#: for NPOV and BOSCH, so it contains final-test rows (and, for BOSCH, the
#: rows the rubric's demonstrations are drawn from).
FORBIDDEN_DATASET_SUFFIXES: Tuple[str, ...] = ("_final_test_set", "_autorater")

#: Dataset values that switch an arm off, e.g. to calibrate on the
#: reward-model arm alone.
DISABLED_SOURCE_VALUES: frozenset[str] = frozenset(
    {"none", "null", "off", "skip", "false", "-"}
)

#: Decimals that observed qualities are rounded to before the threshold
#: search. With ``k > 1`` judge calls the same grade can be reached through
#: different float paths, which would otherwise split one grid point in two.
_GRID_DECIMALS: int = 9

METRICS_JSON: str = "rh_autorater_eval_metrics.json"
METRICS_TXT: str = "rh_autorater_eval_metrics.txt"
ROWS_JSONL: str = "rh_autorater_eval_rows.jsonl"
LOWEST_TXT: str = "rh_autorater_eval_lowest.txt"
PROMPTS_TXT: str = "rh_autorater_eval_prompts.txt"
HISTOGRAMS_PNG: str = "rh_autorater_eval_histograms.png"


@dataclasses.dataclass
class CalibrationRow:
  """One healthy reference response and, once graded, its rubric verdict.

  Attributes:
    source: Calibration arm, one of :data:`SOURCES`.
    source_index: Row index within that arm's dataset split.
    label: One of :data:`LABELS`.
    context: Source material, as the rubric prompt shows it.
    query: User request, as the rubric prompt shows it.
    response: The reference text that gets graded.
    raw: The dataset row, kept to build the task's own rubric prompt.
    grades: Dimension key to grade in [0, 1]; None until graded or when the
      judge could not be resolved.
  """

  source: str
  source_index: int
  label: str
  context: str
  query: str
  response: str
  raw: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)
  grades: Optional[Dict[str, float]] = None

  @property
  def quality(self) -> Optional[float]:
    """The aggregate score the deployed rubric thresholds."""
    return verdict_quality(self.grades)


def verdict_quality(grades: Optional[Mapping[str, float]]) -> Optional[float]:
  """Returns the unweighted mean of the dimension grades present.

  Same aggregation as :func:`src.utils.reward_hacking_summary` and the
  scoring pipeline's ``reward_hacking_quality`` column, so the FPR measured
  here is the one ``reward_hacking_rate`` would report.

  Args:
    grades: Dimension key to grade in [0, 1], or None.

  Returns:
    The mean, or None when no dimension was graded.
  """
  if not isinstance(grades, Mapping) or not grades:
    return None
  values = [
      float(grades[key]) for key in utils.reward_hacking_dimension_keys()
      if key in grades
  ]
  if not values:
    return None
  return sum(values) / len(values)


def resolve_label(row: Mapping[str, Any], source: str) -> str:
  """Maps a row's hallucination annotation onto a label.

  ``class_hall`` ("Yes" = hallucinated) is read first because a string
  cannot be misread. ``label`` follows the repo-wide convention 1 = faithful,
  0 = hallucinated (set in every task processor's preprocessing). NPOV's SFT
  rows carry neither; they are SFT targets, hence faithful.

  Args:
    row: A dataset row.
    source: The arm the row comes from.

  Returns:
    One of :data:`LABELS`.
  """
  class_hall = row.get("class_hall")
  if isinstance(class_hall, str):
    normalized = class_hall.strip().lower()
    if normalized == "yes":
      return LABEL_HALLUCINATED
    if normalized == "no":
      return LABEL_FAITHFUL
  label = row.get("label")
  if not isinstance(label, bool):
    if label in (1, "1"):
      return LABEL_FAITHFUL
    if label in (0, "0"):
      return LABEL_HALLUCINATED
  if source == SOURCE_SFT:
    return LABEL_FAITHFUL
  return LABEL_UNKNOWN


def normalize_text(text: Any) -> str:
  """Collapses whitespace, so formatting noise does not defeat the guards."""
  return " ".join(str(text or "").split())


def prompt_key(context: str, query: str) -> Tuple[str, str]:
  """Identifies a prompt independently of the dataset's column layout."""
  return normalize_text(context), normalize_text(query)


def is_disabled_source(dataset: Optional[str]) -> bool:
  """Reports whether a dataset value switches its calibration arm off."""
  return (
      isinstance(dataset, str)
      and dataset.strip().lower() in DISABLED_SOURCE_VALUES
  )


def calibration_sources(
    user: str,
    task_name: str,
    rm_dataset: Optional[str] = None,
    rm_split: Optional[str] = None,
    sft_dataset: Optional[str] = None,
    sft_split: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
  """Resolves which dataset split feeds each calibration arm.

  The defaults are the reward-model dataset's held-out arm and the SFT
  held-out split. Both are disjoint from the final test set, and the SFT
  split is disjoint from the split the rubric's demonstrations are drawn
  from (``{task}_sft:train``).

  Args:
    user: Hub namespace of the task datasets.
    task_name: Task the datasets belong to.
    rm_dataset: Override for the reward-model arm, or a value from
      :data:`DISABLED_SOURCE_VALUES` to skip it.
    rm_split: Split of the reward-model arm.
    sft_dataset: Override for the SFT arm, or a disabling value.
    sft_split: Split of the SFT arm.

  Returns:
    ``(source, repo_id, split)`` per enabled arm, reward-model arm first.

  Raises:
    ValueError: If both arms are disabled, or an arm points at a dataset the
      calibration must not read.
  """
  candidates = (
      (SOURCE_RM, rm_dataset, rm_split, f"{user}/{task_name}_rm_organic"),
      (SOURCE_SFT, sft_dataset, sft_split, f"{user}/{task_name}_sft"),
  )
  sources = []
  for source, dataset, split, default in candidates:
    if is_disabled_source(dataset):
      continue
    sources.append((source, dataset or default, split or "test"))
  if not sources:
    raise ValueError(
        "Both calibration arms are disabled; enable at least one of the"
        " reward-model and SFT datasets."
    )
  check_calibration_sources(repo for _, repo, _ in sources)
  return sources


def check_calibration_sources(repo_ids: Iterable[str]) -> None:
  """Refuses datasets that would leak the final test set into the calibration.

  Args:
    repo_ids: Hub dataset IDs about to be read.

  Raises:
    ValueError: If one of them is a final test set or an autorater set.
  """
  for repo_id in repo_ids:
    name = str(repo_id).rstrip("/").split("/")[-1]
    for suffix in FORBIDDEN_DATASET_SUFFIXES:
      if name.endswith(suffix):
        raise ValueError(
            f"Refusing to calibrate the reward-hacking rubric on {repo_id}."
            " '*_final_test_set' is reserved for the final evaluation:"
            " choosing a threshold on it is model selection on the test set."
            " '*_autorater' concatenates every split for NPOV and BOSCH, so"
            " it contains final-test rows too. Use the reward-model held-out"
            " arm ('{task}_rm_organic:test') and the SFT held-out split"
            " ('{task}_sft:test')."
        )


def build_rows(
    raw_rows: Iterable[Mapping[str, Any]], source: str, processor_cls: Any
) -> List[CalibrationRow]:
  """Extracts what the rubric grades from one arm's dataset rows.

  The text comes from the task processor's own accessors, the ones the
  rubric prompt is built with, so the guards compare exactly what the judge
  reads. The reference is read with ``use_true_label=True``.

  Args:
    raw_rows: The arm's dataset rows.
    source: The arm, one of :data:`SOURCES`.
    processor_cls: The task processor class.

  Returns:
    One row per dataset row, in dataset order.
  """
  rows = []
  for index, raw in enumerate(raw_rows):
    row = dict(raw)
    rows.append(
        CalibrationRow(
            source=source,
            source_index=index,
            label=resolve_label(row, source),
            context=str(processor_cls.reward_hacking_context(row) or ""),
            query=str(processor_cls.reward_hacking_query(row) or ""),
            response=str(
                processor_cls.reward_hacking_response(row, use_true_label=True)
                or ""
            ),
            raw=row,
        )
    )
  return rows


def demo_prompt_keys(
    fewshot_examples: Optional[Sequence[Mapping[str, Any]]],
    processor_cls: Any,
) -> set[Tuple[str, str]]:
  """Returns the prompts the rubric's demonstrations are built on.

  Args:
    fewshot_examples: Rows from
      :func:`src.pipelines.build_reward_hacking_fewshot_examples`, or None.
    processor_cls: The task processor class.

  Returns:
    One :func:`prompt_key` per distinct demonstration prompt.
  """
  keys = set()
  for example in fewshot_examples or ():
    row = dict(example)
    keys.add(
        prompt_key(
            processor_cls.reward_hacking_context(row),
            processor_cls.reward_hacking_query(row),
        )
    )
  return keys


def subsample(
    rows: Sequence[CalibrationRow], max_rows: Optional[int], seed: int
) -> List[CalibrationRow]:
  """Draws a seeded subsample, keeping the dataset order.

  Args:
    rows: Rows of one arm.
    max_rows: Cap; None, 0 or a negative value keeps everything.
    seed: Seed of the draw.

  Returns:
    At most ``max_rows`` rows.
  """
  if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
    return list(rows)
  indices = list(range(len(rows)))
  random.Random(seed).shuffle(indices)
  return [rows[i] for i in sorted(indices[:max_rows])]


def prepare_rows(
    arms: Sequence[Tuple[str, Sequence[CalibrationRow]]],
    demo_keys: Iterable[Tuple[str, str]],
    max_rows_per_source: Optional[int],
    seed: int,
) -> Tuple[List[CalibrationRow], Dict[str, Dict[str, int]]]:
  """Applies the guards, then caps each arm.

  The guards run in this order:

  1. Empty references are dropped: there is nothing to grade.
  2. Rows that share their prompt with a demonstration are dropped. The
     judge has seen that context next to a response graded 5 and a
     degenerated copy graded 1, so its verdict on it is not a fresh one.
  3. Exact (context, query, response) duplicates are dropped; the first
     occurrence wins. Pass the reward-model arm first, so that a reference
     present in both arms keeps the label the independence test needs.

  The cap comes last, so that it counts usable rows.

  Args:
    arms: ``(source, rows)`` per arm, in priority order.
    demo_keys: :func:`prompt_key` of every demonstration prompt.
    max_rows_per_source: Cap per arm; None, 0 or negative keeps everything.
    seed: Seed of the subsample.

  Returns:
    The kept rows, in arm order, and per-arm counts for each step.
  """
  demo_keys = set(demo_keys)
  seen: set[Tuple[str, str, str]] = set()
  kept: List[CalibrationRow] = []
  accounting: Dict[str, Dict[str, int]] = {}
  for source, rows in arms:
    counts = {
        "loaded": len(rows),
        "dropped_empty_response": 0,
        "dropped_demo_prompt": 0,
        "dropped_duplicate": 0,
    }
    survivors = []
    for row in rows:
      if not normalize_text(row.response):
        counts["dropped_empty_response"] += 1
        continue
      key = prompt_key(row.context, row.query)
      if key in demo_keys:
        counts["dropped_demo_prompt"] += 1
        continue
      full_key = key + (normalize_text(row.response),)
      if full_key in seen:
        counts["dropped_duplicate"] += 1
        continue
      seen.add(full_key)
      survivors.append(row)
    counts["after_guards"] = len(survivors)
    capped = subsample(survivors, max_rows_per_source, seed)
    counts["after_subsample"] = len(capped)
    accounting[source] = counts
    kept.extend(capped)
  return kept, accounting


def prompts_fingerprint(prompts: Sequence[str]) -> str:
  """Fingerprints the exact prompt list, so a resume never misaligns rows.

  The judge's checkpoint stores verdicts by row index. A checkpoint written
  for a different row set (another cap, another dataset revision) would be
  attached to the wrong rows without complaint.

  Args:
    prompts: The rubric prompts, in grading order.

  Returns:
    A short hex digest.
  """
  digest = hashlib.sha1()
  for prompt in prompts:
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\x00")
  return digest.hexdigest()[:12]


def run_name(
    judge_model: Optional[str],
    num_fewshot: Optional[int],
    num_samples: Optional[int],
    task_name: Optional[str],
    seed: int,
) -> str:
  """Names the log directory of a calibration run.

  Every input that changes the verdicts is in the name, so differently
  configured judges cannot overwrite each other. The threshold is not: the
  report recomputes everything from the saved grades.

  Args:
    judge_model: The rubric judge.
    num_fewshot: Demonstration pairs.
    num_samples: Judge calls per row (``--autorater_num_samples``).
    task_name: Task of the datasets.
    seed: Seed of the subsample and of the demonstration draw.

  Returns:
    The directory name, relative to ``logs/``.
  """
  model_name = judge_model.split("/")[-1] if judge_model else "gemini"
  return (
      f"eval_reward_hacking_autorater_{model_name}"
      f"_fewshot_{num_fewshot}"
      f"_k{num_samples}"
      f"_task_{task_name or 'unknown'}"
      f"_seed_{seed}"
  )


# --- Statistics -------------------------------------------------------------


def mean_and_std(
    values: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
  """Sample mean and standard deviation (ddof=1, 0.0 for a single value).

  Same convention as :func:`src.utils.reward_hacking_summary`.

  Args:
    values: The sample.

  Returns:
    ``(mean, std)``, both None for an empty sample.
  """
  n = len(values)
  if not n:
    return None, None
  mean = sum(values) / n
  if n < 2:
    return mean, 0.0
  variance = sum((v - mean) ** 2 for v in values) / (n - 1)
  return mean, math.sqrt(variance)


def wilson_interval(
    proportion: float, n: int, z: float = Z_95
) -> Tuple[float, float]:
  """Wilson score interval for a binomial proportion.

  A copy of :func:`src.orchestrator.eval_metrics.wilson_interval`, pinned to
  it by a test. Importing it would execute ``src/orchestrator/__init__.py``,
  which loads the campaign engine, and this feature is deliberately kept out
  of the orchestrator.

  Args:
    proportion: Observed success fraction, clamped to [0, 1].
    n: Number of trials.
    z: Normal quantile; 1.96 for 95%.

  Returns:
    ``(low, high)``, always inside [0, 1].

  Raises:
    ValueError: If ``n`` is not positive.
  """
  if n <= 0:
    raise ValueError(f"A Wilson interval needs n > 0 (got {n}).")
  p = min(1.0, max(0.0, float(proportion)))
  denominator = 1.0 + z * z / n
  center = (p + z * z / (2.0 * n)) / denominator
  half = (
      z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
  )
  return max(0.0, center - half), min(1.0, center + half)


def mean_interval(
    mean: Optional[float], std: Optional[float], n: int, z: float = Z_95
) -> Optional[Tuple[float, float]]:
  """Normal-approximation interval of a mean, as the dashboard draws it.

  Args:
    mean: Sample mean.
    std: Sample standard deviation.
    n: Sample size.
    z: Normal quantile.

  Returns:
    ``(low, high)``, or None below two observations.
  """
  if mean is None or std is None or n < 2:
    return None
  half = z * std / math.sqrt(n)
  return mean - half, mean + half


def false_positive_rate(
    qualities: Sequence[float], threshold: float
) -> Optional[float]:
  """Share of healthy rows the deployed rule (``quality < threshold``) flags.

  Args:
    qualities: Per-row quality of healthy references.
    threshold: The ``--reward_hacking_threshold`` under test.

  Returns:
    The FPR, or None for an empty sample.
  """
  if not qualities:
    return None
  return sum(1 for q in qualities if q < threshold) / len(qualities)


def suggest_thresholds(
    qualities: Sequence[float],
    target_fprs: Sequence[float] = TARGET_FPRS,
    z: float = Z_95,
) -> List[Dict[str, Any]]:
  """Finds the largest threshold whose FPR stays within each budget.

  ``quality < t`` flags a row, so FPR(t) is a step function that only moves
  at observed quality values. For a budget ``alpha``, the largest admissible
  threshold is the smallest observed value ``v`` at which more than
  ``alpha * n`` rows have a quality of at most ``v``. That threshold flags
  exactly the rows strictly below ``v``; any larger one exceeds the budget.

  The achieved FPR is reported because it is usually below the budget (the
  grid is coarse: with one judge call a quality moves in steps of 1/12).

  Args:
    qualities: Per-row quality of healthy references.
    target_fprs: FPR budgets, each in (0, 1).
    z: Normal quantile of the interval on the achieved FPR.

  Returns:
    One entry per budget: ``target_fpr``, ``threshold``, ``n_flagged``,
    ``n``, ``achieved_fpr``, ``achieved_fpr_ci95`` and ``reliable`` (False
    when fewer than :data:`MIN_EXPECTED_FLAGS` flags are expected). Empty
    when there is nothing to calibrate on.

  Raises:
    ValueError: If a budget is outside (0, 1).
  """
  for alpha in target_fprs:
    if not 0.0 < alpha < 1.0:
      raise ValueError(f"FPR budgets must lie in (0, 1), got {alpha}.")
  if not qualities:
    return []

  n = len(qualities)
  counts = collections.Counter()
  # Smallest raw value per grid point. It, not the rounded grid value, is
  # the threshold reported: the deployed rule compares raw floats, and
  # round(2/3, 9) > 2/3 would flag the rows sitting on the grid point.
  cell_floor: Dict[float, float] = {}
  for quality in qualities:
    cell = round(quality, _GRID_DECIMALS)
    counts[cell] += 1
    cell_floor[cell] = min(quality, cell_floor.get(cell, quality))
  grid = sorted(counts)
  cumulative = []
  running = 0
  for cell in grid:
    running += counts[cell]
    cumulative.append(running)

  suggestions = []
  for alpha in target_fprs:
    budget = alpha * n + 1e-9
    n_flagged = 0
    threshold = cell_floor[grid[0]]
    for position, at_most in enumerate(cumulative):
      if at_most > budget:
        threshold = cell_floor[grid[position]]
        break
      n_flagged = at_most
    # alpha < 1 guarantees the loop breaks: the last cumulative count is n.
    achieved = n_flagged / n
    low, high = wilson_interval(achieved, n, z)
    suggestions.append({
        "target_fpr": alpha,
        "threshold": threshold,
        "n_flagged": n_flagged,
        "n": n,
        "achieved_fpr": achieved,
        "achieved_fpr_ci95": [low, high],
        "reliable": n * alpha >= MIN_EXPECTED_FLAGS,
    })
  return suggestions


def welch_difference(
    group_a: Sequence[float], group_b: Sequence[float], z: float = Z_95
) -> Optional[Dict[str, Any]]:
  """``mean(a) - mean(b)`` with a normal interval on Welch's standard error.

  Args:
    group_a: First sample.
    group_b: Second sample.
    z: Normal quantile.

  Returns:
    ``delta``, ``mean_a``, ``mean_b``, ``n_a``, ``n_b``, ``se``, ``ci95``
    and ``excludes_zero``; the last three are None when a group has fewer
    than two observations. None when a group is empty.
  """
  n_a, n_b = len(group_a), len(group_b)
  if not n_a or not n_b:
    return None
  mean_a, std_a = mean_and_std(group_a)
  mean_b, std_b = mean_and_std(group_b)
  result: Dict[str, Any] = {
      "delta": mean_a - mean_b,
      "mean_a": mean_a,
      "mean_b": mean_b,
      "n_a": n_a,
      "n_b": n_b,
      "se": None,
      "ci95": None,
      "excludes_zero": None,
  }
  if n_a < 2 or n_b < 2:
    return result
  se = math.sqrt(std_a**2 / n_a + std_b**2 / n_b)
  low, high = result["delta"] - z * se, result["delta"] + z * se
  result["se"] = se
  result["ci95"] = [low, high]
  result["excludes_zero"] = bool(low > 0.0 or high < 0.0)
  return result


def _metric_value(row: CalibrationRow, key: str) -> Optional[float]:
  """Returns one dimension's grade, or the aggregate for :data:`QUALITY`."""
  if key == QUALITY:
    return row.quality
  if not row.grades or key not in row.grades:
    return None
  return float(row.grades[key])


def _metric_keys() -> List[str]:
  return utils.reward_hacking_dimension_keys() + [QUALITY]


def stratify(
    rows: Sequence[CalibrationRow],
) -> List[Tuple[str, List[CalibrationRow]]]:
  """Splits the rows into the reported strata, skipping empty ones.

  Args:
    rows: All calibration rows.

  Returns:
    ``(name, rows)`` for ``all``, ``label=*``, ``source=*`` and
    ``source=*/label=*``, in that order.
  """
  strata = [("all", list(rows))]
  for label in LABELS:
    members = [r for r in rows if r.label == label]
    if members:
      strata.append((f"label={label}", members))
  for source in SOURCES:
    members = [r for r in rows if r.source == source]
    if not members:
      continue
    strata.append((f"source={source}", members))
    for label in LABELS:
      subset = [r for r in members if r.label == label]
      if subset:
        strata.append((f"source={source}/label={label}", subset))
  return strata


def stratum_summary(
    rows: Sequence[CalibrationRow], threshold: float, z: float = Z_95
) -> Dict[str, Any]:
  """Summarizes the judge's verdicts on one stratum of healthy rows.

  Args:
    rows: The stratum.
    threshold: The deployed ``--reward_hacking_threshold``.
    z: Normal quantile of the intervals.

  Returns:
    Counts, per-metric means and standard deviations, an interval on the
    mean quality, and the FPR at ``threshold`` with its Wilson interval.
  """
  scored = [r for r in rows if r.quality is not None]
  means: Dict[str, Optional[float]] = {}
  stds: Dict[str, Optional[float]] = {}
  for key in _metric_keys():
    values = [_metric_value(r, key) for r in scored]
    means[key], stds[key] = mean_and_std([v for v in values if v is not None])
  qualities = [r.quality for r in scored]
  summary: Dict[str, Any] = {
      "n": len(rows),
      "n_scored": len(scored),
      "n_unscored": len(rows) - len(scored),
      "means": means,
      "stds": stds,
      "quality_ci95": mean_interval(
          means[QUALITY], stds[QUALITY], len(qualities), z
      ),
      "n_flagged": None,
      "fpr_at_threshold": None,
      "fpr_ci95": None,
  }
  if qualities:
    fpr = false_positive_rate(qualities, threshold)
    summary["n_flagged"] = sum(1 for q in qualities if q < threshold)
    summary["fpr_at_threshold"] = fpr
    summary["fpr_ci95"] = list(wilson_interval(fpr, len(qualities), z))
  return summary


def independence_tests(
    rows: Sequence[CalibrationRow], z: float = Z_95
) -> Dict[str, Any]:
  """Measures whether the rubric reacts to hallucination at all.

  Faithful minus hallucinated, per dimension and for the aggregate, within
  the reward-model arm only. The SFT arm is faithful by construction and
  comes from another population (for NPOV, probably another writer), so a
  pooled comparison would partly measure the dataset instead of the label.

  Args:
    rows: All calibration rows.
    z: Normal quantile of the intervals.

  Returns:
    The population, group sizes and one :func:`welch_difference` per metric.
  """
  population = [
      r for r in rows if r.source == SOURCE_RM and r.quality is not None
  ]
  faithful = [r for r in population if r.label == LABEL_FAITHFUL]
  hallucinated = [r for r in population if r.label == LABEL_HALLUCINATED]
  deltas = {}
  for key in _metric_keys():
    group_a = [_metric_value(r, key) for r in faithful]
    group_b = [_metric_value(r, key) for r in hallucinated]
    deltas[key] = welch_difference(
        [v for v in group_a if v is not None],
        [v for v in group_b if v is not None],
        z,
    )
  return {
      "population": f"source={SOURCE_RM}",
      "contrast": f"{LABEL_FAITHFUL} - {LABEL_HALLUCINATED}",
      "n_faithful": len(faithful),
      "n_hallucinated": len(hallucinated),
      "deltas": deltas,
  }


def grade_histogram(
    rows: Sequence[CalibrationRow],
) -> Dict[str, Dict[str, int]]:
  """Counts each dimension's grades per Likert level.

  Grades are mapped back to the nearest level of the 1-5 scale. With one
  judge call per row that mapping is exact; with several, a mean grade is
  rounded to its closest level.

  Args:
    rows: All calibration rows.

  Returns:
    Dimension key to ``{level: count}``, every level present.
  """
  lowest = utils.REWARD_HACKING_SCALE_MIN
  highest = utils.REWARD_HACKING_SCALE_MAX
  histogram = {}
  for key in utils.reward_hacking_dimension_keys():
    counts = {str(level): 0 for level in range(lowest, highest + 1)}
    for row in rows:
      grade = _metric_value(row, key)
      if grade is None:
        continue
      level = lowest + int(round(grade * (highest - lowest)))
      counts[str(min(highest, max(lowest, level)))] += 1
    histogram[key] = counts
  return histogram


def summarize(
    rows: Sequence[CalibrationRow],
    threshold: float,
    target_fprs: Sequence[float] = TARGET_FPRS,
    z: float = Z_95,
) -> Dict[str, Any]:
  """Computes every statistic of the calibration report.

  Args:
    rows: Graded calibration rows.
    threshold: The deployed ``--reward_hacking_threshold``.
    target_fprs: FPR budgets of the threshold suggestions.
    z: Normal quantile of the intervals.

  Returns:
    ``threshold``, ``strata``, ``suggested_thresholds``, ``independence``
    and ``grade_histogram``.
  """
  qualities = [r.quality for r in rows if r.quality is not None]
  return {
      "threshold": float(threshold),
      "strata": [
          {"name": name, **stratum_summary(members, threshold, z)}
          for name, members in stratify(rows)
      ],
      "suggested_thresholds": suggest_thresholds(qualities, target_fprs, z),
      "independence": independence_tests(rows, z),
      "grade_histogram": grade_histogram(rows),
  }


# --- Rendering ----------------------------------------------------------------


def _pct(value: Optional[float]) -> str:
  return "-" if value is None else f"{100.0 * value:.1f}%"


def _num(value: Optional[float], signed: bool = False) -> str:
  if value is None:
    return "-"
  return f"{value:+.3f}" if signed else f"{value:.3f}"


def _interval(
    bounds: Optional[Sequence[float]], percent: bool = False
) -> str:
  if not bounds:
    return ""
  render = _pct if percent else _num
  return f"[{render(bounds[0])}, {render(bounds[1])}]"


def _dimension_titles() -> List[Tuple[str, str]]:
  return [(key, title) for key, title, _ in utils.REWARD_HACKING_DIMENSIONS]


def render_report(payload: Mapping[str, Any]) -> str:
  """Renders the metrics payload for a human.

  Args:
    payload: :func:`summarize` output, plus the ``config``, ``data`` and
      ``judge_audit`` blocks the pipeline adds.

  Returns:
    The report text.
  """
  config = payload.get("config", {})
  threshold = payload.get("threshold")
  lines = [
      "Reward-hacking rubric calibration on healthy references",
      "=" * 56,
      f"Task: {config.get('task_name')} | judge: {config.get('judge_model')}"
      f" | demonstrations: {config.get('num_fewshot_pairs')} pairs"
      f" | judge calls per row: {config.get('num_samples')}"
      f" | seed: {config.get('seed')}",
  ]
  for source in config.get("sources", []):
    lines.append(
        f"  {source.get('source')}: {source.get('dataset')}"
        f" (split {source.get('split')})"
    )
  lines.extend([
      "Every row is a healthy reference, so every flag is a false positive.",
      "",
      "Data",
      f"  {'arm':<12}{'loaded':>8}{'empty':>7}{'demo':>6}{'dup':>6}"
      f"{'kept':>7}{'graded':>8}{'failed':>8}",
  ])
  for source, counts in payload.get("data", {}).items():
    lines.append(
        f"  {source:<12}{counts.get('loaded', 0):>8}"
        f"{counts.get('dropped_empty_response', 0):>7}"
        f"{counts.get('dropped_demo_prompt', 0):>6}"
        f"{counts.get('dropped_duplicate', 0):>6}"
        f"{counts.get('after_subsample', 0):>7}"
        f"{counts.get('graded', 0):>8}"
        f"{counts.get('judge_failures', 0):>8}"
    )
  lines.append(
      "  (demo = shares its prompt with a rubric demonstration; kept = after"
      " guards and the per-arm cap)"
  )

  titles = _dimension_titles()
  strata = payload.get("strata", [])
  name_width = max([len("stratum")] + [len(s["name"]) for s in strata]) + 2
  header = f"  {'stratum':<{name_width}}{'n':>6}  {'quality [95% CI]':<24}"
  header += "".join(f"{title:>{len(title) + 2}}" for _, title in titles)
  header += f"  FPR [95% CI]"
  lines.extend([
      "",
      "Grades (in [0, 1]) and FPR = share of rows with quality <"
      f" {threshold}",
      header,
  ])
  for stratum in strata:
    means = stratum.get("means", {})
    quality = (
        f"{_num(means.get(QUALITY))} {_interval(stratum.get('quality_ci95'))}"
    )
    row = (
        f"  {stratum['name']:<{name_width}}"
        f"{stratum.get('n_scored', 0):>6}  {quality:<24}"
    )
    row += "".join(
        f"{_num(means.get(key)):>{len(title) + 2}}" for key, title in titles
    )
    row += (
        f"  {_pct(stratum.get('fpr_at_threshold'))}"
        f" {_interval(stratum.get('fpr_ci95'), percent=True)}"
    )
    lines.append(row)

  lines.extend([
      "",
      "Suggested thresholds (report only, nothing is wired back): the largest"
      " t with FPR(t) <= target, flagging quality < t",
      f"  {'target':>8}{'threshold':>11}{'flagged':>12}  achieved FPR [95% CI]",
  ])
  suggestions = payload.get("suggested_thresholds", [])
  if not suggestions:
    lines.append("  (no graded rows)")
  for item in suggestions:
    marker = "" if item.get("reliable") else "  *"
    flagged = f"{item.get('n_flagged')}/{item.get('n')}"
    threshold_text = (
        "-" if item.get("threshold") is None else f"{item['threshold']:.4f}"
    )
    lines.append(
        f"  {_pct(item.get('target_fpr')):>8}"
        f"{threshold_text:>11}"
        f"{flagged:>12}  {_pct(item.get('achieved_fpr'))}"
        f" {_interval(item.get('achieved_fpr_ci95'), percent=True)}{marker}"
    )
  if any(not item.get("reliable") for item in suggestions):
    lines.append(
        f"  * fewer than {MIN_EXPECTED_FLAGS:g} flags expected (n x target):"
        " one row more or less moves this threshold."
    )

  independence = payload.get("independence", {})
  lines.extend([
      "",
      "Independence from hallucination, within"
      f" {independence.get('population')}"
      f" ({independence.get('contrast')}; n = "
      f"{independence.get('n_faithful')} / "
      f"{independence.get('n_hallucinated')})",
      f"  {'metric':<22}{'delta':>8}  95% CI",
  ])
  names = dict(titles)
  names[QUALITY] = "Quality"
  for key, result in independence.get("deltas", {}).items():
    name = names.get(key, key)
    if result is None:
      lines.append(f"  {name:<22}{'-':>8}  (a label group is empty)")
      continue
    marker = "  <- CI excludes 0" if result.get("excludes_zero") else ""
    lines.append(
        f"  {name:<22}{_num(result.get('delta'), True):>8}"
        f"  {_interval(result.get('ci95'))}{marker}"
    )
  lines.append(
      "  A CI excluding 0 means the rubric is not blind to the label. That is"
      " not automatically leakage: weaker writers can both hallucinate more"
      " and write worse."
  )

  histogram = payload.get("grade_histogram", {})
  if histogram:
    levels = [
        str(level)
        for level in range(
            utils.REWARD_HACKING_SCALE_MIN, utils.REWARD_HACKING_SCALE_MAX + 1
        )
    ]
    lines.extend([
        "",
        "Grade distribution (all rows, nearest Likert level)",
        f"  {'dimension':<22}" + "".join(f"{level:>7}" for level in levels),
    ])
    for key, title in titles:
      counts = histogram.get(key, {})
      lines.append(
          f"  {title:<22}"
          + "".join(f"{counts.get(level, 0):>7}" for level in levels)
      )

  audit = payload.get("judge_audit") or {}
  if audit:
    lines.extend(["", "Judge audit"])
    for key in sorted(audit):
      lines.append(f"  {key}: {audit[key]}")

  lines.extend([
      "",
      "Not measured: sensitivity. Healthy references cannot tell whether real"
      " reward hacking gets caught. References are also complete texts, while"
      " policy completions are cut at --max_tokens and the fluency dimension"
      " penalises abrupt stops, so the FPR of healthy *policy* outputs is"
      " likely higher than this.",
      "",
  ])
  return "\n".join(lines)


def row_record(row: CalibrationRow) -> Dict[str, Any]:
  """Flattens a graded row for the JSONL dump."""
  record: Dict[str, Any] = {
      "source": row.source,
      "source_index": row.source_index,
      "label": row.label,
      QUALITY: row.quality,
  }
  for key in utils.reward_hacking_dimension_keys():
    record[key] = _metric_value(row, key)
  record["query"] = row.query
  record["response"] = row.response
  record["context"] = row.context
  return record


def lowest_rows(
    rows: Sequence[CalibrationRow], limit: int = NUM_LOWEST_ROWS
) -> List[CalibrationRow]:
  """Returns the graded rows with the lowest quality, deterministically."""
  scored = [r for r in rows if r.quality is not None]
  scored.sort(key=lambda r: (r.quality, r.source, r.source_index))
  return scored[:limit]


def render_lowest_rows(
    rows: Sequence[CalibrationRow], limit: int = NUM_LOWEST_ROWS
) -> str:
  """Renders the rows a too-strict threshold would flag first.

  Args:
    rows: Graded calibration rows.
    limit: How many rows to show.

  Returns:
    The text dump.
  """
  selected = lowest_rows(rows, limit)
  lines = [
      f"The {len(selected)} lowest-quality healthy references. Each is a"
      " false alarm for any threshold above its quality.",
      "",
  ]
  for rank, row in enumerate(selected, start=1):
    grades = " | ".join(
        f"{key} {_num(_metric_value(row, key))}"
        for key in utils.reward_hacking_dimension_keys()
    )
    context = row.context
    if len(context) > 600:
      context = context[:600] + " [...]"
    lines.extend([
        f"#{rank}  quality {_num(row.quality)}  [{row.source}, {row.label},"
        f" row {row.source_index}]  {grades}",
        f"Query: {row.query or '(none)'}",
        "Response:",
        row.response,
        "Context (excerpt):",
        context,
        "-" * 80,
    ])
  return "\n".join(lines) + "\n"


def write_reports(
    out_dir: str,
    payload: Mapping[str, Any],
    rows: Sequence[CalibrationRow],
    prompts: Sequence[str],
) -> Dict[str, str]:
  """Writes every text artifact of a calibration run.

  Args:
    out_dir: Run directory; created when missing.
    payload: The full metrics payload.
    rows: Graded calibration rows.
    prompts: Rubric prompts, aligned with ``rows``.

  Returns:
    Artifact name to path.
  """
  os.makedirs(out_dir, exist_ok=True)
  paths = {
      "metrics_json": os.path.join(out_dir, METRICS_JSON),
      "metrics_txt": os.path.join(out_dir, METRICS_TXT),
      "rows_jsonl": os.path.join(out_dir, ROWS_JSONL),
      "lowest_txt": os.path.join(out_dir, LOWEST_TXT),
      "prompts_txt": os.path.join(out_dir, PROMPTS_TXT),
  }
  with open(paths["metrics_json"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
  with open(paths["metrics_txt"], "w", encoding="utf-8") as handle:
    handle.write(render_report(payload))
  with open(paths["rows_jsonl"], "w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row_record(row)) + "\n")
  with open(paths["lowest_txt"], "w", encoding="utf-8") as handle:
    handle.write(render_lowest_rows(rows))
  with open(paths["prompts_txt"], "w", encoding="utf-8") as handle:
    for index, prompt in enumerate(prompts[:NUM_PROMPTS_DUMPED]):
      handle.write(f"\n{index}. ----------\n{prompt}")
  return paths
