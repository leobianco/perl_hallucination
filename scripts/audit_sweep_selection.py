#!/usr/bin/env python3
"""Explains, run by run, why the orchestrator picked the winner it picked.

The orchestrator ranks a sweep with one number per trial and then materializes
the winner. When that winner looks arbitrary, the question is always one of:

  1. Which runs were even eligible? (crashed and killed runs are dropped, and
     unfinished ones are only used when *nothing* finished.)
  2. What number was each eligible run scored on? Under
     ``selection_strategy="final"`` that is the single last logged value, which
     for PE-RL is one optimizer step's mean over ``num_generations`` samples -
     a very noisy point estimate.
  3. Is that number comparable across trials at all? It is not, whenever a
     swept hyperparameter changes the definition of the metric. PE-RL sweeps
     ``reward_penalty_alpha``, which multiplies negative rewards, so a trial
     that drew a smaller alpha is scored on an easier scale.

This script answers all three against the live W&B sweep, reusing the
orchestrator's own resolution helpers so the reported pick is the real one and
not a re-implementation that might disagree.

Usage:
  python3 scripts/audit_sweep_selection.py --sweep <entity/project/sweep_id>
  python3 scripts/audit_sweep_selection.py --sweep abc12345 --tail 10
"""

import argparse
import collections
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

from src.orchestrator.sweep_controller import _coerce_float
from src.orchestrator.sweep_controller import _iter_history_rows
from src.orchestrator.sweep_controller import _resolve_metric_key

#: The states :meth:`SweepController._fetch_best_run_once` treats as eligible.
#: Anything outside these two tuples is dropped silently, which is exactly the
#: behaviour this script exists to make visible.
FINISHED_STATES = ("finished",)
FALLBACK_STATES = ("running", "failed")

#: Hyperparameters that change what the reward *means* rather than how well
#: the policy performs. Ranking trials that disagree on any of these compares
#: scores measured with different rulers.
SCALE_CHANGING_PARAMS = ("reward_penalty_alpha",)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--sweep",
      required=True,
      help="Sweep id, either bare or fully qualified as entity/project/id.",
  )
  parser.add_argument(
      "--entity", default="leobianco", help="W&B entity for a bare sweep id."
  )
  parser.add_argument(
      "--project", default="new_perl", help="W&B project for a bare sweep id."
  )
  parser.add_argument(
      "--metric",
      default="train/rewards/reward_fn/mean",
      help="Metric the sweep ranks on.",
  )
  parser.add_argument(
      "--goal", default="maximize", choices=("maximize", "minimize")
  )
  parser.add_argument(
      "--tail",
      type=int,
      default=10,
      help=(
          "How many trailing logged points to average for the alternative "
          "'tail mean' ranking. This is the robust counterpart of the single "
          "last point the orchestrator uses."
      ),
  )
  return parser.parse_args(argv)


def qualify(sweep: str, entity: str, project: str) -> str:
  """Returns ``sweep`` as a fully qualified ``entity/project/id`` path."""
  return sweep if sweep.count("/") >= 2 else f"{entity}/{project}/{sweep}"


def tail_mean(rows: Optional[Sequence[Any]], key: str, n: int) -> Optional[float]:
  """Averages the last ``n`` usable values of ``key`` in ``rows``.

  Args:
    rows: History rows as returned by the orchestrator's reader, or None.
    key: The resolved metric key.
    n: Window size.

  Returns:
    The mean, or None when the history held no usable value.
  """
  if not rows:
    return None
  values = [
      value
      for row in rows
      if isinstance(row, dict)
      for value in (_coerce_float(row.get(key)),)
      if value is not None
  ]
  if not values:
    return None
  return statistics.fmean(values[-n:])


def describe_run(run: Any, metric: str, tail: int) -> Dict[str, Any]:
  """Collects everything needed to explain one trial's score."""
  summary = dict(getattr(run, "summary", {}) or {})
  config = dict(getattr(run, "config", {}) or {})
  key = _resolve_metric_key(summary, metric)
  rows = _iter_history_rows(run, key) if key else None
  values = (
      [
          value
          for row in rows or []
          if isinstance(row, dict)
          for value in (_coerce_float(row.get(key)),)
          if value is not None
      ]
      if key
      else []
  )
  return {
      "id": run.id,
      "name": getattr(run, "name", "?"),
      "state": getattr(run, "state", "?"),
      "key": key,
      "score": _coerce_float(summary.get(key)) if key else None,
      "tail_mean": tail_mean(rows, key, tail) if key else None,
      "points": len(values),
      "spread": (max(values) - min(values)) if len(values) > 1 else None,
      "config": config,
  }


def eligible(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
  """Applies the orchestrator's finished-runs-first eligibility rule."""
  scored = [r for r in records if r["score"] is not None]
  finished = [r for r in scored if r["state"] in FINISHED_STATES]
  if finished:
    return finished
  return [r for r in scored if r["state"] in FALLBACK_STATES]


def rank(records: List[Dict[str, Any]], field: str, goal: str) -> List[Dict[str, Any]]:
  """Sorts ``records`` best-first on ``field``."""
  usable = [r for r in records if r[field] is not None]
  return sorted(usable, key=lambda r: r[field], reverse=goal == "maximize")


def fmt(value: Optional[float], width: int = 9) -> str:
  """Formats an optional float for the table."""
  return f"{value:{width}.5f}" if value is not None else " " * (width - 1) + "-"


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Prints the audit. Returns a process exit code."""
  args = parse_args(argv)
  import wandb  # pylint: disable=g-import-not-at-top

  path = qualify(args.sweep, args.entity, args.project)
  sweep = wandb.Api().sweep(path)
  runs = list(sweep.runs)
  if not runs:
    print(f"Sweep {path} has no runs.")
    return 1

  records = [describe_run(run, args.metric, args.tail) for run in runs]
  chosen = eligible(records)
  chosen_ids = {r["id"] for r in chosen}

  print(f"\nSweep {path}")
  print(f"Metric '{args.metric}', goal={args.goal}, selection=final\n")

  print("ALL RUNS")
  header = (
      f"{'run id':10} {'state':9} {'elig':5} {'final':>9} "
      f"{'tail@' + str(args.tail):>9} {'pts':>5} {'spread':>9}  name"
  )
  print(header)
  print("-" * len(header))
  for record in sorted(records, key=lambda r: str(r["state"])):
    print(
        f"{record['id']:10} {str(record['state']):9} "
        f"{'yes' if record['id'] in chosen_ids else 'NO':5} "
        f"{fmt(record['score'])} {fmt(record['tail_mean'])} "
        f"{record['points']:5d} {fmt(record['spread'])}  {record['name']}"
    )

  dropped = [r for r in records if r["id"] not in chosen_ids]
  if dropped:
    print(
        f"\n{len(dropped)} run(s) were NOT eligible. A run is dropped when it "
        "never logged the metric, or when its state is anything other than "
        f"{FINISHED_STATES + FALLBACK_STATES} (e.g. 'crashed', 'killed')."
    )

  keys = {r["key"] for r in chosen}
  if len(keys) > 1:
    print(
        f"\n!! The eligible runs were scored on DIFFERENT keys: {sorted(keys)}. "
        "Those numbers are not comparable; the ranking below is meaningless."
    )

  by_final = rank(chosen, "score", args.goal)
  by_tail = rank(chosen, "tail_mean", args.goal)
  if not by_final:
    print("\nNo eligible run carried a usable score.")
    return 1

  winner = by_final[0]
  print(f"\nORCHESTRATOR PICK: {winner['id']} ({winner['name']})")
  print(f"  scored {fmt(winner['score'])} = its single last logged point")
  if winner["spread"] is not None:
    print(
        f"  that run's metric spanned {winner['spread']:.5f} across "
        f"{winner['points']} logged points, so the last point is worth "
        "roughly nothing on its own"
    )
  if by_tail and by_tail[0]["id"] != winner["id"]:
    print(
        f"  ranking on the tail mean instead would have picked "
        f"{by_tail[0]['id']} ({by_tail[0]['name']}) - the pick is an artifact "
        "of where each run happened to stop"
    )

  for param in SCALE_CHANGING_PARAMS:
    groups = collections.defaultdict(list)
    for record in chosen:
      groups[record["config"].get(param)].append(record)
    if len(groups) <= 1:
      continue
    print(
        f"\n!! '{param}' VARIES ACROSS TRIALS ({sorted(map(str, groups))}). "
        "It changes the reward's definition, not the policy's quality, so "
        "these scores were measured with different rulers:"
    )
    for value, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
      scores = [m["score"] for m in members if m["score"] is not None]
      mean = statistics.fmean(scores) if scores else float("nan")
      print(f"   {param}={value!s:6} n={len(members)}  mean final={mean:.5f}")
    print(
        f"   The winner ran with {param}="
        f"{winner['config'].get(param)!s}."
    )

  print("\nCONFIG OF THE PICK")
  for name in sorted(winner["config"]):
    if not name.startswith("_"):
      print(f"   {name} = {winner['config'][name]}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
