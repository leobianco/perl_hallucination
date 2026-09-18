"""Post-hoc analysis of a saved autorater run.

Answers the question the ROC cannot: *is this judge a ranker or a classifier?*
If the score mass is concentrated at the endpoints, the "best threshold" from
`compute_best_roc_threshold` is fitted noise and the AUC is just balanced
accuracy. In that regime you should report classifier-agreement metrics at a
fixed threshold instead.

Reads the two files written by `mode=autoratereval`:
    logs/<run>/eval_autorater_ground_truth.txt
    logs/<run>/eval_autorater_scores.txt

Usage:
    python3 scripts/analyze_autorater_scores.py logs/<run>
    python3 scripts/analyze_autorater_scores.py logs/<run> --threshold 0.5

Note: runs produced before the full-precision fix wrote scores rounded to 5
decimals, which collapses the saturated cluster into a single value. Such a
file can still show *how many* samples are saturated, but the exact tie counts
are a lower bound. Re-run scoring to get full precision.
"""

import argparse
import math
import os
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def read_scores(path: str) -> List[Optional[float]]:
  values = []
  with open(path, "r") as f:
    for line in f:
      token = line.strip()
      if not token or token.lower() in ("none", "nan", "null"):
        values.append(None)
        continue
      try:
        value = float(token)
      except ValueError:
        values.append(None)
        continue
      values.append(None if math.isnan(value) else value)
  return values


def read_labels(path: str) -> List[int]:
  with open(path, "r") as f:
    return [int(float(line.strip())) for line in f if line.strip()]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  """Wilson score interval: sane even when the rate is near 0 or 1."""
  if n == 0:
    return (float("nan"), float("nan"))
  p = k / n
  denom = 1.0 + z * z / n
  center = (p + z * z / (2 * n)) / denom
  margin = (
      z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
  )
  return (max(0.0, center - margin), min(1.0, center + margin))


def report_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> None:
  """Prints agreement metrics for a fixed decision threshold."""
  pred = (scores >= threshold).astype(int)
  tp = int(np.sum((pred == 1) & (labels == 1)))
  fp = int(np.sum((pred == 1) & (labels == 0)))
  tn = int(np.sum((pred == 0) & (labels == 0)))
  fn = int(np.sum((pred == 0) & (labels == 1)))
  tpr = tp / (tp + fn) if (tp + fn) else float("nan")
  fpr = fp / (fp + tn) if (fp + tn) else float("nan")
  print(f"\n  threshold = {threshold!r}")
  print(f"    TP/FP/TN/FN : {tp}/{fp}/{tn}/{fn}")
  print(f"    accuracy    : {accuracy_score(labels, pred):.5f}")
  print(f"    precision   : {precision_score(labels, pred, zero_division=0):.5f}")
  print(f"    recall (TPR): {recall_score(labels, pred, zero_division=0):.5f}")
  print(f"    FPR         : {fpr:.5f}")
  print(f"    F1          : {f1_score(labels, pred, zero_division=0):.5f}")
  print(f"    balanced acc: {(tpr + (1.0 - fpr)) / 2.0:.5f}")
  print(f"    Cohen kappa : {cohen_kappa_score(labels, pred):.5f}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dir", help="logs/<run> directory")
  parser.add_argument(
      "--threshold",
      type=float,
      default=0.5,
      help="Fixed threshold to report alongside the ROC-fitted one.",
  )
  args = parser.parse_args()

  scores_path = os.path.join(args.run_dir, "eval_autorater_scores.txt")
  labels_path = os.path.join(args.run_dir, "eval_autorater_ground_truth.txt")
  raw_scores = read_scores(scores_path)
  raw_labels = read_labels(labels_path)
  if len(raw_scores) != len(raw_labels):
    raise SystemExit(
        f"length mismatch: {len(raw_scores)} scores vs {len(raw_labels)} labels"
    )

  keep = [i for i, s in enumerate(raw_scores) if s is not None]
  scores = np.array([raw_scores[i] for i in keep], dtype=np.float64)
  labels = np.array([raw_labels[i] for i in keep], dtype=int)
  n = len(scores)
  print(f"Scored samples : {n}/{len(raw_scores)}")
  print(f"Positives      : {int(labels.sum())}  Negatives: {int(n - labels.sum())}")

  print("\n--- Score resolution ---")
  n_unique = int(np.unique(scores).size)
  n_one = int(np.sum(scores == 1.0))
  n_zero = int(np.sum(scores == 0.0))
  n_sat = int(np.sum((scores > 1.0 - 1e-4) | (scores < 1e-4)))
  print(f"Distinct values      : {n_unique}/{n}")
  print(f"Exactly 1.0 / 0.0    : {n_one} / {n_zero}")
  print(f"Within 1e-4 of 0 or 1: {n_sat} ({100.0 * n_sat / n:.1f}%)")
  interior = scores[(scores > 1e-4) & (scores < 1.0 - 1e-4)]
  print(f"Interior scores      : {interior.size}")
  if interior.size:
    print(
        "  quantiles 10/50/90 :"
        f" {np.quantile(interior, 0.1):.4f} /"
        f" {np.quantile(interior, 0.5):.4f} /"
        f" {np.quantile(interior, 0.9):.4f}"
    )

  auc = roc_auc_score(labels, scores)
  # Margin ranking: a monotone transform of P(No) that does not saturate in
  # float64. If it moves the AUC, the probabilities were losing ties to
  # rounding; if it does not, the judge genuinely carries no ranking signal.
  eps = 1e-300
  margin = np.log(np.clip(scores, eps, 1.0)) - np.log(
      np.clip(1.0 - scores, eps, 1.0)
  )
  auc_margin = roc_auc_score(labels, margin)
  print("\n--- Ranking quality ---")
  print(f"AUC (probability)    : {auc:.5f}")
  print(f"AUC (logit margin)   : {auc_margin:.5f}")

  print("\n--- Operating points ---")
  report_at(labels, scores, args.threshold)

  from sklearn.metrics import roc_curve  # pylint: disable=g-import-not-at-top

  fpr_curve, tpr_curve, thresholds = roc_curve(labels, scores)
  finite = np.isfinite(thresholds)
  best_idx = np.where(finite)[0][
      int(np.argmax(tpr_curve[finite] - fpr_curve[finite]))
  ]
  fitted = float(thresholds[best_idx])
  report_at(labels, scores, fitted)

  print("\n--- Hallucination rate (autorater decision) ---")
  flagged = int(np.sum(scores < args.threshold))
  low, high = wilson_interval(flagged, n)
  print(
      f"Flagged at {args.threshold}: {flagged}/{n} ="
      f" {100.0 * flagged / n:.2f}%  (95% Wilson CI"
      f" {100.0 * low:.2f}% - {100.0 * high:.2f}%)"
  )

  print("\n--- Verdict ---")
  tpr_b = tpr_curve[best_idx]
  fpr_b = fpr_curve[best_idx]
  balanced = (tpr_b + (1.0 - fpr_b)) / 2.0
  print(f"AUC - balanced accuracy = {auc - balanced:+.6f}")
  print(f"AUC(margin) - AUC(prob) = {auc_margin - auc:+.6f}")
  if abs(auc - balanced) < 1e-5 and abs(auc_margin - balanced) < 1e-5:
    print("The judge is a CLASSIFIER, not a ranker: AUC == balanced accuracy")
    print("even under a non-saturating monotone transform. Report agreement")
    print("metrics at a fixed threshold (0.5); do not tune or report an AUC.")
  elif auc_margin - auc > 1e-4:
    print("Float saturation was costing you ranking signal: the logit margin")
    print("scores higher than the probability. Rank on the margin instead.")
  else:
    print("The score carries genuine ranking information; a tuned threshold")
    print("is meaningful. Validate it on a held-out split before reporting.")


if __name__ == "__main__":
  main()
