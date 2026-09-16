"""Decides which checkpoints a training run publishes to the Hugging Face Hub.

Every training stage produces two checkpoints worth keeping: the one that
scored best on the evaluation metric, and the one at the final optimizer step.
Which of the two a stage considers canonical differs - SFT and the reward
model want the best one, PE-RL wants the final one, because a peak training
reward is one lucky batch away from being a collapsed policy.

Rather than discarding the other, a run publishes:

* the canonical checkpoint at the repository **root**, so a bare
  ``from_pretrained(repo_id)`` keeps returning the model the campaign chose;
* the other one under the ``best/`` or ``last/`` **subfolder**, reachable via
  ``from_pretrained(repo_id, subfolder="best")`` or the ``repo_id:best``
  reference syntax understood by ``pipelines.parse_hf_repo_reference``;
* a ``checkpoints.json`` manifest recording which is which, at which step,
  and under which metric.

This module holds the part of that decision that is pure bookkeeping, so it
can be exercised without importing the training stack.
"""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any, Dict, Optional

#: Subfolder names used for the non-canonical checkpoint.
BEST_CHECKPOINT_SUBFOLDER = "best"
LAST_CHECKPOINT_SUBFOLDER = "last"

#: Manifest written at the repository root.
CHECKPOINT_MANIFEST_FILENAME = "checkpoints.json"

#: Training state that is useless for evaluation or for resuming from the Hub
#: and would dominate the upload size. A LoRA adapter is a few dozen MB; the
#: optimizer state and RNG dumps next to it are not worth the bandwidth.
CHECKPOINT_UPLOAD_IGNORE_PATTERNS = (
    "optimizer.pt",
    "optimizer_*.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "*.distcp",
    "global_step*/*",
    "latest",
    "zero_to_fp32.py",
)

_CHECKPOINT_DIR_RE = re.compile(r"checkpoint-(\d+)$")


def step_from_checkpoint_dir(path: Optional[str]) -> Optional[int]:
  """Extracts the global step encoded in a ``checkpoint-N`` directory name.

  Args:
    path: A path or bare directory name, with or without a trailing slash.

  Returns:
    The step, or None if the name does not carry one.
  """
  if not path:
    return None
  match = _CHECKPOINT_DIR_RE.search(str(path).rstrip("/"))
  if not match:
    return None
  try:
    return int(match.group(1))
  except (TypeError, ValueError):
    return None


def coerce_step(value: Any) -> Optional[int]:
  """Converts a trainer-state step into a positive int, or None.

  ``TrainerState.global_step`` and ``best_global_step`` are normally ints, but
  they are attributes on an object we do not own: a stubbed or partially
  initialised state hands back something that is not comparable to an int.
  Publication is a best-effort step that runs *after* the model was pushed, so
  it must degrade rather than raise.

  Args:
    value: Whatever the trainer state holds.

  Returns:
    The step if it is a positive integer, otherwise None.
  """
  # Anything that merely *implements* __int__ is refused on purpose: a mock
  # or a partially initialised state would otherwise coerce to 1 and send us
  # looking for a "checkpoint-1" that never existed.
  if isinstance(value, bool) or value is None:
    return None
  if isinstance(value, int):
    step = value
  elif isinstance(value, float):
    if not math.isfinite(value):
      return None
    step = int(value)
  elif isinstance(value, str):
    try:
      step = int(value.strip())
    except ValueError:
      return None
  else:
    return None
  return step if step > 0 else None


@dataclasses.dataclass(frozen=True)
class PublicationPlan:
  """What a run should upload, on top of the root checkpoint.

  Attributes:
    default_name: Which checkpoint sits at the repository root, ``best`` or
      ``last``.
    default_step: Global step of the root checkpoint, if known.
    companion_name: Subfolder to publish the other checkpoint under, or None
      when there is nothing to publish (it is missing, or it *is* the root
      checkpoint).
    companion_dir: Local directory holding the companion checkpoint.
    companion_step: Global step of the companion checkpoint, if known.
    duplicate: True when the best checkpoint is also the final one, so the
      root already holds both roles.
  """

  default_name: str
  default_step: Optional[int]
  companion_name: Optional[str]
  companion_dir: Optional[str]
  companion_step: Optional[int]
  duplicate: bool


def plan_publication(
    root_is_best: bool,
    best_dir: Optional[str],
    best_step: Optional[int],
    last_dir: Optional[str],
    last_step: Optional[int],
) -> PublicationPlan:
  """Works out which checkpoint still needs to be uploaded, and where.

  Args:
    root_is_best: True when the training run left the best checkpoint at the
      repository root, i.e. ``--load_best_model_at_end True``.
    best_dir: Local directory of the best checkpoint, if it survived.
    best_step: Global step of the best checkpoint.
    last_dir: Local directory of the final-step checkpoint.
    last_step: Global step of the final-step checkpoint.

  Returns:
    The publication plan.
  """
  duplicate = (
      best_step is not None and last_step is not None and best_step == last_step
  )

  if root_is_best:
    default_name = BEST_CHECKPOINT_SUBFOLDER
    default_step = best_step
    companion_name = LAST_CHECKPOINT_SUBFOLDER
    companion_dir, companion_step = last_dir, last_step
  else:
    default_name = LAST_CHECKPOINT_SUBFOLDER
    default_step = last_step
    companion_name = BEST_CHECKPOINT_SUBFOLDER
    companion_dir, companion_step = best_dir, best_step

  if duplicate or not companion_dir:
    return PublicationPlan(
        default_name=default_name,
        default_step=default_step,
        companion_name=None,
        companion_dir=None,
        companion_step=None,
        duplicate=duplicate,
    )

  return PublicationPlan(
      default_name=default_name,
      default_step=default_step,
      companion_name=companion_name,
      companion_dir=companion_dir,
      companion_step=companion_step,
      duplicate=False,
  )


def build_manifest(
    plan: PublicationPlan,
    best_step: Optional[int],
    last_step: Optional[int],
    companion_published: bool,
    metric_for_best_model: Optional[str] = None,
    greater_is_better: Optional[bool] = None,
    best_metric: Optional[float] = None,
) -> Dict[str, Any]:
  """Describes the published repository.

  Args:
    plan: The plan returned by :func:`plan_publication`.
    best_step: Global step of the best checkpoint.
    last_step: Global step of the final-step checkpoint.
    companion_published: Whether the companion upload actually succeeded. A
      failed upload must not be advertised in the manifest.
    metric_for_best_model: Metric the best checkpoint was chosen on.
    greater_is_better: Direction of that metric.
    best_metric: Value the best checkpoint reached.

  Returns:
    A JSON-serialisable manifest. ``subfolder: None`` means "at the root".
  """

  def entry(name: str, step: Optional[int]) -> Dict[str, Any]:
    at_root = plan.default_name == name or plan.duplicate
    if at_root:
      return {"step": step, "subfolder": None, "available": True}
    published = companion_published and plan.companion_name == name
    return {
        "step": step if published else None,
        "subfolder": name if published else None,
        "available": published,
    }

  return {
      "default": plan.default_name,
      "default_step": plan.default_step,
      "metric_for_best_model": metric_for_best_model,
      "greater_is_better": greater_is_better,
      "best_metric": best_metric,
      "checkpoints": {
          BEST_CHECKPOINT_SUBFOLDER: entry(
              BEST_CHECKPOINT_SUBFOLDER, best_step
          ),
          LAST_CHECKPOINT_SUBFOLDER: entry(
              LAST_CHECKPOINT_SUBFOLDER, last_step
          ),
      },
  }
