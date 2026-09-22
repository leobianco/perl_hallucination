"""Discovery and parsing of campaign state files into view models.

The orchestrator writes each campaign to
``<root>/checkpoints/<task>/<name>_state.json`` with :func:`os.replace`, and
archives superseded ones under ``<task>/archive/``. This module globs those
files and turns them into flat, JSON-serializable view models that the site
builder can render without ever importing the campaign engine.

Three properties are load-bearing:

* **Nothing here raises on a bad file.** A half-written, hand-edited or
  newer-schema state file yields ``None`` and is skipped, because one bad file
  must never empty a dashboard that summarises fifty good ones.
* **Nothing here touches the network.** Every link is derived (see
  :mod:`src.dashboard.links`), so a build works offline and a published
  snapshot has no runtime dependencies at all.
* **Nothing here trusts ``config_dict``.** It is dumped wholesale by the
  orchestrator and is published verbatim, so it is redacted on the way in.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import datetime
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.orchestrator import eval_metrics

#: Campaign statuses that mean the campaign will not progress further on its
#: own. Used by the watcher to publish immediately (a terminal transition is
#: the most valuable snapshot of a campaign's life) and by the UI's default
#: "completed only" filter.
TERMINAL_STATUSES = frozenset(
    {"COMPLETED", "FAILED", "STOPPED", "ABORTED"}
)

#: Statuses that mean a human is expected to come back to this campaign.
ATTENDED_STATUSES = frozenset({"PAUSED"})

#: Keys whose values are replaced by a placeholder before anything is written
#: into the published site. ``config_dict`` is a verbatim dump of the campaign
#: configuration; today it holds no credentials, but the dashboard publishes
#: to a remote host and "today" is not a security model.
_SECRET_KEY_RE = re.compile(
    r"(key|token|secret|password|credential|api[-_]?key)", re.IGNORECASE
)
_REDACTED = "\u2022\u2022\u2022 redacted"

#: Human titles for the stage kinds the orchestrator can run.
_STAGE_TITLES = {
    "autorater": "Autorater calibration",
    "sft": "SFT",
    "rm": "Reward model",
    "perl": "PE-RL",
    "eval": "Evaluation",
}

#: Metric keys inside the eval stage that are not per-target measurements.
_EVAL_META_METRICS = frozenset({"num_samples"})

#: The metrics worth a row in the comparison table, in display order.
#:
#: The eval stage records upwards of forty keys per target: every generation
#: statistic in mean/std/median form, every ROUGE variant in F1/precision/
#: recall form, plus a `provenance_*` audit trail whose *values* are adapter
#: repo ids. Tabulating all of them produced a table wide enough that the two
#: numbers the campaign is about scrolled off the screen. The full set is
#: still one click away under the table, and the report below has all of it.
HEADLINE_EVAL_METRICS: Tuple[str, ...] = (
    "decoding_temperature",
    "hallucination_rate",
    "faithfulness_rate",
    "reward_hacking_quality",
    "reward_hacking_rate",
    "reward_hacking_fluency",
    "reward_hacking_non_repetition",
    "reward_hacking_non_extractiveness",
    "repetition_rate_mean",
    "rouge1_f1_mean",
    "rouge2_f1_mean",
    "rougeL_f1_mean",
    "bertscore_f1_mean",
    "bertscore_f1_std",
    "perplexity_mean",
)

#: Older single-temperature runs wrote these bare, without the `_mean`
#: suffix that the generation-metrics evaluator adds today. Both spellings
#: are headline metrics; only one of them will ever be present.
_HEADLINE_ALIASES: Dict[str, str] = {
    "bertscore_f1": "bertscore_f1_mean",
    "perplexity": "perplexity_mean",
    "repetition_rate": "repetition_rate_mean",
    "rouge1_f1": "rouge1_f1_mean",
    "rouge2_f1": "rouge2_f1_mean",
    "rougeL_f1": "rougeL_f1_mean",
}


def is_headline_metric(name: str) -> bool:
  """Whether a metric earns a row in the main comparison table.

  Args:
    name: The bare metric name, without its target prefix.

  Returns:
    True for the curated set, including the pre-`_mean` spellings.
  """
  return name in HEADLINE_EVAL_METRICS or name in _HEADLINE_ALIASES


def is_tabulated_metric(name: str) -> bool:
  """Whether a metric belongs in a table at all.

  ``provenance_*`` keys hold adapter repo ids and checkpoint directories, and
  most ``autorater_*`` keys describe how the judge behaved rather than how
  the model did. The orchestrator's own report excludes them from its
  comparison table; this keeps the dashboard's definition identical instead
  of letting the two drift.

  Args:
    name: The bare metric name.

  Returns:
    Whether the metric is a measurement rather than an audit trail.
  """
  if name in _EVAL_META_METRICS:
    return False
  if name in eval_metrics.TABULATED_AUDIT_KEYS:
    return True
  return not name.startswith(eval_metrics.AUDIT_PREFIXES)


def _metric_sort_key(name: str) -> Tuple[int, str]:
  """Orders metrics by the curated sequence, then alphabetically.

  Args:
    name: The bare metric name.

  Returns:
    A sort key.
  """
  canonical = _HEADLINE_ALIASES.get(name, name)
  if canonical in HEADLINE_EVAL_METRICS:
    return (HEADLINE_EVAL_METRICS.index(canonical), name)
  return (len(HEADLINE_EVAL_METRICS), name)

# The orchestrator's dry-run mode fabricates sweep ids of the form
# "mock_sweep_<epoch>". Campaigns predating the persisted ``dry_run`` config
# key can only be recognised by this marker.
_MOCK_SWEEP_MARKER = "mock_sweep"


def _redact(value: Any) -> Any:
  """Recursively replaces secret-looking values in a parsed JSON structure.

  Args:
    value: Any JSON-decoded value.

  Returns:
    A copy with values under secret-looking keys replaced by a placeholder.
    Non-container values are returned unchanged.
  """
  if isinstance(value, dict):
    out = {}
    for key, item in value.items():
      if isinstance(key, str) and _SECRET_KEY_RE.search(key):
        # Only mask things that could carry a secret. Masking `max_tokens`
        # because it contains "token" would make the config page useless,
        # so numbers and booleans are left alone.
        out[key] = _REDACTED if isinstance(item, str) and item else item
      else:
        out[key] = _redact(item)
    return out
  if isinstance(value, list):
    return [_redact(item) for item in value]
  return value


def _parse_iso(value: Optional[str]) -> Optional[datetime.datetime]:
  """Parses an ISO timestamp written by the orchestrator, tolerantly.

  Args:
    value: An ISO-8601 string, or None.

  Returns:
    The parsed datetime, or None if absent or unparseable.
  """
  if not value or not isinstance(value, str):
    return None
  try:
    return datetime.datetime.fromisoformat(value)
  except ValueError:
    return None


def _elapsed_seconds(
    start: Optional[str], end: Optional[str]
) -> Optional[float]:
  """Returns the wall-clock duration between two ISO timestamps.

  Args:
    start: ISO start timestamp.
    end: ISO end timestamp.

  Returns:
    Seconds elapsed, or None when either endpoint is missing or invalid.
    Negative results (clock changes, hand-edited files) are discarded rather
    than displayed as a negative duration.
  """
  first, last = _parse_iso(start), _parse_iso(end)
  if first is None or last is None:
    return None
  delta = (last - first).total_seconds()
  return delta if delta >= 0 else None


def split_stage_id(stage_id: str) -> Tuple[str, Optional[str]]:
  """Splits a stage id into its kind and optional dataset flavor.

  Branched campaigns name their stages ``rm:organic`` /
  ``perl:synthetic_struct``; unbranched ones use a bare ``rm`` / ``perl``.

  Args:
    stage_id: Stage identifier as recorded in the state file.

  Returns:
    A ``(kind, flavor)`` tuple, with ``flavor`` None when unbranched.
  """
  if ":" in stage_id:
    kind, _, flavor = stage_id.partition(":")
    return kind, flavor or None
  return stage_id, None


def stage_title(stage_id: str) -> str:
  """Returns a human-readable title for a stage id.

  Args:
    stage_id: Stage identifier, possibly flavor-branched.

  Returns:
    A display title such as ``"Reward model (organic)"``.
  """
  kind, flavor = split_stage_id(stage_id)
  title = _STAGE_TITLES.get(kind, kind.upper())
  if flavor:
    title = f"{title} ({flavor.replace('_', ' ')})"
  return title


@dataclass
class StageView:
  """A single stage of a campaign, flattened for rendering."""

  stage_id: str
  kind: str
  flavor: Optional[str]
  title: str
  status: str = "PENDING"
  sweep_id: Optional[str] = None
  sweep_name: Optional[str] = None
  best_run_id: Optional[str] = None
  best_metric_val: Optional[float] = None
  final_metric_val: Optional[float] = None
  metric_name: Optional[str] = None
  goal: Optional[str] = None
  model_repo_id: Optional[str] = None
  trials_done: int = 0
  trials_total: int = 0
  selection_strategy: Optional[str] = None
  selection_step: Optional[int] = None
  selection_window: Optional[int] = None
  sweep_outcome: Optional[str] = None
  warnings: List[str] = field(default_factory=list)
  error_message: Optional[str] = None
  start_time: Optional[str] = None
  end_time: Optional[str] = None
  elapsed_seconds: Optional[float] = None
  metrics: Dict[str, Any] = field(default_factory=dict)

  @property
  def is_sweep(self) -> bool:
    """Whether this stage ran a W&B sweep (and so has trials and a winner)."""
    return self.kind in ("sft", "rm", "perl")

  @property
  def progress_label(self) -> str:
    """Returns the ``NN/NN`` trial counter, or an em dash for non-sweeps."""
    if not self.is_sweep or not self.trials_total:
      return "\u2014"
    return f"{self.trials_done}/{self.trials_total}"

  def to_dict(self) -> Dict[str, Any]:
    """Returns a JSON-serializable copy for ``data/index.json``."""
    payload = dataclasses.asdict(self)
    payload["progress_label"] = self.progress_label
    payload["is_sweep"] = self.is_sweep
    return payload


@dataclass
class EvalTarget:
  """One evaluated policy in the eval stage's metric table.

  The eval stage keys its metrics ``"<label>/<metric>"``, where the label is
  the target (``sft``, ``perl``, ``perl:organic``) optionally suffixed with
  the decoding temperature it was sampled at (``sft@t0.7``). Deltas use the
  reserved ``delta`` label.
  """

  label: str
  base_label: str
  temperature: Optional[float]
  is_delta: bool
  metrics: Dict[str, float] = field(default_factory=dict)
  model_repo_id: Optional[str] = None
  #: Each delta re-expressed as a fraction of the baseline it improved on,
  #: keyed by the same bare metric name. Populated for delta targets only,
  #: and only for metrics whose baseline could be recovered and was
  #: positive, so a missing key means "not expressible" rather than "zero".
  metric_ratios: Dict[str, float] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    """Returns a JSON-serializable copy."""
    return dataclasses.asdict(self)


@dataclass
class CampaignSummary:
  """Everything the dashboard knows about one campaign."""

  slug: str
  campaign_id: str
  task: str
  status: str
  current_stage: Optional[str]
  created_at: Optional[str]
  updated_at: Optional[str]
  state_path: str
  is_archived: bool = False
  report_path: Optional[str] = None
  user: str = ""
  project: str = ""
  wandb_entity: Optional[str] = None
  eval_wandb_project: Optional[str] = None
  base_model: Optional[str] = None
  reward_base_model: Optional[str] = None
  seed: Optional[int] = None
  rm_dataset_flavors: List[str] = field(default_factory=list)
  requested_stages: List[str] = field(default_factory=list)
  stages: List[StageView] = field(default_factory=list)
  eval_targets: List[EvalTarget] = field(default_factory=list)
  eval_metric_names: List[str] = field(default_factory=list)
  config: Dict[str, Any] = field(default_factory=dict)

  @property
  def entity(self) -> str:
    """Resolved W&B entity, mirroring the reporter's fallback."""
    return self.wandb_entity or self.user

  @property
  def is_terminal(self) -> bool:
    """Whether the campaign has finished, one way or another."""
    return self.status in TERMINAL_STATUSES

  @property
  def is_completed(self) -> bool:
    """Whether the campaign ran to a clean completion."""
    return self.status == "COMPLETED"

  @property
  def is_dry_run(self) -> bool:
    """Whether the campaign was a rehearsal rather than real training.

    Dry runs never touch W&B or Hugging Face: the orchestrator fabricates
    sweep and run ids locally, so every outbound link the dashboard builds
    for them points at something that does not exist. That is worth saying
    out loud rather than letting the reader discover it by clicking.

    The ``dry_run`` config key is authoritative, but campaigns recorded
    before it was persisted only betray themselves through their mock sweep
    ids, so both are checked.

    Returns:
      True when the campaign's recorded ids are fabricated.
    """
    if self.config.get("dry_run"):
      return True
    return any(
        _MOCK_SWEEP_MARKER in (stage.sweep_id or "") for stage in self.stages
    )

  @property
  def elapsed_seconds(self) -> Optional[float]:
    """Wall-clock time from campaign creation to its last update."""
    return _elapsed_seconds(self.created_at, self.updated_at)

  @property
  def warnings(self) -> List[str]:
    """All stage warnings, prefixed with the stage that raised them."""
    out = []
    for stage in self.stages:
      for warning in stage.warnings:
        out.append(f"{stage.title}: {warning}")
    return out

  @property
  def errors(self) -> List[str]:
    """All stage error messages, prefixed with the stage that failed."""
    return [
        f"{stage.title}: {stage.error_message}"
        for stage in self.stages
        if stage.error_message
    ]

  def stage(self, stage_id: str) -> Optional[StageView]:
    """Returns a stage by id, or None.

    Args:
      stage_id: Stage identifier as recorded in the state file.

    Returns:
      The matching :class:`StageView`, or None.
    """
    for candidate in self.stages:
      if candidate.stage_id == stage_id:
        return candidate
    return None

  def stages_of_kind(self, kind: str) -> List[StageView]:
    """Returns every stage of a given kind, across dataset-flavor branches.

    Args:
      kind: Stage kind, e.g. ``"perl"``.

    Returns:
      The matching stages, in campaign order.
    """
    return [stage for stage in self.stages if stage.kind == kind]

  @property
  def headline_metric_names(self) -> List[str]:
    """The curated metrics, in display order.

    A state written across a version boundary can hold both spellings of the
    same quantity - `bertscore_f1` and `bertscore_f1_mean` - which rendered
    as two identically labelled rows with different numbers. The current
    spelling wins.
    """
    names = [n for n in self.eval_metric_names if is_headline_metric(n)]
    canonical = {_HEADLINE_ALIASES.get(n, n) for n in names}
    return [
        n
        for n in names
        if n not in _HEADLINE_ALIASES or _HEADLINE_ALIASES[n] not in canonical
    ]

  @property
  def secondary_metric_names(self) -> List[str]:
    """Everything else that is still a measurement, in display order.

    Includes any superseded spelling dropped from the headline set, so that
    no recorded measurement disappears from the page entirely.
    """
    headline = set(self.headline_metric_names)
    return [n for n in self.eval_metric_names if n not in headline]

  @property
  def headline(self) -> Dict[str, Optional[float]]:
    """The two numbers worth putting in a list row.

    Returns:
      A mapping with the PE-RL hallucination rate and its delta against the
      matched SFT baseline, or Nones when the campaign never got that far.
    """
    rate: Optional[float] = None
    delta: Optional[float] = None
    for target in self.eval_targets:
      if target.is_delta and delta is None:
        delta = target.metrics.get("hallucination_rate")
      elif target.base_label.startswith("perl") and rate is None:
        rate = target.metrics.get("hallucination_rate")
    return {"hallucination_rate": rate, "hallucination_delta": delta}

  def to_dict(self) -> Dict[str, Any]:
    """Returns a JSON-serializable copy for ``data/index.json``."""
    return {
        "slug": self.slug,
        "campaign_id": self.campaign_id,
        "task": self.task,
        "status": self.status,
        "current_stage": self.current_stage,
        "created_at": self.created_at,
        "updated_at": self.updated_at,
        "state_path": self.state_path,
        "is_archived": self.is_archived,
        "report_path": self.report_path,
        "has_report": bool(self.report_path),
        "user": self.user,
        "project": self.project,
        "entity": self.entity,
        "eval_wandb_project": self.eval_wandb_project,
        "base_model": self.base_model,
        "reward_base_model": self.reward_base_model,
        "seed": self.seed,
        "rm_dataset_flavors": list(self.rm_dataset_flavors),
        "requested_stages": list(self.requested_stages),
        "stages": [stage.to_dict() for stage in self.stages],
        "eval_targets": [target.to_dict() for target in self.eval_targets],
        "eval_metric_names": list(self.eval_metric_names),
        "is_terminal": self.is_terminal,
        "is_completed": self.is_completed,
        "is_dry_run": self.is_dry_run,
        "elapsed_seconds": self.elapsed_seconds,
        "headline": self.headline,
        "warnings": self.warnings,
        "errors": self.errors,
    }


def _parse_eval_label(label: str) -> Tuple[str, Optional[float], bool]:
  """Splits an eval metric label into base target, temperature and delta flag.

  Args:
    label: Label such as ``"sft"``, ``"sft@t0.7"``, ``"perl:organic"`` or
      ``"delta"``.

  Returns:
    A ``(base_label, temperature, is_delta)`` tuple. The temperature is None
    when the label carries no ``@t`` suffix, which means the target was
    scored at ``eval.temperature``.
  """
  base, _, temp_part = label.partition("@t")
  temperature: Optional[float] = None
  if temp_part:
    try:
      temperature = float(temp_part)
    except ValueError:
      temperature = None
  return base, temperature, base.split(":")[0] == "delta"


def _eval_target_kind(target: "EvalTarget") -> str:
  """Returns the target's stage kind, e.g. ``perl`` for ``perl:organic``."""
  return target.base_label.partition(":")[0]


def _eval_target_flavor(target: "EvalTarget") -> str:
  """Returns the target's branch flavor, empty for unbranched campaigns."""
  return target.base_label.partition(":")[2]


def _order_eval_targets(targets: Iterable["EvalTarget"]) -> List["EvalTarget"]:
  """Orders eval columns so every policy sits next to its own delta.

  Baselines come first (``sft``, ``sft@t0.6``, ``sft@t1.0``), then one block
  per trained policy: the policy column immediately followed by the delta
  measured against it. Sorting deltas to the end instead pushed the two
  numbers a reader compares to opposite ends of the table, and a fan-out
  campaign made that worse by interleaving the flavors.

  Args:
    targets: The eval targets, in any order.

  Returns:
    A new list in display order.
  """
  targets = list(targets)
  baselines = sorted(
      (
          t
          for t in targets
          if not t.is_delta and _eval_target_kind(t) != "perl"
      ),
      key=lambda t: t.label,
  )
  policies = sorted(
      (t for t in targets if not t.is_delta and _eval_target_kind(t) == "perl"),
      key=lambda t: t.label,
  )
  deltas_by_flavor: Dict[str, List["EvalTarget"]] = {}
  for target in sorted(
      (t for t in targets if t.is_delta), key=lambda t: t.label
  ):
    deltas_by_flavor.setdefault(_eval_target_flavor(target), []).append(target)

  ordered: List["EvalTarget"] = list(baselines)
  paired: set = set()
  for policy in policies:
    ordered.append(policy)
    for delta in deltas_by_flavor.get(_eval_target_flavor(policy), []):
      if id(delta) not in paired:
        ordered.append(delta)
        paired.add(id(delta))
  # An orphan delta - one whose policy never made it into the metrics -
  # still belongs in the table; it just has nothing to sit beside.
  ordered.extend(
      t
      for deltas in deltas_by_flavor.values()
      for t in deltas
      if id(t) not in paired
  )
  return ordered


def _policy_for_delta(
    delta: "EvalTarget", targets: Iterable["EvalTarget"]
) -> Optional["EvalTarget"]:
  """Finds the policy column a delta column was derived from.

  Args:
    delta: A delta target.
    targets: All eval targets.

  Returns:
    The PE-RL target sharing the delta's flavor, or None when the metrics
    hold a delta whose policy never made it into the table.
  """
  flavor = _eval_target_flavor(delta)
  for target in targets:
    if target.is_delta or _eval_target_kind(target) != "perl":
      continue
    if _eval_target_flavor(target) == flavor:
      return target
  return None


def _attach_metric_ratios(targets: List["EvalTarget"]) -> None:
  """Fills in each delta's relative change, in place.

  An absolute delta is hard to read on its own: whether ``+0.043`` is a rout
  or a rounding error depends entirely on what it started from. The baseline
  is not stored - a policy is compared against the SFT row sampled at *its
  own* decoding temperature, which differs per branch - but it is recoverable
  by inverting the delta against the policy column beside it. Doing it that
  way means the ratio automatically uses whichever baseline the eval stage
  chose, including the per-flavor temperature matching, without this module
  having to know the pairing rules.

  Args:
    targets: The eval targets; delta entries are mutated in place.
  """
  for delta in targets:
    if not delta.is_delta:
      continue
    policy = _policy_for_delta(delta, targets)
    if policy is None:
      continue
    for metric, improvement in delta.metrics.items():
      ratio = eval_metrics.relative_change(
          metric, improvement, policy.metrics.get(metric)
      )
      if ratio is not None:
        delta.metric_ratios[metric] = ratio


def _extract_eval_targets(
    stage: Optional[StageView], stages: Iterable[StageView]
) -> Tuple[List[EvalTarget], List[str]]:
  """Reshapes flat ``label/metric`` eval metrics into per-target records.

  Args:
    stage: The eval stage, if the campaign has one.
    stages: All campaign stages, used to attribute a model repo to a target.

  Returns:
    A ``(targets, metric_names)`` tuple. ``metric_names`` is the ordered
    union of metric suffixes seen across targets, so the renderer can lay out
    a table without re-scanning.
  """
  if stage is None or not stage.metrics:
    return [], []

  by_label: Dict[str, EvalTarget] = {}
  metric_names: List[str] = []
  for key, value in stage.metrics.items():
    label, _, metric = key.rpartition("/")
    if not label:
      # A bare metric with no target prefix; keep it out of the table rather
      # than inventing a column for it.
      continue
    if label not in by_label:
      base, temperature, is_delta = _parse_eval_label(label)
      by_label[label] = EvalTarget(
          label=label,
          base_label=base,
          temperature=temperature,
          is_delta=is_delta,
      )
    by_label[label].metrics[metric] = value
    if metric not in metric_names and is_tabulated_metric(metric):
      metric_names.append(metric)

  # The decoding temperature is recorded as a metric rather than in the
  # label for targets scored at the campaign default, so prefer it when the
  # label carried no `@t` suffix.
  for target in by_label.values():
    if target.temperature is None:
      recorded = target.metrics.get("decoding_temperature")
      if isinstance(recorded, (int, float)):
        target.temperature = float(recorded)

  stage_by_kind: Dict[str, StageView] = {}
  for candidate in stages:
    stage_by_kind.setdefault(candidate.stage_id, candidate)
    stage_by_kind.setdefault(candidate.kind, candidate)
  for target in by_label.values():
    if target.is_delta:
      continue
    owner = stage_by_kind.get(target.base_label)
    if owner is not None:
      target.model_repo_id = owner.model_repo_id

  ordered = _order_eval_targets(by_label.values())
  _attach_metric_ratios(ordered)
  metric_names.sort(key=_metric_sort_key)
  return ordered, metric_names


def _stage_config(config: Dict[str, Any], kind: str) -> Dict[str, Any]:
  """Returns the per-stage config block for a stage kind.

  Args:
    config: The campaign's ``config_dict``.
    kind: Stage kind (``sft``, ``rm``, ``perl``, ``eval``).

  Returns:
    The stage's config mapping, or an empty dict.
  """
  block = config.get(kind)
  return block if isinstance(block, dict) else {}


def _resolve_report_path(
    root: str, config: Dict[str, Any], campaign_id: str
) -> Optional[str]:
  """Finds the markdown report a campaign produced, if it produced one.

  The orchestrator does not record the path, so it is reconstructed from the
  same two pieces the reporter used: ``reporting.reports_dir`` and the
  campaign name.

  Args:
    root: Repository root the campaign ran from.
    config: The campaign's ``config_dict``.
    campaign_id: Campaign name, which is also its report's filename stem.

  Returns:
    A path relative to ``root``, or None when no report exists on disk.
  """
  reporting = config.get("reporting")
  reports_dir = "reports"
  if isinstance(reporting, dict):
    reports_dir = reporting.get("reports_dir") or "reports"
  name = config.get("name") or campaign_id
  relative = os.path.join(reports_dir, f"{name}_summary.md")
  if os.path.isfile(os.path.join(root, relative)):
    return relative
  return None


def _slug_for(root: str, state_path: str) -> str:
  """Builds a stable, readable, filesystem-safe page id from a state file.

  The campaign id alone is not unique: ``--fresh`` archives a state file
  under the same name, so an archived and a live campaign can collide. The
  archive's timestamp suffix disambiguates them, and the task is prepended
  only when the campaign name does not already carry it (auto-generated
  names do; names pinned in a YAML may not).

  Args:
    root: Repository root.
    state_path: Absolute path of the state file.

  Returns:
    A slug safe to use as an HTML filename and a URL fragment, such as
    ``npov-campaign-2609211206`` or ``npov-campaign-2609211206-20260921120606``
    for its archived namesake.
  """
  directory, filename = os.path.split(state_path)
  stem = re.sub(r"_state(_\d+)?\.json$", r"\1", filename)
  task = os.path.basename(directory)
  if task == "archive":
    task = os.path.basename(os.path.dirname(directory))
  if task and not stem.lower().startswith(task.lower()):
    stem = f"{task}_{stem}"
  slug = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-")
  del root  # The slug is path-independent; only the file's name matters.
  return slug or "campaign"


def load_campaign(
    state_path: str, root: str = "."
) -> Optional[CampaignSummary]:
  """Parses one state file into a :class:`CampaignSummary`.

  Args:
    state_path: Path to a ``*_state.json`` file.
    root: Repository root, used to resolve the report path and the slug.

  Returns:
    The parsed summary, or None when the file is missing, unreadable, mid
    write, or not a campaign state file at all. Callers skip Nones; one bad
    file must never take down the whole index.
  """
  try:
    with open(state_path, "r", encoding="utf-8") as handle:
      data = json.load(handle)
  except (OSError, ValueError):
    return None
  if not isinstance(data, dict) or "campaign_id" not in data:
    return None

  config = data.get("config_dict")
  config = _redact(config) if isinstance(config, dict) else {}
  raw_stages = data.get("stages")
  raw_stages = raw_stages if isinstance(raw_stages, dict) else {}

  # Campaign order first, then any stage present on disk but absent from the
  # order (a state file written by a newer build, or a hand edit).
  order = [s for s in data.get("stages_order") or [] if s in raw_stages]
  order += [s for s in raw_stages if s not in order]

  stages: List[StageView] = []
  for stage_id in order:
    payload = raw_stages.get(stage_id)
    if not isinstance(payload, dict):
      continue
    kind, flavor = split_stage_id(stage_id)
    stage_cfg = _stage_config(config, kind)
    metrics = payload.get("metrics")
    stages.append(
        StageView(
            stage_id=stage_id,
            kind=kind,
            flavor=flavor,
            title=stage_title(stage_id),
            status=str(payload.get("status") or "PENDING"),
            sweep_id=payload.get("sweep_id"),
            sweep_name=payload.get("sweep_name"),
            best_run_id=payload.get("best_run_id"),
            best_metric_val=payload.get("best_metric_val"),
            final_metric_val=payload.get("final_metric_val"),
            metric_name=stage_cfg.get("metric"),
            goal=stage_cfg.get("goal"),
            model_repo_id=payload.get("model_repo_id"),
            trials_done=int(payload.get("trials_done") or 0),
            trials_total=int(payload.get("trials_total") or 0),
            selection_strategy=payload.get("selection_strategy"),
            selection_step=payload.get("selection_step"),
            selection_window=payload.get("selection_window"),
            sweep_outcome=payload.get("sweep_outcome"),
            warnings=list(payload.get("warnings") or []),
            error_message=payload.get("error_message"),
            start_time=payload.get("start_time"),
            end_time=payload.get("end_time"),
            elapsed_seconds=_elapsed_seconds(
                payload.get("start_time"), payload.get("end_time")
            ),
            metrics=metrics if isinstance(metrics, dict) else {},
        )
    )

  campaign_id = str(data["campaign_id"])
  eval_stage = next((s for s in stages if s.kind == "eval"), None)
  eval_targets, eval_metric_names = _extract_eval_targets(eval_stage, stages)
  eval_cfg = _stage_config(config, "eval")

  return CampaignSummary(
      slug=_slug_for(root, state_path),
      campaign_id=campaign_id,
      task=str(data.get("task_name") or config.get("task_name") or "unknown"),
      status=str(data.get("status") or "IN_PROGRESS"),
      current_stage=data.get("current_stage"),
      created_at=data.get("created_at"),
      updated_at=data.get("updated_at"),
      state_path=os.path.relpath(state_path, root),
      is_archived=f"{os.sep}archive{os.sep}" in state_path,
      report_path=_resolve_report_path(root, config, campaign_id),
      user=str(config.get("user") or ""),
      project=str(config.get("project") or ""),
      wandb_entity=config.get("wandb_entity"),
      eval_wandb_project=eval_cfg.get("wandb_project"),
      base_model=config.get("base_model"),
      reward_base_model=config.get("reward_base_model"),
      seed=config.get("seed"),
      rm_dataset_flavors=list(config.get("rm_dataset_flavors") or []),
      requested_stages=list(config.get("stages") or []),
      stages=stages,
      eval_targets=eval_targets,
      eval_metric_names=eval_metric_names,
      config=config,
  )


def discover_state_files(
    root: str = ".", include_archived: bool = True
) -> List[str]:
  """Finds every campaign state file under a repository root.

  Mirrors the orchestrator's own one-level glob
  (``checkpoints/<task>/*_state.json``) and optionally the archive directory
  that ``run --fresh`` moves superseded campaigns into.

  Args:
    root: Repository root to search.
    include_archived: Whether to include ``checkpoints/<task>/archive/``.

  Returns:
    Absolute paths, sorted for determinism.
  """
  checkpoints = os.path.join(root, "checkpoints")
  found: List[str] = []
  if not os.path.isdir(checkpoints):
    return found
  for task in sorted(os.listdir(checkpoints)):
    task_dir = os.path.join(checkpoints, task)
    if not os.path.isdir(task_dir):
      continue
    directories = [task_dir]
    if include_archived:
      archive = os.path.join(task_dir, "archive")
      if os.path.isdir(archive):
        directories.append(archive)
    for directory in directories:
      try:
        entries = sorted(os.listdir(directory))
      except OSError:
        continue
      for entry in entries:
        if entry.endswith(".json") and "_state" in entry:
          found.append(os.path.abspath(os.path.join(directory, entry)))
  return found


def build_index(
    root: str = ".", include_archived: bool = True
) -> List[CampaignSummary]:
  """Loads every discoverable campaign, newest first.

  Args:
    root: Repository root to search.
    include_archived: Whether to include archived campaigns.

  Returns:
    Parsed campaigns sorted by ``updated_at`` descending. Unparseable files
    are silently skipped.
  """
  campaigns = []
  for path in discover_state_files(root, include_archived=include_archived):
    summary = load_campaign(path, root=root)
    if summary is not None:
      campaigns.append(summary)
  campaigns.sort(
      key=lambda c: (c.updated_at or "", c.campaign_id), reverse=True
  )
  return campaigns
