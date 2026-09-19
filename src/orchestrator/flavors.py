"""Reward-model dataset flavors and the campaign branches they create.

A campaign trains its reward model on one of several datasets, and that
choice propagates: the RM sweep trains on it, PE-RL is then optimised
against the resulting reward model, and the evaluation scores the policy
that came out. Selecting two flavors therefore does not mean "two RM
sweeps", it means two independent *branches* that rejoin only at the
evaluation.

This module owns the vocabulary for that fan-out:

* which flavors exist and which dataset each names,
* how a branch is identified (``rm:synthetic_struct``),
* how the branches of a campaign are laid out in execution order.

The single-flavor case is deliberately indistinguishable from the campaigns
that existed before branches did: the stage ids stay ``rm`` and ``perl``, so
existing state files resume and existing reports render unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple


ORGANIC = "organic"
SYNTHETIC_STRUCT = "synthetic_struct"

#: Flavors the orchestrator can train a reward model on, in the order their
#: branches execute. These are exactly the dataset suffixes
#: ``scripts/reward_model.sh`` appends to ``{user}/{task}_rm``.
#:
#: ``synthetic_llm`` is intentionally absent. Its dataset exists, but it is
#: the only flavor whose training split is *assembled* at load time from two
#: other datasets via ``--num_organic_hallus_to_keep`` /
#: ``--num_struct_hallus_to_keep`` (see
#: ``BaseTaskProcessor.augment_training_split``). Exposing it as a plain
#: flavor would hide those two knobs and silently train on a mixture nobody
#: asked for.
RM_DATASET_FLAVORS: Tuple[str, ...] = (ORGANIC, SYNTHETIC_STRUCT)

#: Human-readable names for sweep names, dashboard rows and the report.
FLAVOR_TITLES: Dict[str, str] = {
    ORGANIC: "Organic",
    SYNTHETIC_STRUCT: "Synthetic Struct",
}

#: Short tags embedded in Hugging Face repo ids. Not the flavor string
#: itself: a Hub repo id may hold 96 characters and the RM and PE-RL names
#: already spend ~75 of them on model, seed, hyperparameters and timestamp.
FLAVOR_SLUGS: Dict[str, str] = {
    ORGANIC: "organic",
    SYNTHETIC_STRUCT: "synstruct",
}

#: Stage kinds that get one execution per flavor. SFT is upstream of the
#: reward model and is shared by every branch - training it twice would burn
#: GPU hours to produce the same checkpoint. Evaluation is downstream and
#: scores every branch in a single pass, which is the point of running them
#: in one campaign.
BRANCHED_KINDS: Tuple[str, ...] = ("rm", "perl")

#: Separates the kind from the flavor in a branched stage id.
SEPARATOR = ":"


def normalize_flavors(raw: Any) -> List[str]:
  """Cleans a user-supplied flavor list into canonical execution order.

  Args:
    raw: Whatever the YAML or the wizard produced - possibly a bare string,
      possibly padded, duplicated, or in click order.

  Returns:
    Lowercase, de-duplicated, known-flavors-first list. Unknown entries are
    preserved at the end so that :meth:`CampaignConfig.validate` can report
    them by name instead of silently dropping them.
  """
  if isinstance(raw, str):
    raw = [raw]
  seen: List[str] = []
  for item in raw or []:
    if not isinstance(item, str):
      continue
    key = item.strip().lower()
    if key and key not in seen:
      seen.append(key)
  known = [f for f in RM_DATASET_FLAVORS if f in seen]
  unknown = [f for f in seen if f not in RM_DATASET_FLAVORS]
  return known + unknown


def campaign_flavors(config: Any) -> List[str]:
  """Returns the flavors ``config`` branches over, never empty."""
  flavors = normalize_flavors(getattr(config, "rm_dataset_flavors", None))
  return flavors or [ORGANIC]


def is_branched(config: Any) -> bool:
  """True when the campaign runs more than one flavor and so needs branch ids."""
  return len(campaign_flavors(config)) > 1


def dataset_repo_id(user: str, task_name: str, flavor: str) -> str:
  """Returns the Hub dataset a reward-model branch trains on.

  Mirrors ``scripts/reward_model.sh``, which starts from
  ``{user}/{task}_rm`` and appends the flavor.

  Args:
    user: Hub namespace.
    task_name: Campaign task, e.g. ``ragtruth``.
    flavor: One of :data:`RM_DATASET_FLAVORS`.

  Returns:
    The fully qualified dataset repo id.
  """
  return f"{user}/{task_name}_rm_{flavor}"


def flavor_slug(flavor: Optional[str]) -> str:
  """Returns the short tag for ``flavor`` used inside model repo ids."""
  if not flavor:
    return ""
  return FLAVOR_SLUGS.get(flavor, flavor.replace("_", ""))


def flavor_title(flavor: Optional[str]) -> str:
  """Returns the human-readable name for ``flavor``."""
  if not flavor:
    return ""
  return FLAVOR_TITLES.get(flavor, flavor.replace("_", " ").title())


def make_stage_id(kind: str, flavor: Optional[str], branched: bool) -> str:
  """Builds the identifier a branch is known by in state, events and the UI.

  Args:
    kind: ``sft``, ``rm``, ``perl`` or ``eval``.
    flavor: The branch's flavor, or None for unbranched kinds.
    branched: Whether the campaign runs more than one flavor.

  Returns:
    ``rm`` for a single-flavor campaign, ``rm:synthetic_struct`` otherwise.
    Keeping the plain form when there is nothing to disambiguate is what
    lets pre-branch state files resume untouched.
  """
  if not branched or not flavor or kind not in BRANCHED_KINDS:
    return kind
  return f"{kind}{SEPARATOR}{flavor}"


def split_stage_id(stage_id: str) -> Tuple[str, Optional[str]]:
  """Splits a stage id into ``(kind, flavor)``.

  Args:
    stage_id: ``rm`` or ``rm:synthetic_struct``.

  Returns:
    The kind, and the flavor when the id carries one.
  """
  if not isinstance(stage_id, str):
    return "", None
  if SEPARATOR not in stage_id:
    return stage_id, None
  kind, _, flavor = stage_id.partition(SEPARATOR)
  return kind, flavor or None


@dataclasses.dataclass(frozen=True)
class PlannedStage:
  """One executable stage of a campaign, after branch expansion.

  Attributes:
    stage_id: Key used in the state file, the event bus and the dashboard.
    kind: Which stage implementation runs (``sft``/``rm``/``perl``/``eval``).
      Also the attribute name of its configuration on ``CampaignConfig``.
    flavor: The branch's dataset flavor, or None for shared stages.
    title: Display title for the dashboard and the report.
  """

  stage_id: str
  kind: str
  flavor: Optional[str] = None
  title: str = ""


def build_plan(
    config: Any, stage_titles: Optional[Dict[str, str]] = None
) -> List[PlannedStage]:
  """Expands ``config.stages`` into the stages that will actually execute.

  A single-flavor campaign maps one-to-one onto the configured stages. A
  two-flavor campaign expands ``rm`` and ``perl`` into one branch each.

  Expansion is *kind-major*: both RM branches run before either PE-RL
  branch. The alternative, finishing one flavor end to end before starting
  the other, was rejected because a PE-RL sweep costs far more GPU time than
  an RM sweep - running both reward models first means a failure in the
  cheap half is discovered before the expensive half has begun, and it lets
  the user compare the two ROC-AUCs before any policy training starts.

  Args:
    config: The campaign configuration.
    stage_titles: Optional base titles by kind, e.g. ``{"rm": "RM Sweep"}``.

  Returns:
    The executable stages, in execution order.
  """
  titles = stage_titles or {}
  flavors = campaign_flavors(config)
  branched = len(flavors) > 1

  raw_stages = list(getattr(config, "stages", None) or [])
  seen: List[str] = []
  for stage in raw_stages:
    if isinstance(stage, str):
      key = stage.strip().lower()
      if key and key not in seen:
        seen.append(key)

  plan: List[PlannedStage] = []
  for kind in seen:
    base_title = titles.get(kind, kind.upper())
    if kind not in BRANCHED_KINDS:
      plan.append(PlannedStage(stage_id=kind, kind=kind, title=base_title))
      continue
    for flavor in flavors:
      title = base_title
      if branched:
        title = f"{base_title} ({flavor_title(flavor)})"
      plan.append(
          PlannedStage(
              stage_id=make_stage_id(kind, flavor, branched),
              kind=kind,
              flavor=flavor,
              title=title,
          )
      )
  return plan


def branch_stage_ids(config: Any, kind: str) -> List[str]:
  """Returns the stage ids of every branch of ``kind``, in execution order.

  Deliberately independent of ``config.stages``: this answers "how many
  branches does this kind have in this campaign", not "is it scheduled to
  run right now". The report and the resume path both need the former, since
  a `--stages eval` invocation still has to find the reward models and
  policies that earlier invocations produced.

  Args:
    config: The campaign configuration.
    kind: ``sft``, ``rm``, ``perl`` or ``eval``.

  Returns:
    One id per flavor for a branched kind, otherwise the single plain id.
  """
  if kind not in BRANCHED_KINDS:
    return [kind]
  branched = is_branched(config)
  return [
      make_stage_id(kind, flavor, branched)
      for flavor in campaign_flavors(config)
  ]


def stage_id_for_flavor(config: Any, kind: str, flavor: Optional[str]) -> str:
  """Returns the stage id of ``kind``'s branch for ``flavor``."""
  return make_stage_id(kind, flavor, is_branched(config))


def describe_flavors(flavors: Sequence[str]) -> str:
  """Returns a human-readable summary such as 'Organic + Synthetic Struct'."""
  titles = [flavor_title(f) for f in flavors]
  return " + ".join(titles) if titles else ""


def log_tag(stage_id: Optional[str], width: int = 4) -> str:
  """Returns the fixed-width tag prefixing this branch's log lines.

  A plain truncation to ``width`` is not enough once a campaign branches:
  ``perl:organic`` and ``perl:synthetic_struct`` both cut down to ``PERL``,
  which makes every PE-RL line in the transcript ambiguous about which
  branch produced it. Branched ids therefore spend two of their characters
  on ``:`` plus the flavor's initial.

  Args:
    stage_id: ``rm``, ``perl:organic``, or None for non-stage output.
    width: Column width the tag is padded to.

  Returns:
    ``SFT ``, ``RM  ``, ``RM:O``, ``PE:S`` - always exactly ``width`` long.
  """
  kind, flavor = split_stage_id(stage_id or "run")
  if not kind:
    kind = "run"
  if flavor:
    head = kind.upper()[: max(1, width - 2)]
    tag = f"{head}{SEPARATOR}{flavor[0].upper()}"
  else:
    tag = kind.upper()[:width]
  return tag[:width].ljust(width)
