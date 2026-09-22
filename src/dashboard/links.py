"""Construction of every outbound W&B and Hugging Face link.

The dashboard makes **no API calls**: the entire integration with W&B and the
Hub is a set of hyperlinks that your browser follows with credentials the
dashboard never sees. That is why this module exists and why it is the only
place a URL may be assembled — a fallback implemented twice is a 404
implemented twice.

Two conventions are re-derived rather than read from the state file, because
the orchestrator does not record them:

* the **W&B entity**, which falls back from ``wandb_entity`` to ``user``
  exactly as :mod:`src.orchestrator.reporter` does;
* the **evaluation completions dataset**, whose id is recomputed with the very
  function the eval stage used (:func:`src.utils.build_eval_dataset_repo_id`).

The second is a deliberate coupling: it saves persisting the id at campaign
time, at the cost that changing the naming convention retroactively breaks
links for old campaigns. ``tests/test_links.py`` pins the current convention so
that the change is at least noisy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.dashboard import index as index_mod

try:  # pragma: no cover - exercised implicitly by the golden tests.
  from src.utils import build_eval_dataset_repo_id
except Exception:  # pylint: disable=broad-except
  # src.utils guards its own optional imports (numpy, sklearn, matplotlib),
  # so this should not trigger; if it ever does, the dashboard renders
  # without completions links rather than failing to build.
  build_eval_dataset_repo_id = None  # type: ignore[assignment]

_WANDB = "https://wandb.ai"
_HF = "https://huggingface.co"


@dataclass(frozen=True)
class Link:
  """A single outbound link rendered in the campaign page."""

  label: str
  url: str
  #: Extra context shown next to the link, e.g. a temperature or a caveat.
  note: str = ""
  #: True when the target was recomputed from a naming convention rather
  #: than read from the state file, and so may 404 for old campaigns.
  derived: bool = False

  def to_dict(self) -> dict:
    """Returns a JSON-serializable copy."""
    return {
        "label": self.label,
        "url": self.url,
        "note": self.note,
        "derived": self.derived,
    }


@dataclass(frozen=True)
class LinkGroup:
  """A titled set of links, e.g. all W&B sweeps of a campaign."""

  title: str
  links: List[Link]

  def to_dict(self) -> dict:
    """Returns a JSON-serializable copy."""
    return {
        "title": self.title,
        "links": [link.to_dict() for link in self.links],
    }


def _split_sweep_id(sweep_id: str) -> tuple:
  """Splits a recorded sweep id into ``(entity, project, sweep)`` parts.

  W&B's Python API accepts both a bare sweep id and a fully qualified
  ``entity/project/id`` path, and the orchestrator records whichever it was
  handed.

  Args:
    sweep_id: The recorded sweep identifier.

  Returns:
    A ``(entity, project, sweep)`` tuple whose first two elements are None
    when the id was bare.
  """
  parts = [part for part in str(sweep_id).split("/") if part]
  if len(parts) >= 3:
    return parts[-3], parts[-2], parts[-1]
  return None, None, parts[-1] if parts else ""


def project_url(campaign: index_mod.CampaignSummary) -> Optional[str]:
  """Returns the W&B project page for a campaign's training runs.

  Args:
    campaign: The campaign.

  Returns:
    The project URL, or None when the campaign records no project.
  """
  if not campaign.entity or not campaign.project:
    return None
  return f"{_WANDB}/{campaign.entity}/{campaign.project}"


def eval_project_url(campaign: index_mod.CampaignSummary) -> Optional[str]:
  """Returns the W&B project the evaluation stage logs into.

  Args:
    campaign: The campaign.

  Returns:
    The evaluation project URL, or None when unconfigured.
  """
  if not campaign.entity or not campaign.eval_wandb_project:
    return None
  return f"{_WANDB}/{campaign.entity}/{campaign.eval_wandb_project}"


def sweep_url(
    campaign: index_mod.CampaignSummary, stage: index_mod.StageView
) -> Optional[str]:
  """Returns the W&B sweep page for a stage.

  Args:
    campaign: The campaign owning the stage.
    stage: The stage.

  Returns:
    The sweep URL, or None when the stage never registered a sweep.
  """
  if not stage.sweep_id:
    return None
  entity, project, sweep = _split_sweep_id(stage.sweep_id)
  entity = entity or campaign.entity
  project = project or campaign.project
  if not entity or not project or not sweep:
    return None
  return f"{_WANDB}/{entity}/{project}/sweeps/{sweep}"


def run_url(
    campaign: index_mod.CampaignSummary, stage: index_mod.StageView
) -> Optional[str]:
  """Returns the W&B page of the trial that won a stage.

  Args:
    campaign: The campaign owning the stage.
    stage: The stage.

  Returns:
    The run URL, or None when no winner has been picked yet.
  """
  if not stage.best_run_id:
    return None
  entity, project, _ = _split_sweep_id(stage.sweep_id or "")
  entity = entity or campaign.entity
  project = project or campaign.project
  if not entity or not project:
    return None
  return f"{_WANDB}/{entity}/{project}/runs/{stage.best_run_id}"


def model_url(repo_id: Optional[str]) -> Optional[str]:
  """Returns the Hub page of a model repository.

  Args:
    repo_id: A ``user/name`` model repo id.

  Returns:
    The model URL, or None when ``repo_id`` is empty.
  """
  if not repo_id:
    return None
  return f"{_HF}/{repo_id}"


def model_files_url(repo_id: Optional[str]) -> Optional[str]:
  """Returns the file browser of a model repository.

  Useful because every campaign publishes a companion checkpoint in a
  subfolder and a ``checkpoints.json`` manifest describing both.

  Args:
    repo_id: A ``user/name`` model repo id.

  Returns:
    The file-tree URL, or None when ``repo_id`` is empty.
  """
  if not repo_id:
    return None
  return f"{_HF}/{repo_id}/tree/main"


def dataset_url(repo_id: Optional[str]) -> Optional[str]:
  """Returns the Hub page of a dataset repository.

  Args:
    repo_id: A ``user/name`` dataset repo id.

  Returns:
    The dataset URL, or None when ``repo_id`` is empty.
  """
  if not repo_id:
    return None
  return f"{_HF}/datasets/{repo_id}"


def eval_dataset_repo_id(
    campaign: index_mod.CampaignSummary, target: index_mod.EvalTarget
) -> Optional[str]:
  """Recomputes the completions dataset an evaluation target pushed to.

  Reproduces ``EvalStage._summary_path``: same user, same writer adapter,
  same temperature, same few-shot count, same seed and token cap, and the
  same rule for whether an SFT adapter was stacked underneath.

  Args:
    campaign: The campaign the target belongs to.
    target: The evaluated target.

  Returns:
    The dataset repo id, or None when it cannot be derived (a delta column,
    a target whose model is unknown, or a missing ``src.utils``).
  """
  if build_eval_dataset_repo_id is None or target.is_delta:
    return None
  if not target.model_repo_id or not campaign.user:
    return None

  eval_cfg = campaign.config.get("eval")
  eval_cfg = eval_cfg if isinstance(eval_cfg, dict) else {}
  temperature = target.temperature
  if temperature is None:
    temperature = eval_cfg.get("temperature", 0.0)

  sft_stage = next(
      (s for s in campaign.stages_of_kind("sft") if s.model_repo_id), None
  )
  sft_repo = sft_stage.model_repo_id if sft_stage else None
  # Mirrors `_stacked_sft_repo_id`: evaluating the SFT adapter itself stacks
  # nothing, so the marker differs and so does the repo name.
  stacked = sft_repo if sft_repo and sft_repo != target.model_repo_id else None

  try:
    return build_eval_dataset_repo_id(
        user=campaign.user,
        writer_model_lora=target.model_repo_id,
        temperature=float(temperature),
        writer_num_fewshot=int(eval_cfg.get("writer_num_fewshot", 0) or 0),
        task_name=campaign.task,
        sft_model_path=stacked,
        seed=int(eval_cfg.get("seed", 12345) or 12345),
        max_tokens=int(eval_cfg.get("max_tokens", 250) or 250),
    )
  except Exception:  # pylint: disable=broad-except
    # A naming helper that raises must not take the whole page down.
    return None


def training_dataset_repo_ids(
    campaign: index_mod.CampaignSummary,
) -> List[tuple]:
  """Returns the Hub datasets a campaign trained and calibrated on.

  These mirror the ids the stages inject into their sweep commands
  (``{user}/{task}_sft``, ``{user}/{task}_rm_{flavor}``,
  ``{user}/{task}_perl``, ``{user}/{task}_autorater``).

  Args:
    campaign: The campaign.

  Returns:
    A list of ``(label, repo_id)`` pairs, empty when the user is unknown.
  """
  if not campaign.user or not campaign.task:
    return []
  user, task = campaign.user, campaign.task
  pairs: List[tuple] = []
  kinds = {stage.kind for stage in campaign.stages}
  if "autorater" in kinds:
    pairs.append(("Autorater calibration set", f"{user}/{task}_autorater"))
  if "sft" in kinds:
    pairs.append(("SFT training set", f"{user}/{task}_sft"))
  for stage in campaign.stages_of_kind("rm"):
    flavor = stage.flavor or (
        campaign.rm_dataset_flavors[0]
        if campaign.rm_dataset_flavors
        else "organic"
    )
    label = f"Reward model training set ({flavor.replace('_', ' ')})"
    pairs.append((label, f"{user}/{task}_rm_{flavor}"))
  if "perl" in kinds:
    pairs.append(("PE-RL prompt set", f"{user}/{task}_perl"))
  return pairs


def _temperature_note(target: index_mod.EvalTarget) -> str:
  """Returns a short human note describing a target's decoding temperature.

  Args:
    target: The evaluated target.

  Returns:
    A note such as ``"greedy"`` or ``"T=0.7"``, or an empty string.
  """
  if target.temperature is None:
    return ""
  if float(target.temperature) == 0.0:
    return "greedy"
  return f"T={target.temperature:g}"


def link_groups(campaign: index_mod.CampaignSummary) -> List[LinkGroup]:
  """Assembles every outbound link for a campaign's detail page.

  Args:
    campaign: The campaign.

  Returns:
    Non-empty link groups, in the order they should be rendered.
  """
  groups: List[LinkGroup] = []

  wandb_links: List[Link] = []
  for stage in campaign.stages:
    url = sweep_url(campaign, stage)
    if url:
      wandb_links.append(
          Link(
              label=f"{stage.title} sweep",
              url=url,
              note=stage.sweep_name or "",
          )
      )
  for stage in campaign.stages:
    url = run_url(campaign, stage)
    if url:
      wandb_links.append(
          Link(label=f"{stage.title} winning run", url=url, note="tagged best")
      )
  project = project_url(campaign)
  if project:
    wandb_links.append(Link(label="Project", url=project))
  evaluation = eval_project_url(campaign)
  if evaluation:
    wandb_links.append(Link(label="Evaluation project", url=evaluation))
  if wandb_links:
    groups.append(LinkGroup(title="Weights & Biases", links=wandb_links))

  model_links: List[Link] = []
  for stage in campaign.stages:
    if not stage.model_repo_id or stage.kind == "eval":
      continue
    url = model_url(stage.model_repo_id)
    if url:
      model_links.append(
          Link(
              label=f"{stage.title} adapter",
              url=url,
              note=stage.model_repo_id,
          )
      )
      files = model_files_url(stage.model_repo_id)
      if files:
        model_links.append(
            Link(
                label=f"{stage.title} files",
                url=files,
                note="companion checkpoint + checkpoints.json",
            )
        )
  if model_links:
    groups.append(LinkGroup(title="Hugging Face models", links=model_links))

  generation_links: List[Link] = []
  for target in campaign.eval_targets:
    repo = eval_dataset_repo_id(campaign, target)
    url = dataset_url(repo)
    if url:
      generation_links.append(
          Link(
              label=f"{target.label} generations",
              url=url,
              note=_temperature_note(target),
              derived=True,
          )
      )
  if generation_links:
    groups.append(
        LinkGroup(title="Hugging Face generations", links=generation_links)
    )

  data_links = [
      Link(label=label, url=dataset_url(repo) or "", note=repo, derived=True)
      for label, repo in training_dataset_repo_ids(campaign)
  ]
  data_links = [link for link in data_links if link.url]
  if data_links:
    groups.append(LinkGroup(title="Hugging Face datasets", links=data_links))

  return groups
