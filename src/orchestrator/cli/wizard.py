"""Interactive setup wizard for configuring and launching campaigns.

The wizard is split into three layers so that the interesting logic can be
tested without a terminal:

  * :class:`Prompter` - a tiny abstraction over the question widgets. Three
    implementations exist: ``questionary`` (arrow keys, validation), a plain
    ``input()`` fallback, and :class:`ScriptedPrompter` used by the tests.
  * Pure helpers - :func:`build_config`, :func:`estimate_runtime`,
    :func:`equivalent_command`, :func:`config_summary_lines`.
  * :func:`run_setup_wizard` - the orchestration of the conversation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli.console import UiConsole
from src.orchestrator.cli.renderables import banner
from src.orchestrator import config as config_mod
from src.orchestrator import flavors
from src.orchestrator.config import CampaignConfig, VALID_TASKS

#: Human friendly task descriptions shown in the picker.
TASK_DESCRIPTIONS: Dict[str, str] = {
    "npov": "Neutral Point of View rewriting (Wikipedia NPOV edits)",
    "bosch": "Factuality / hallucination reduction on Bosch reports",
    "ragtruth": "RAG factuality benchmark (all sub-tasks)",
    "ragtruth-qa": "RAG factuality - question answering split",
    "ragtruth-summarization": "RAG factuality - summarization split",
}

#: Stage catalogue: key, label, one-line description.
STAGE_CATALOGUE: List[Tuple[str, str, str]] = [
    (
        "autorater",
        "Autorater calibration",
        "Scores the Gemini judge against human labels, fits its threshold",
    ),
    ("sft", "SFT sweep", "Supervised fine-tuning of the writer policy"),
    ("rm", "Reward model sweep", "Trains the preference/reward model"),
    ("perl", "PE-RL sweep", "RLOO policy optimization with LoRA adapters"),
    ("eval", "Final evaluation", "Gemini autorater + BertScore + perplexity"),
]

#: One-line descriptions of the reward-model training datasets, shown in the
#: picker. Selecting both fans the campaign out into two RM+PE-RL branches.
RM_FLAVOR_DESCRIPTIONS: Dict[str, str] = {
    flavors.ORGANIC: "Human-written hallucinations as labelled in the corpus",
    flavors.SYNTHETIC_STRUCT:
        "Hallucinations injected by structured perturbation of the answers",
}

#: Budget presets. Values are (sft_runs, rm_runs, perl_runs, eval_samples).
PRESETS: Dict[str, Tuple[int, int, int, int]] = {
    "smoke": (1, 1, 1, 20),
    "quick": (5, 5, 3, 200),
    "standard": (30, 30, 10, 1000),
    "thorough": (60, 60, 20, 2000),
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    "smoke": "1/1/1 trials, 20 eval samples - wiring check (minutes)",
    "quick": "5/5/3 trials, 200 eval samples - fast signal (a few hours)",
    "standard": "30/30/10 trials, 1000 eval samples - project default",
    "thorough": "60/60/20 trials, 2000 eval samples - publication run",
    "custom": "Pick the trial budget for each stage by hand",
}

#: Rough per-trial wall-clock estimates (minutes) used for the ETA preview.
#: Shared with :func:`src.orchestrator.config.sweep_timeout_minutes` so the
#: advertised estimate and the sweep timeout budget stay consistent.
MINUTES_PER_TRIAL: Dict[str, float] = config_mod.MINUTES_PER_TRIAL

_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def validate_repo_id(value: str) -> Optional[str]:
  """Validates a Hugging Face repo id; returns an error message or ``None``."""
  text = (value or "").strip()
  if not text:
    return "A checkpoint id is required (format: owner/model_name)."
  if text.startswith("./") or text.startswith("/"):
    return None  # Local path: accepted as-is.
  if not _REPO_ID_RE.match(text):
    return "Expected a Hugging Face repo id like 'leobianco/npov_SFT_gemma'."
  return None


def validate_positive_int(value: str, maximum: int = 10000) -> Optional[str]:
  """Validates a positive integer entry; returns an error message or ``None``."""
  text = (value or "").strip()
  if not text.isdigit():
    return "Please enter a whole number."
  number = int(text)
  if number <= 0:
    return "The value must be greater than zero."
  if number > maximum:
    return f"That looks too large (max {maximum})."
  return None


def estimate_runtime(config: CampaignConfig) -> Tuple[float, float]:
  """Estimates the campaign wall-clock time in hours as a ``(low, high)`` range.

  The numbers are intentionally coarse - their purpose is to stop someone
  from casually launching a 40 hour job before lunch. That is also why the
  reward-model dataset fan-out is counted: selecting two flavors runs the RM
  and PE-RL sweeps twice, and an estimate that ignored it would understate
  the most expensive campaigns by half.

  Evaluation is priced per *pass*, not per stage: one for the SFT baseline and
  one per PE-RL branch. With ``match_perl_rollout_temperature`` on there may
  be up to one further baseline pass per branch, but how many is not knowable
  until the sweeps have picked their temperatures, so that cost is carried by
  the upper bound alone.

  Args:
    config: The campaign to price.

  Returns:
    A ``(low, high)`` estimate in hours.
  """
  minutes = 0.0
  # Speculative work that may or may not happen; only widens the upper bound.
  optimistic_extra = 0.0
  branches = len(flavors.campaign_flavors(config))
  for stage in theme_mod.iter_stage_names(config.stages):
    if stage == "eval":
      samples = getattr(config.eval, "max_eval_samples", 0) or 0
      per_pass = 15.0 + samples / 100.0 * 2.0
      minutes += per_pass * (1 + branches)
      if getattr(config.eval, "match_perl_rollout_temperature", False):
        # Worst case: every branch wins at a different temperature, none of
        # them the configured one, so each needs its own extra baseline.
        optimistic_extra += per_pass * branches
      continue
    if stage == "autorater":
      # One judging pass over the labelled set - no generation, no GPU, so
      # cheaper than an evaluation target but not free.
      samples = getattr(config.eval, "max_eval_samples", 0) or 0
      minutes += 5.0 + samples / 100.0 * 1.5
      continue
    stage_cfg = getattr(config, stage, None)
    runs = int(getattr(stage_cfg, "max_runs", 0) or 0)
    repeats = branches if stage in flavors.BRANCHED_KINDS else 1
    minutes += repeats * runs * MINUTES_PER_TRIAL.get(stage, 10.0)
  if config.dry_run:
    return (0.02, 0.05)
  return (minutes / 60.0 * 0.6, (minutes + optimistic_extra) / 60.0 * 1.5)


def format_estimate(low: float, high: float) -> str:
  """Formats an hour range for display."""
  if high < 1.0:
    return f"~{int(round(low * 60))}-{int(round(high * 60))} min"
  return f"~{low:.1f}-{high:.1f} h"


def equivalent_command(config: CampaignConfig) -> str:
  """Builds the non-interactive command reproducing this configuration."""
  parts = ["python3 scripts/run_campaign.py run", f"--task {config.task_name}"]
  stages = theme_mod.iter_stage_names(config.stages)
  if stages != list(config_mod.VALID_STAGES):
    parts.append(f"--stages {','.join(stages)}")
  if "sft" in stages and config.sft.max_runs != 30:
    parts.append(f"--sft-runs {config.sft.max_runs}")
  if "rm" in stages and config.rm.max_runs != 30:
    parts.append(f"--rm-runs {config.rm.max_runs}")
  if "perl" in stages and config.perl.max_runs != 10:
    parts.append(f"--perl-runs {config.perl.max_runs}")
  if config.perl.sft_model_path and config.perl.sft_model_path != "auto":
    parts.append(f'--sft-model "{config.perl.sft_model_path}"')
  if config.perl.reward_model_path and config.perl.reward_model_path != "auto":
    parts.append(f'--reward-model "{config.perl.reward_model_path}"')
  campaign_flavors = flavors.campaign_flavors(config)
  if campaign_flavors != [flavors.ORGANIC]:
    parts.append(f"--rm-datasets {','.join(campaign_flavors)}")
  if config.eval.max_eval_samples != 1000:
    parts.append(f"--eval-samples {config.eval.max_eval_samples}")
  if config.dry_run:
    parts.append("--dry-run")
  return " ".join(parts)


def config_summary_lines(config: CampaignConfig, theme: theme_mod.Theme) -> List[str]:
  """Renders the human-readable review block shown before launching."""
  stages = theme_mod.iter_stage_names(config.stages)
  low, high = estimate_runtime(config)
  rows: List[Tuple[str, str]] = [
      ("Task", f"{config.task_name} - {TASK_DESCRIPTIONS.get(config.task_name, '')}"),
      ("Base model", config.base_model),
      ("Campaign id", config.name),
      ("Stages", " " + f" {theme.glyphs.arrow} ".join(s.upper() for s in stages)),
  ]
  # A two-flavor campaign silently doubles the RM and PE-RL budgets, so the
  # review block has to say so before the user confirms the estimate.
  campaign_flavors = flavors.campaign_flavors(config)
  if "rm" in stages or len(campaign_flavors) > 1:
    rows.append(("RM dataset", flavors.describe_flavors(campaign_flavors)))
  for stage in stages:
    if stage == "eval":
      if config.eval.match_perl_rollout_temperature:
        decoding = "PE-RL at its own rollout temperature, SFT matched"
      else:
        decoding = f"every target at temperature {config.eval.temperature}"
      rows.append(("Decoding", decoding))
      rows.append(
          ("Eval budget", f"{config.eval.max_eval_samples} samples "
                          f"({config.eval.evaluator_model})")
      )
    elif stage == "autorater":
      # No budget row: calibration has no sweep and no configuration of its
      # own, by design - it must use the evaluation's judge settings for
      # its fitted threshold to transfer.
      rows.append((
          "Judge check",
          f"{config.eval.evaluator_model} vs human labels, warn below "
          f"ROC-AUC {config.eval.min_autorater_auc:.2f}",
      ))
    else:
      stage_cfg = getattr(config, stage)
      rows.append(
          (
              f"{stage.upper()} budget",
              f"{stage_cfg.max_runs} trials, optimize {stage_cfg.metric} "
              f"({stage_cfg.goal})",
          )
      )
  if "perl" in stages:
    if config.perl.sft_model_path and config.perl.sft_model_path != "auto":
      rows.append(("SFT checkpoint", config.perl.sft_model_path))
    if config.perl.reward_model_path and config.perl.reward_model_path != "auto":
      rows.append(("RM checkpoint", config.perl.reward_model_path))
  rows.append(("Seed", str(config.seed)))
  rows.append(("Mode", "DRY-RUN (no GPU work)" if config.dry_run else "LIVE"))
  rows.append(("Estimated time", format_estimate(low, high)))
  rows.append(("State file", config.state_file or "-"))
  # Only shown when armed. A machine that powers itself off is the single
  # most surprising thing this tool can do, so it has to appear in the block
  # the user confirms, not only in the YAML they may have inherited.
  if getattr(config, "shutdown_when_done", False):
    grace = getattr(config, "shutdown_grace_seconds", 0)
    rows.append((
        "On finish",
        f"POWER OFF the VM after a {grace}s cancellable countdown",
    ))

  width = max(len(label) for label, _ in rows)
  return [
      f"{theme.markup(label.ljust(width), 'muted')}  {theme.markup(str(value), 'value')}"
      for label, value in rows
  ]


@dataclass
class WizardAnswers:
  """Raw answers collected from the user."""

  task: str = "npov"
  stages: List[str] = field(
      default_factory=lambda: list(config_mod.VALID_STAGES)
  )
  preset: str = "standard"
  sft_runs: int = 30
  rm_runs: int = 30
  perl_runs: int = 10
  eval_samples: int = 1000
  sft_model: str = "auto"
  reward_model: str = "auto"
  rm_dataset_flavors: List[str] = field(
      default_factory=lambda: [flavors.ORGANIC]
  )
  dry_run: bool = False

  def to_dict(self) -> Dict[str, Any]:
    return dataclasses.asdict(self)


def build_config(answers: WizardAnswers) -> CampaignConfig:
  """Turns wizard answers into a validated :class:`CampaignConfig`."""
  config = CampaignConfig.create_default(
      task_name=answers.task,
      sft_runs=answers.sft_runs,
      rm_runs=answers.rm_runs,
      perl_runs=answers.perl_runs,
      dry_run=answers.dry_run,
      rm_dataset_flavors=answers.rm_dataset_flavors,
  )
  config.stages = theme_mod.iter_stage_names(answers.stages)
  config.perl.sft_model_path = answers.sft_model or "auto"
  config.perl.reward_model_path = answers.reward_model or "auto"
  config.eval.max_eval_samples = int(answers.eval_samples)
  config.validate()
  return config


# --------------------------------------------------------------------------
# Prompters
# --------------------------------------------------------------------------
class Prompter:
  """Question widget interface used by the wizard."""

  def select(
      self, message: str, choices: Sequence[Tuple[str, str]], default: Optional[str] = None
  ) -> Optional[str]:
    raise NotImplementedError

  def checkbox(
      self, message: str, choices: Sequence[Tuple[str, str, bool]]
  ) -> Optional[List[str]]:
    raise NotImplementedError

  def text(
      self,
      message: str,
      default: str = "",
      validate: Optional[Callable[[str], Optional[str]]] = None,
  ) -> Optional[str]:
    raise NotImplementedError

  def confirm(self, message: str, default: bool = True) -> Optional[bool]:
    raise NotImplementedError


class QuestionaryPrompter(Prompter):
  """Arrow-key driven prompts backed by ``questionary``/``prompt_toolkit``."""

  def __init__(self, module: Any):
    self.q = module

  def select(self, message, choices, default=None):
    values = [value for value, _ in choices]
    labels = {value: label for value, label in choices}
    answer = self.q.select(
        message,
        choices=[labels[value] for value in values],
        default=labels.get(default) if default in labels else None,
        instruction="(↑/↓ then Enter)",
    ).ask()
    if answer is None:
      return None
    for value, label in labels.items():
      if label == answer:
        return value
    return None

  def checkbox(self, message, choices):
    labels = {label: value for value, label, _ in choices}
    answer = self.q.checkbox(
        message,
        choices=[
            {"name": label, "checked": checked} for _, label, checked in choices
        ],
        instruction="(space to toggle, Enter to accept)",
    ).ask()
    if answer is None:
      return None
    return [labels[label] for label in answer if label in labels]

  def text(self, message, default="", validate=None):
    def _validate(value: str) -> Any:
      if validate is None:
        return True
      error = validate(value)
      return True if error is None else error

    return self.q.text(message, default=str(default), validate=_validate).ask()

  def confirm(self, message, default=True):
    return self.q.confirm(message, default=default).ask()


class PlainPrompter(Prompter):
  """Numbered-menu fallback used when ``questionary`` is not installed."""

  def __init__(self, console: UiConsole, input_fn: Callable[[str], str] = input):
    self.console = console
    self.input_fn = input_fn
    self.theme = console.theme

  def _ask(self, prompt: str) -> Optional[str]:
    try:
      return self.input_fn(theme_mod.strip_markup(prompt))
    except (EOFError, KeyboardInterrupt):
      return None

  def select(self, message, choices, default=None):
    self.console.print(self.theme.markup(f"? {message}", "heading"))
    for index, (value, label) in enumerate(choices, start=1):
      marker = "*" if value == default else " "
      self.console.print(
          f"  {self.theme.markup(str(index), 'accent')}{marker} {label}"
      )
    default_index = next(
        (i for i, (value, _) in enumerate(choices, start=1) if value == default),
        1,
    )
    while True:
      raw = self._ask(f"  Choice [1-{len(choices)}] (default {default_index}): ")
      if raw is None:
        return None
      raw = raw.strip()
      if not raw:
        return choices[default_index - 1][0]
      if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return choices[int(raw) - 1][0]
      for value, _ in choices:
        if raw.lower() == value.lower():
          return value
      self.console.warn("Invalid choice, try again.")

  def checkbox(self, message, choices):
    self.console.print(self.theme.markup(f"? {message}", "heading"))
    for index, (_, label, checked) in enumerate(choices, start=1):
      # The mark is escaped: "[x]" would otherwise be parsed as a style tag
      # and silently removed from the plain-text rendering.
      mark = theme_mod.escape_markup("[x]" if checked else "[ ]")
      self.console.print(
          f"  {self.theme.markup(str(index), 'accent')} {mark} {label}"
      )
    defaults = [value for value, _, checked in choices if checked]
    raw = self._ask(
        "  Comma-separated numbers (Enter keeps the checked ones): "
    )
    if raw is None:
      return None
    raw = raw.strip()
    if not raw:
      return defaults
    picked: List[str] = []
    for token in raw.split(","):
      token = token.strip()
      if token.isdigit() and 1 <= int(token) <= len(choices):
        picked.append(choices[int(token) - 1][0])
      else:
        for value, _, _ in choices:
          if token.lower() == value.lower():
            picked.append(value)
    # Deliberately *not* ordered here: this widget also asks non-stage
    # questions, and the stage caller canonicalises its own answer.
    deduped: List[str] = []
    for value in picked:
      if value not in deduped:
        deduped.append(value)
    return deduped

  def text(self, message, default="", validate=None):
    hint = theme_mod.escape_markup(f"[{default}]")
    while True:
      raw = self._ask(f"? {message} {hint}: ")
      if raw is None:
        return None
      value = raw.strip() or str(default)
      if validate is not None:
        error = validate(value)
        if error:
          self.console.warn(error)
          continue
      return value

  def confirm(self, message, default=True):
    hint = theme_mod.escape_markup("[Y/n]" if default else "[y/N]")
    raw = self._ask(f"? {message} {hint}: ")
    if raw is None:
      return None
    raw = raw.strip().lower()
    if not raw:
      return default
    return raw.startswith("y")


class ScriptedPrompter(Prompter):
  """Deterministic prompter replaying canned answers (used in tests)."""

  def __init__(self, answers: Sequence[Any]):
    self.answers = list(answers)
    self.asked: List[str] = []

  def _next(self, message: str) -> Any:
    self.asked.append(message)
    if not self.answers:
      return None
    return self.answers.pop(0)

  def select(self, message, choices, default=None):
    return self._next(message)

  def checkbox(self, message, choices):
    return self._next(message)

  def text(self, message, default="", validate=None):
    value = self._next(message)
    if value is None:
      return None
    if validate is not None:
      error = validate(str(value))
      if error:
        raise AssertionError(f"Scripted answer rejected: {error}")
    return str(value)

  def confirm(self, message, default=True):
    value = self._next(message)
    return default if value is None else bool(value)


def make_prompter(console: UiConsole) -> Prompter:
  """Returns the best available prompter for the current environment."""
  try:
    import questionary  # pylint: disable=g-import-not-at-top

    return QuestionaryPrompter(questionary)
  except ImportError:
    return PlainPrompter(console)


# --------------------------------------------------------------------------
# Wizard flow
# --------------------------------------------------------------------------
def run_setup_wizard(
    console: Optional[UiConsole] = None,
    prompter: Optional[Prompter] = None,
    save_yaml: bool = True,
) -> Optional[CampaignConfig]:
  """Runs the interactive wizard.

  Args:
    console: Output surface (created from terminal capabilities if omitted).
    prompter: Question widget implementation; auto-detected if omitted.
    save_yaml: Whether to offer saving the configuration to ``configs/``.

  Returns:
    The assembled :class:`CampaignConfig`, or ``None`` if the user cancelled.
  """
  console = console or UiConsole()
  theme = console.theme
  prompter = prompter or make_prompter(console)

  console.blank()
  console.print(banner(theme, "Automated PE-RL scientific campaign orchestrator"))
  console.blank()

  answers = WizardAnswers()

  # 1. Task ------------------------------------------------------------
  task = prompter.select(
      "Which task do you want to optimize?",
      [
          (name, f"{name:<24} {TASK_DESCRIPTIONS.get(name, '')}")
          for name in VALID_TASKS
      ],
      default="npov",
  )
  if not task:
    return _cancelled(console)
  answers.task = task

  # 2. Stages ----------------------------------------------------------
  stages = prompter.checkbox(
      "Which stages should this campaign run?",
      [
          (key, f"{label:<22} {description}", True)
          for key, label, description in STAGE_CATALOGUE
      ],
  )
  if stages is None:
    return _cancelled(console)
  stages = theme_mod.iter_stage_names(stages)
  if not stages:
    console.error("No stages selected - nothing to run.")
    return None
  answers.stages = stages

  # 2b. Reward-model dataset ------------------------------------------
  # Only meaningful when this campaign actually trains a reward model. If
  # PE-RL is reusing an existing RM checkpoint there is no dataset to pick,
  # and offering two flavors would promise a fan-out we could not deliver
  # from a single `--reward-model` path.
  if "rm" in stages:
    picked = prompter.checkbox(
        "Which reward-model training dataset(s)?",
        [
            (
                flavor,
                f"{flavors.flavor_title(flavor):<18} "
                f"{RM_FLAVOR_DESCRIPTIONS.get(flavor, '')}",
                flavor == flavors.ORGANIC,
            )
            for flavor in flavors.RM_DATASET_FLAVORS
        ],
    )
    if picked is None:
      return _cancelled(console)
    picked = flavors.normalize_flavors(picked)
    if not picked:
      console.error("No reward-model dataset selected - nothing to train on.")
      return None
    answers.rm_dataset_flavors = picked
    if len(picked) > 1:
      console.hint(
          "Two datasets selected: the campaign will run one RM sweep and "
          "one PE-RL sweep per dataset, then score both policies in a "
          "single evaluation."
      )

  # 3. Budget ----------------------------------------------------------
  preset = prompter.select(
      "Pick a trial budget preset:",
      [
          (name, f"{name:<10} {PRESET_DESCRIPTIONS[name]}")
          for name in ["smoke", "quick", "standard", "thorough", "custom"]
      ],
      default="standard",
  )
  if not preset:
    return _cancelled(console)
  answers.preset = preset
  if preset in PRESETS:
    answers.sft_runs, answers.rm_runs, answers.perl_runs, answers.eval_samples = (
        PRESETS[preset]
    )
  else:
    if "sft" in stages:
      value = prompter.text(
          "Max trials for the SFT sweep",
          default="30",
          validate=validate_positive_int,
      )
      if value is None:
        return _cancelled(console)
      answers.sft_runs = int(value)
    if "rm" in stages:
      value = prompter.text(
          "Max trials for the reward model sweep",
          default="30",
          validate=validate_positive_int,
      )
      if value is None:
        return _cancelled(console)
      answers.rm_runs = int(value)
    if "perl" in stages:
      value = prompter.text(
          "Max trials for the PE-RL sweep",
          default="10",
          validate=validate_positive_int,
      )
      if value is None:
        return _cancelled(console)
      answers.perl_runs = int(value)
    if "eval" in stages:
      value = prompter.text(
          "Evaluation samples",
          default="1000",
          validate=lambda v: validate_positive_int(v, maximum=100000),
      )
      if value is None:
        return _cancelled(console)
      answers.eval_samples = int(value)

  # 4. Upstream checkpoints -------------------------------------------
  if "perl" in stages and "sft" not in stages:
    value = prompter.text(
        "Existing SFT checkpoint to start PE-RL from",
        default=f"leobianco/{answers.task}_SFT_",
        validate=validate_repo_id,
    )
    if value is None:
      return _cancelled(console)
    answers.sft_model = value
  if "perl" in stages and "rm" not in stages:
    value = prompter.text(
        "Existing reward model checkpoint",
        default=f"leobianco/{answers.task}_RM_",
        validate=validate_repo_id,
    )
    if value is None:
      return _cancelled(console)
    answers.reward_model = value

  # 5. Dry run ---------------------------------------------------------
  dry_run = prompter.confirm(
      "Dry-run first (simulate the whole DAG, no GPU work)?", default=False
  )
  if dry_run is None:
    return _cancelled(console)
  answers.dry_run = bool(dry_run)

  # 6. Review loop -----------------------------------------------------
  config = build_config(answers)
  console.blank()
  console.panel(
      config_summary_lines(config, theme), title="Campaign review", style="accent"
  )
  console.blank()
  console.print(theme.markup("Equivalent non-interactive command:", "muted"))
  console.print(f"  {theme.markup(equivalent_command(config), 'accent')}")
  console.blank()

  low, high = estimate_runtime(config)
  if high > 12 and not config.dry_run:
    console.warn(
        f"This campaign may run for {format_estimate(low, high)}. "
        "Consider tmux + 'run_campaign.py status --watch' in a second pane."
    )

  if save_yaml:
    should_save = prompter.confirm(
        "Save this configuration to configs/ for reproducibility?", default=True
    )
    if should_save:
      path = os.path.join("configs", f"campaign_{config.task_name}.yaml")
      try:
        config.to_yaml(path)
        console.success(f"Configuration saved to {path}")
        console.hint(f"Re-run it later with: --config {path}")
      except OSError as exc:
        console.warn(f"Could not save configuration: {exc}")

  launch = prompter.confirm("Launch this campaign now?", default=True)
  if not launch:
    console.info("Nothing launched. The configuration above is ready when you are.")
    return None
  return config


def _cancelled(console: UiConsole) -> None:
  """Prints a cancellation notice and returns ``None``."""
  console.blank()
  console.info("Wizard cancelled - no campaign was started.")
  return None
