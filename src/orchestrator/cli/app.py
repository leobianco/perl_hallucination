"""Command line application for the Auto-PERL campaign orchestrator.

Living in the package (rather than inside ``scripts/``) keeps the CLI
importable and therefore testable: the unit tests call :func:`main` with
argument vectors and an in-memory console, and assert on the rendered text.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import logging
import os
import shutil
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.orchestrator import flavors
from src.orchestrator import shutdown
from src.orchestrator.cli import renderables
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.cli.console import UiConsole
from src.orchestrator.config import CampaignConfig, VALID_TASKS
from src.orchestrator.state import CampaignState

logger = logging.getLogger(__name__)

PROGRAM = "run_campaign.py"
CHECKPOINTS_ROOT = "./checkpoints"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def _global_flags() -> argparse.ArgumentParser:
  """Flags accepted both before and after the subcommand.

  ``default=SUPPRESS`` is essential: these options are registered on both the
  top-level parser and every subparser, and without it the subparser would
  re-apply its own ``False`` default on top of a flag the user typed *before*
  the subcommand (``run_campaign.py --no-color status``).

  Returns:
    A parent parser holding the shared appearance/debug flags.
  """
  parent = argparse.ArgumentParser(add_help=False)
  parent.add_argument(
      "--no-color",
      action="store_true",
      default=argparse.SUPPRESS,
      help="Disable ANSI colors (also honors the NO_COLOR env var).",
  )
  parent.add_argument(
      "--ascii",
      action="store_true",
      default=argparse.SUPPRESS,
      help="Use ASCII-only glyphs for terminals without Unicode support.",
  )
  parent.add_argument(
      "--plain",
      action="store_true",
      default=argparse.SUPPRESS,
      help="Plain text output: no live dashboard, no styling.",
  )
  parent.add_argument(
      "--debug",
      action="store_true",
      default=argparse.SUPPRESS,
      help="Show full tracebacks instead of friendly error messages.",
  )
  return parent


def _apply_global_defaults(args: argparse.Namespace) -> argparse.Namespace:
  """Fills in the suppressed global flags so callers can read them freely."""
  for name in ("no_color", "ascii", "plain", "debug"):
    if not hasattr(args, name):
      setattr(args, name, False)
  return args


def build_parser() -> argparse.ArgumentParser:
  """Builds the full argument parser."""
  parent = _global_flags()
  parser = argparse.ArgumentParser(
      prog=PROGRAM,
      parents=[parent],
      description=(
          "Auto-PERL: automated scientific campaign orchestrator.\n"
          "Run 'wizard' for a guided setup, 'doctor' for a pre-flight check."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=(
          "Examples:\n"
          f"  {PROGRAM} wizard\n"
          f"  {PROGRAM} doctor --task npov\n"
          f"  {PROGRAM} run --task npov --preset quick\n"
          f"  {PROGRAM} run --task npov --stages perl,eval --sft-model owner/m\n"
          f"  {PROGRAM} status --task npov --watch\n"
          f"  {PROGRAM} resume --task npov\n"
      ),
  )
  subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

  # --- run ---
  run_parser = subparsers.add_parser(
      "run", parents=[parent], help="Launch a campaign."
  )
  run_parser.add_argument("--task", "-t", default="npov", choices=VALID_TASKS,
                          help="Task identifier.")
  run_parser.add_argument("--config", "-c", default=None,
                          help="Path to a YAML campaign configuration.")
  run_parser.add_argument("--stages", "-s", default=None,
                          help="Comma-separated stages, e.g. 'sft,rm,perl,eval'.")
  run_parser.add_argument("--preset", default=None,
                          choices=["smoke", "quick", "standard", "thorough"],
                          help="Trial budget preset (overridden by explicit flags).")
  run_parser.add_argument("--sft-runs", type=int, default=None,
                          help="Max trials for the SFT sweep.")
  run_parser.add_argument("--rm-runs", type=int, default=None,
                          help="Max trials for the reward model sweep.")
  run_parser.add_argument("--perl-runs", type=int, default=None,
                          help="Max trials for the PE-RL sweep.")
  run_parser.add_argument("--eval-samples", type=int, default=None,
                          help="Number of evaluation samples.")
  run_parser.add_argument("--sft-model", default=None,
                          help="Existing SFT checkpoint (skips SFT training).")
  run_parser.add_argument("--reward-model", default=None,
                          help="Existing reward model checkpoint.")
  run_parser.add_argument(
      "--rm-datasets", default=None,
      help=(
          "Comma-separated reward-model training datasets: "
          + ",".join(flavors.RM_DATASET_FLAVORS)
          + ". Naming two runs an RM and a PE-RL sweep for each, then "
          "scores both policies in one evaluation."
      ),
  )
  run_parser.add_argument("--dry-run", action="store_true",
                          help="Simulate the DAG without GPU workloads.")
  run_parser.add_argument("--no-tui", action="store_true",
                          help="Disable the live dashboard (plain streaming).")
  run_parser.add_argument("--entity", "--wandb-entity", default=None,
                          help="Weights & Biases entity (username or team). Defaults to WANDB_ENTITY or authenticated user.")
  run_parser.add_argument("--user", "--hf-user", default=None,
                          help="Hugging Face Hub username/namespace for model uploads (default: leobianco).")
  run_parser.add_argument("--yes", "-y", action="store_true",
                          help="Skip the pre-launch confirmation.")
  run_parser.add_argument("--interactive", "-i", action="store_true",
                          help="Start from the guided wizard.")
  # Overrides shutdown_when_done in the YAML, in both directions, so an
  # overnight launch can arm it without editing a file and a daytime launch
  # can disarm a YAML that has it on.
  shutdown_group = run_parser.add_mutually_exclusive_group()
  shutdown_group.add_argument(
      "--shutdown", dest="shutdown", action="store_true", default=None,
      help="Power the VM off when the campaign finishes or fails.",
  )
  shutdown_group.add_argument(
      "--no-shutdown", dest="shutdown", action="store_false", default=None,
      help="Keep the VM up even if the config asks for a shutdown.",
  )
  # What to do when a state file for this campaign already exists. Neither
  # answer is safe to guess: resuming silently adopts stale sweep ids and
  # finished stages, starting over silently discards hours of GPU time.
  existing = run_parser.add_mutually_exclusive_group()
  existing.add_argument(
      "--fresh", action="store_true",
      help=(
          "Start this campaign from scratch, archiving any existing state "
          "for the same campaign name."
      ),
  )
  existing.add_argument(
      "--resume", action="store_true",
      help=(
          "Continue an existing campaign of the same name, keeping completed "
          "stages and running sweeps."
      ),
  )

  # --- wizard ---
  subparsers.add_parser(
      "wizard", parents=[parent], help="Guided interactive setup."
  )

  # --- status ---
  status_parser = subparsers.add_parser(
      "status", parents=[parent], help="Inspect campaign progress."
  )
  status_parser.add_argument("--task", "-t", default="npov", help="Task name.")
  status_parser.add_argument("--state-file", default=None,
                             help="Explicit path to a campaign state JSON.")
  status_parser.add_argument("--json", action="store_true",
                             help="Emit machine-readable JSON.")
  status_parser.add_argument("--watch", action="store_true",
                             help="Refresh continuously (great in a tmux pane).")
  status_parser.add_argument("--interval", type=float, default=5.0,
                             help="Seconds between refreshes in --watch mode.")
  status_parser.add_argument("--max-refreshes", type=int, default=0,
                             help=argparse.SUPPRESS)

  # --- resume ---
  resume_parser = subparsers.add_parser(
      "resume", parents=[parent], help="Resume an interrupted campaign."
  )
  resume_parser.add_argument("--task", "-t", default="npov", help="Task name.")
  resume_parser.add_argument("--state-file", default=None,
                             help="Explicit path to a campaign state JSON.")
  resume_parser.add_argument("--entity", "--wandb-entity", default=None,
                             help="Weights & Biases entity (username or team).")
  resume_parser.add_argument("--user", "--hf-user", default=None,
                             help="Hugging Face Hub username/namespace for model uploads.")
  resume_parser.add_argument("--dry-run", action="store_true",
                             help="Simulate the resumed stages.")
  resume_parser.add_argument("--no-tui", action="store_true",
                             help="Disable the live dashboard.")
  resume_parser.add_argument("--yes", "-y", action="store_true",
                             help="Skip the confirmation prompt.")
  resume_shutdown = resume_parser.add_mutually_exclusive_group()
  resume_shutdown.add_argument(
      "--shutdown", dest="shutdown", action="store_true", default=None,
      help="Power the VM off when the campaign finishes or fails.",
  )
  resume_shutdown.add_argument(
      "--no-shutdown", dest="shutdown", action="store_false", default=None,
      help="Keep the VM up even if the config asks for a shutdown.",
  )

  # --- report ---
  report_parser = subparsers.add_parser(
      "report", parents=[parent], help="Regenerate reports for a campaign."
  )
  report_parser.add_argument("--task", "-t", default="npov", help="Task name.")
  report_parser.add_argument("--state-file", default=None,
                             help="Explicit path to a campaign state JSON.")

  # --- list ---
  list_parser = subparsers.add_parser(
      "list", parents=[parent], aliases=["ls"], help="List known campaigns."
  )
  list_parser.add_argument("--task", "-t", default=None,
                           help="Restrict the listing to one task.")
  list_parser.add_argument("--json", action="store_true",
                           help="Emit machine-readable JSON.")

  # --- doctor ---
  doctor_parser = subparsers.add_parser(
      "doctor", parents=[parent],
      help="Pre-flight environment check before a long campaign."
  )
  doctor_parser.add_argument("--task", "-t", default="npov", choices=VALID_TASKS,
                             help="Task the checks should assume.")
  doctor_parser.add_argument("--entity", "--wandb-entity", default=None,
                             help="W&B entity to verify.")
  doctor_parser.add_argument("--json", action="store_true",
                             help="Emit machine-readable JSON.")
  return parser


# --------------------------------------------------------------------------
# State discovery helpers
# --------------------------------------------------------------------------
def find_state_files(task_name: Optional[str] = None, root: str = CHECKPOINTS_ROOT) -> List[str]:
  """Returns campaign state files, newest first."""
  pattern = os.path.join(root, task_name or "*", "*_state.json")
  candidates = [path for path in glob.glob(pattern) if os.path.isfile(path)]
  candidates.sort(key=os.path.getmtime, reverse=True)
  return candidates


def find_latest_state(task_name: str, root: str = CHECKPOINTS_ROOT) -> Optional[str]:
  """Returns the most recently modified state file for a task."""
  matches = find_state_files(task_name, root=root)
  return matches[0] if matches else None


def config_for_state(state: CampaignState) -> CampaignConfig:
  """Rebuilds the campaign config from the state file.

  Falls back to task defaults for campaigns created before configs were
  persisted, so ``status``/``resume`` keep working on old state files.
  """
  data = getattr(state, "config_dict", None)
  if data:
    try:
      config = CampaignConfig.from_dict(dict(data))
      # A hand-edited or truncated state file must not produce a config that
      # only explodes later, halfway through a stage.
      config.validate()
      config.name = state.campaign_id
      return config
    except (TypeError, ValueError, KeyError, AttributeError):
      pass
  config = CampaignConfig.create_default(task_name=state.task_name)
  config.name = state.campaign_id
  if getattr(state, "stages_order", None):
    # `stages_order` is the *executed plan*, so it may carry branch ids like
    # 'rm:organic'. `config.stages` only ever names stage kinds.
    kinds: List[str] = []
    for stage_id in state.stages_order:
      kind, _ = flavors.split_stage_id(stage_id)
      if kind and kind not in kinds:
        kinds.append(kind)
    config.stages = kinds
  return config


def state_progress(state: CampaignState, config: CampaignConfig) -> Tuple[int, int]:
  """Returns ``(completed_stages, total_stages)``."""
  views = renderables.build_stage_views(config, state)
  done = sum(1 for v in views if str(v.status).upper() in ("COMPLETED", "SKIPPED"))
  return done, len(views)


def archive_state_file(state_file: str) -> str:
  """Moves ``state_file`` aside so a fresh campaign can reuse its name.

  The file is *moved*, never deleted: it is the only record of which
  checkpoints and sweeps a previous campaign produced, and a campaign
  represents hours of GPU time. The archive lives one directory deeper than
  the glob used by ``status``/``resume``/``list`` so archived runs stop
  showing up as live campaigns.

  Args:
    state_file: Path to the state file to retire.

  Returns:
    The path the state file was moved to.
  """
  directory, filename = os.path.split(os.path.abspath(state_file))
  archive_dir = os.path.join(directory, "archive")
  os.makedirs(archive_dir, exist_ok=True)
  stem, extension = os.path.splitext(filename)
  stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  destination = os.path.join(archive_dir, f"{stem}_{stamp}{extension}")
  # Collisions only happen when archiving twice inside one second.
  counter = 1
  while os.path.exists(destination):
    destination = os.path.join(
        archive_dir, f"{stem}_{stamp}_{counter}{extension}"
    )
    counter += 1
  shutil.move(state_file, destination)
  return destination


def describe_existing_state(state_file: str) -> List[str]:
  """Summarizes an existing campaign so the user can choose knowingly.

  Args:
    state_file: Path to the campaign state JSON.

  Returns:
    Human-readable lines; a single fallback line if the file is unreadable.
  """
  try:
    state = CampaignState.load(state_file)
  except (OSError, ValueError, KeyError, TypeError):
    return [f"An unreadable state file already exists at {state_file}."]

  config = config_for_state(state)
  done, total = state_progress(state, config)
  lines = [
      f"Campaign : {state.campaign_id}",
      f"Status   : {getattr(state, 'status', 'UNKNOWN')}",
      f"Progress : {done}/{total} stages completed",
      f"State    : {state_file}",
  ]
  stage_bits = []
  for key in getattr(config, "stages", []) or []:
    result = (getattr(state, "stages", {}) or {}).get(key)
    raw = getattr(result, "status", None) if result else None
    status = getattr(raw, "value", raw) or "PENDING"
    stage_bits.append(f"{key}={str(status).lower()}")
  if stage_bits:
    lines.append("Stages   : " + ", ".join(stage_bits))
  return lines


def resolve_existing_campaign(
    config: CampaignConfig,
    args: argparse.Namespace,
    console: UiConsole,
) -> Optional[int]:
  """Decides what ``run`` should do about a pre-existing state file.

  ``run`` used to load whatever state file matched the campaign name, which
  made two very different operations look identical: continuing a campaign
  and starting one. That is only harmless until the state is stale - a
  deleted W&B sweep, an edited config, a half-finished stage - at which point
  the silent adoption turns into a confusing mid-run failure. Neither answer
  is safe to guess, so this asks (interactively) or refuses (otherwise).

  Args:
    config: The campaign config, already validated. ``config.state_file``
      determines which campaign we might be colliding with.
    args: Parsed ``run`` arguments; reads ``resume``, ``fresh`` and ``yes``.
    console: Console used for the explanation.

  Returns:
    ``None`` when ``run`` should proceed, otherwise the exit code to return.
  """
  state_file = config.state_file
  exists = bool(state_file) and os.path.exists(state_file)

  if not exists:
    if getattr(args, "resume", False):
      # Silently starting fresh here would look like a successful resume and
      # quietly re-run stages the user believed were already done.
      console.error(f"Nothing to resume: no state file at {state_file}.")
      console.hint(
          f"List known campaigns with: {PROGRAM} list. "
          "Start a new one by dropping --resume."
      )
      return EXIT_NOT_FOUND
    return None

  if getattr(args, "resume", False):
    console.info("Resuming the existing campaign:")
    console.print_lines(describe_existing_state(state_file))
    return None

  if getattr(args, "fresh", False):
    destination = archive_state_file(state_file)
    console.info(f"Archived the previous campaign state to {destination}")
    return None

  # No flag: the user has not said which of the two they meant.
  console.warn(f"A campaign named '{config.name}' already exists.")
  console.print_lines(describe_existing_state(state_file))

  if _is_interactive() and not getattr(args, "yes", False):
    console.blank()
    answer = input(
        "[r]esume it, start [f]resh (archives the old state), or [a]bort? "
    ).strip().lower()
    if answer.startswith("r"):
      args.resume = True
      console.info("Resuming the existing campaign.")
      return None
    if answer.startswith("f"):
      args.fresh = True
      destination = archive_state_file(state_file)
      console.info(f"Archived the previous campaign state to {destination}")
      return None
    console.info("Aborted before launch.")
    return EXIT_OK

  console.error("Refusing to guess whether to resume it or start over.")
  console.hint(
      "Re-run with one of:\n"
      "  --resume   continue it, keeping completed stages and sweeps\n"
      "  --fresh    start over, archiving the state file shown above\n"
      "Or pick a different campaign name in your config's 'name:' field."
  )
  return EXIT_USAGE


# --------------------------------------------------------------------------
# Diagnostics (doctor)
# --------------------------------------------------------------------------
def run_diagnostics(config: CampaignConfig) -> List[Dict[str, str]]:
  """Runs environment pre-flight checks.

  Returns:
    A list of ``{"name", "status", "detail"}`` dicts where status is one of
    ``ok`` / ``warn`` / ``fail``.
  """
  checks: List[Dict[str, str]] = []

  def add(name: str, status: str, detail: str) -> None:
    checks.append({"name": name, "status": status, "detail": detail})

  # Python packages.
  required = {
      "wandb": "sweep orchestration",
      "huggingface_hub": "checkpoint publishing",
      "yaml": "sweep/config parsing",
  }
  optional = {
      "rich": "styled dashboard",
      "questionary": "interactive wizard",
      "torch": "training",
  }
  for module, purpose in required.items():
    try:
      __import__(module)
      add(f"package: {module}", "ok", purpose)
    except ImportError:
      add(f"package: {module}", "fail", f"missing - needed for {purpose}")
  for module, purpose in optional.items():
    try:
      __import__(module)
      add(f"package: {module}", "ok", purpose)
    except ImportError:
      add(f"package: {module}", "warn", f"missing - {purpose} will degrade")

  # Credentials.
  if os.environ.get("WANDB_API_KEY") or os.path.exists(
      os.path.expanduser("~/.netrc")
  ):
    try:
      import wandb  # pylint: disable=g-import-not-at-top

      api = wandb.Api()
      default_ent = getattr(api, "default_entity", None)
      target_ent = (
          config.wandb_entity
          or os.environ.get("WANDB_ENTITY")
          or default_ent
      )
      if default_ent:
        add(
            "W&B entity",
            "ok",
            f"authenticated as '{default_ent}' (target: '{target_ent}')",
        )
      else:
        add("W&B credentials", "ok", "API key or ~/.netrc found")
    except Exception as e:
      add("W&B credentials", "ok", f"API key or ~/.netrc found ({e})")
  else:
    add("W&B credentials", "fail", "run 'wandb login' before launching")

  if (
      os.environ.get("HF_TOKEN")
      or os.environ.get("HUGGING_FACE_HUB_TOKEN")
      or os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
  ):
    add("Hugging Face token", "ok", "token available for pushes")
  else:
    add("Hugging Face token", "warn", "no token: model pushes will fail")

  # Autorater credentials. The eval stage runs last, so a missing credential
  # here is only discovered after a full day of GPU time - check it upfront.
  if "eval" in theme_mod.iter_stage_names(config.stages) and config.eval.enabled:
    vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "true").lower()
    if os.environ.get("GEMINI_API_KEY"):
      add("Gemini credentials", "ok", "GEMINI_API_KEY set (AI Studio)")
    elif vertex == "true" and (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.path.exists(
            os.path.expanduser(
                "~/.config/gcloud/application_default_credentials.json"
            )
        )
    ):
      add("Gemini credentials", "ok", "Vertex AI application default creds")
    else:
      add(
          "Gemini credentials",
          "fail",
          "set GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT + "
          "'gcloud auth application-default login' for Vertex",
      )

  # Sweep configuration files.
  for stage in theme_mod.iter_stage_names(config.stages):
    stage_cfg = getattr(config, stage, None)
    path = getattr(stage_cfg, "sweep_config_path", None)
    if not path:
      continue
    if os.path.exists(path):
      add(f"sweep config: {stage}", "ok", path)
    else:
      add(f"sweep config: {stage}", "fail", f"missing file {path}")

  # Writable output directories.
  for label, path in (
      ("checkpoints dir", os.path.dirname(config.state_file or "./checkpoints/x")),
      ("reports dir", config.reporting.reports_dir),
  ):
    try:
      os.makedirs(path, exist_ok=True)
      add(label, "ok", f"writable: {path}")
    except OSError as exc:
      add(label, "fail", f"cannot create {path}: {exc}")

  # Disk space.
  try:
    usage = shutil.disk_usage(".")
    free_gb = usage.free / (1024**3)
    status = "ok" if free_gb >= 50 else ("warn" if free_gb >= 20 else "fail")
    add("disk space", status, f"{free_gb:.0f} GB free")
  except OSError as exc:
    add("disk space", "warn", str(exc))

  # GPU.
  if shutil.which("nvidia-smi"):
    add("GPU", "ok", "nvidia-smi found")
  else:
    add("GPU", "warn", "nvidia-smi not found: only --dry-run will work")

  # tmux, because a multi-hour SSH session without it is a bad idea.
  if os.environ.get("TMUX"):
    add("tmux session", "ok", "running inside tmux")
  else:
    add("tmux session", "warn", "not in tmux: an SSH drop would kill the run")

  return checks


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``run``."""
  if args.interactive:
    return cmd_wizard(args, console)

  if args.config:
    if not os.path.exists(args.config):
      console.error(f"Config file not found: {args.config}")
      return EXIT_NOT_FOUND
    config = CampaignConfig.from_yaml(args.config)
  else:
    from src.orchestrator.cli.wizard import PRESETS  # pylint: disable=g-import-not-at-top

    sft_runs, rm_runs, perl_runs, eval_samples = PRESETS.get(
        args.preset or "standard", PRESETS["standard"]
    )
    config = CampaignConfig.create_default(
        task_name=args.task,
        sft_runs=args.sft_runs if args.sft_runs is not None else sft_runs,
        rm_runs=args.rm_runs if args.rm_runs is not None else rm_runs,
        perl_runs=args.perl_runs if args.perl_runs is not None else perl_runs,
        dry_run=args.dry_run,
    )
    config.eval.max_eval_samples = (
        args.eval_samples if args.eval_samples is not None else eval_samples
    )

  if args.stages:
    config.stages = theme_mod.iter_stage_names(args.stages.split(","))
  if args.sft_model:
    config.perl.sft_model_path = args.sft_model
  if args.reward_model:
    config.perl.reward_model_path = args.reward_model
  if getattr(args, "rm_datasets", None):
    config.rm_dataset_flavors = flavors.normalize_flavors(
        args.rm_datasets.split(",")
    )
  if getattr(args, "entity", None):
    config.wandb_entity = args.entity
  if getattr(args, "user", None):
    config.user = args.user
  if args.dry_run:
    config.dry_run = True
  if args.no_tui or args.plain:
    config.no_tui = True
  # None means "the flag was not given"; only then does the YAML win.
  if getattr(args, "shutdown", None) is not None:
    config.shutdown_when_done = bool(args.shutdown)

  try:
    config.validate()
  except ValueError as exc:
    console.error(str(exc))
    return EXIT_USAGE

  missing = _missing_upstream_checkpoints(config)
  if missing:
    console.error(
        "PE-RL needs upstream checkpoints: "
        + ", ".join(missing)
        + "."
    )
    console.hint(
        "Add the stage(s) back with --stages, or pass --sft-model / "
        "--reward-model with an existing repo id."
    )
    return EXIT_USAGE

  # An existing state file means `run` would otherwise silently continue a
  # previous campaign. Settle that before printing a preview that would
  # describe a launch which may not happen.
  gate = resolve_existing_campaign(config, args, console)
  if gate is not None:
    return gate

  _print_launch_preview(config, console)
  if not args.yes and not config.dry_run and _is_interactive():
    answer = input("Launch this campaign? [Y/n]: ").strip().lower()
    if answer and not answer.startswith("y"):
      console.info("Aborted before launch.")
      return EXIT_OK

  return _execute_campaign(config, console, plain=bool(args.no_tui or args.plain))


def cmd_wizard(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``wizard``."""
  from src.orchestrator.cli.wizard import run_setup_wizard  # pylint: disable=g-import-not-at-top

  config = run_setup_wizard(console=console)
  if config is None:
    return EXIT_OK
  if getattr(args, "plain", False):
    config.no_tui = True
  return _execute_campaign(config, console, plain=bool(getattr(args, "plain", False)))


def _render_status_safely(
    state_file: Optional[str],
    console: UiConsole,
    task: str,
) -> bool:
  """Renders one status frame, tolerating a campaign in flux.

  A watcher is a read-only observer of a file another process is actively
  rewriting, and it typically runs unattended in a tmux pane for hours. It
  must therefore never die on a transient condition: the campaign may not
  have started yet, may have just been archived by ``run --fresh``, or the
  file may have been hand-edited into something unparseable.

  Args:
    state_file: Path to render, or None when no campaign was found.
    console: Output console.
    task: Task name, used in the "nothing yet" message.

  Returns:
    True when a campaign was rendered, False when the frame was a placeholder.
  """
  if not state_file or not os.path.exists(state_file):
    console.blank()
    console.info(f"No campaign state for task '{task}' yet.")
    console.hint(f"Waiting for one to appear. Start it with: {PROGRAM} run --task {task}")
    return False
  try:
    state = CampaignState.load(state_file)
    config = config_for_state(state)
  except (OSError, ValueError, KeyError, TypeError) as exc:
    console.blank()
    console.warn(f"Could not read {state_file}: {exc}")
    console.hint("Retrying on the next refresh.")
    return False
  _render_status(state, config, console, state_file)
  return True


def cmd_status(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``status``."""
  explicit_state = getattr(args, "state_file", None)
  state_file = explicit_state or find_latest_state(args.task)
  # A watcher is allowed to start before the campaign does; a one-shot status
  # is not - it has nothing to wait for.
  watching = bool(getattr(args, "watch", False)) and not args.json

  if (not state_file or not os.path.exists(state_file)) and not watching:
    if args.json:
      print(json.dumps({"error": "no_state_file", "task": args.task}))
      return EXIT_NOT_FOUND
    console.error(f"No campaign state found for task '{args.task}'.")
    console.hint(f"Start one with: {PROGRAM} run --task {args.task}")
    return EXIT_NOT_FOUND

  if args.json:
    state = CampaignState.load(state_file)
    config = config_for_state(state)
    done, total = state_progress(state, config)
    payload = {
        "campaign_id": state.campaign_id,
        "task": state.task_name,
        "status": state.status,
        "current_stage": state.current_stage,
        "stages_completed": done,
        "stages_total": total,
        "updated_at": state.updated_at,
        "state_file": state_file,
        "stages": {name: result.to_dict() for name, result in state.stages.items()},
    }
    print(json.dumps(payload, indent=2))
    return EXIT_OK

  if not watching:
    state = CampaignState.load(state_file)
    config = config_for_state(state)
    _render_status(state, config, console, state_file)
    return EXIT_OK

  refreshes = 0
  try:
    while True:
      # Re-resolved every refresh: the common workflow is to split the pane
      # and start watching *before* launching, and pinning the path once
      # would leave the watcher stuck on the previous campaign forever.
      current = explicit_state or find_latest_state(args.task)
      if console.is_rich and console.rich is not None:
        console.rich.clear()
      _render_status_safely(current, console, args.task)
      label = os.path.basename(current) if current else f"task {args.task}"
      console.hint(
          f"watching {label} - refresh every "
          f"{args.interval:g}s - Ctrl-C to exit"
      )
      refreshes += 1
      if args.max_refreshes and refreshes >= args.max_refreshes:
        break
      time.sleep(max(0.5, args.interval))
  except KeyboardInterrupt:
    console.blank()
    console.info("Stopped watching.")
  return EXIT_OK


def cmd_resume(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``resume``."""
  state_file = args.state_file or find_latest_state(args.task)
  if not state_file or not os.path.exists(state_file):
    console.error(f"No campaign state found to resume for task '{args.task}'.")
    console.hint(f"Start a new one with: {PROGRAM} run --task {args.task}")
    return EXIT_NOT_FOUND

  state = CampaignState.load(state_file)
  config = config_for_state(state)
  config.state_file = state_file
  if getattr(args, "entity", None):
    config.wandb_entity = args.entity
  if getattr(args, "user", None):
    config.user = args.user
  if args.dry_run:
    config.dry_run = True
  if args.no_tui or args.plain:
    config.no_tui = True
  # The rebuilt config carries whatever the original launch asked for, so an
  # explicit flag here is the only way to change it for this attempt.
  if getattr(args, "shutdown", None) is not None:
    config.shutdown_when_done = bool(args.shutdown)

  views = renderables.build_stage_views(config, state)
  completed = [v.title for v in views if str(v.status).upper() == "COMPLETED"]
  remaining = [v.title for v in views if str(v.status).upper() != "COMPLETED"]

  console.blank()
  console.panel(
      [
          f"{console.theme.markup('Campaign', 'muted')}  {state.campaign_id}",
          f"{console.theme.markup('State   ', 'muted')}  {state_file}",
          f"{console.theme.markup('Updated ', 'muted')}  {state.updated_at}",
          f"{console.theme.markup('Skipping', 'muted')}  "
          + (", ".join(completed) if completed else "nothing - starting from the top"),
          f"{console.theme.markup('Will run', 'muted')}  "
          + (", ".join(remaining) if remaining else "nothing - already complete"),
      ],
      title="Resume plan",
      style="accent",
  )
  if not remaining:
    console.success("This campaign is already complete.")
    console.hint(f"Regenerate its report with: {PROGRAM} report --task {args.task}")
    return EXIT_OK

  if not args.yes and _is_interactive():
    answer = input("Resume this campaign? [Y/n]: ").strip().lower()
    if answer and not answer.startswith("y"):
      console.info("Resume cancelled.")
      return EXIT_OK

  return _execute_campaign(
      config, console, state=state, plain=bool(args.no_tui or args.plain)
  )


def cmd_report(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``report``."""
  state_file = args.state_file or find_latest_state(args.task)
  if not state_file or not os.path.exists(state_file):
    console.error(f"No campaign state found for task '{args.task}'.")
    return EXIT_NOT_FOUND
  state = CampaignState.load(state_file)
  config = config_for_state(state)
  from src.orchestrator.reporter import CampaignReporter  # pylint: disable=g-import-not-at-top

  artifacts = CampaignReporter(config, state).generate_all()
  console.blank()
  console.success("Reports generated:")
  for key, value in artifacts.items():
    console.print(
        f"  {console.theme.markup(key, 'muted')}  "
        f"{console.theme.markup(str(value), 'accent')}"
    )
  return EXIT_OK


def cmd_list(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``list``/``ls``."""
  state_files = find_state_files(args.task)
  entries: List[Dict[str, Any]] = []
  for path in state_files:
    try:
      state = CampaignState.load(path)
    except (OSError, ValueError, KeyError):
      continue
    config = config_for_state(state)
    done, total = state_progress(state, config)
    entries.append({
        "campaign_id": state.campaign_id,
        "task": state.task_name,
        "status": state.status,
        "progress": f"{done}/{total}",
        "updated_at": state.updated_at,
        "state_file": path,
    })

  if args.json:
    print(json.dumps(entries, indent=2))
    return EXIT_OK

  if not entries:
    console.info("No campaigns found yet.")
    console.hint(f"Create one with: {PROGRAM} wizard")
    return EXIT_OK

  theme = console.theme
  if console.is_rich:
    from rich.table import Table  # pylint: disable=g-import-not-at-top
    from rich import box  # pylint: disable=g-import-not-at-top

    table = Table(
        box=box.ROUNDED if theme.use_unicode else box.ASCII,
        header_style=theme.style("heading") or None,
        border_style=theme.style("border") or None,
        title=theme.markup("Campaigns", "heading"),
    )
    for column in ("Campaign", "Task", "Status", "Stages", "Updated"):
      table.add_column(column, no_wrap=(column != "Campaign"))
    for entry in entries:
      status_style = {
          "COMPLETED": "success",
          "FAILED": "error",
          "IN_PROGRESS": "running",
      }.get(entry["status"], "muted")
      table.add_row(
          entry["campaign_id"],
          entry["task"],
          theme.markup(entry["status"], status_style),
          entry["progress"],
          str(entry["updated_at"])[:19].replace("T", " "),
      )
    console.print_renderable(table)
  else:
    for entry in entries:
      console.print(
          f"{entry['campaign_id']:<34} {entry['task']:<10} "
          f"{entry['status']:<12} {entry['progress']:<6} {entry['updated_at'][:19]}"
      )
  console.hint(f"Inspect one with: {PROGRAM} status --task <task>")
  return EXIT_OK


def cmd_doctor(args: argparse.Namespace, console: UiConsole) -> int:
  """Implements ``doctor``."""
  config = CampaignConfig.create_default(task_name=args.task)
  if getattr(args, "entity", None):
    config.wandb_entity = args.entity
  if getattr(args, "user", None):
    config.user = args.user
  checks = run_diagnostics(config)
  if args.json:
    print(json.dumps(checks, indent=2))
    return EXIT_OK if not any(c["status"] == "fail" for c in checks) else EXIT_ERROR

  theme = console.theme
  console.blank()
  console.rule("Pre-flight check")
  marks = {
      "ok": theme.markup(theme.glyphs.completed, "success"),
      "warn": theme.markup("!", "warning"),
      "fail": theme.markup(theme.glyphs.failed, "error"),
  }
  width = max(len(check["name"]) for check in checks)
  for check in checks:
    console.print(
        f"  {marks.get(check['status'], '?')} "
        f"{theme.markup(check['name'].ljust(width), 'value')}  "
        f"{theme.markup(check['detail'], 'muted')}"
    )
  failures = [c for c in checks if c["status"] == "fail"]
  warnings = [c for c in checks if c["status"] == "warn"]
  console.blank()
  if failures:
    console.error(
        f"{len(failures)} blocking issue(s) found - fix them before launching."
    )
    return EXIT_ERROR
  if warnings:
    console.warn(f"{len(warnings)} warning(s); a dry-run is still recommended.")
  else:
    console.success("Environment looks healthy. Boa viagem!")
  console.hint(f"Next: {PROGRAM} run --task {args.task} --preset smoke --dry-run")
  return EXIT_OK


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _is_interactive() -> bool:
  """True when we can safely ask the user a blocking question."""
  try:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())
  except Exception:  # pylint: disable=broad-except
    return False


def _survive_terminal_hangup(console: UiConsole) -> None:
  """Makes the campaign immune to the terminal going away.

  A dropped SSH connection sends SIGHUP, whose default action is to kill the
  process - taking a campaign that is eight hours into its GPU budget with
  it. tmux normally shields us, but nothing forces the user to remember tmux,
  so ignore the signal outright. Ctrl-C (SIGINT) and ``s`` still work.

  Args:
    console: Used to hint at the behaviour when it is worth mentioning.
  """
  if not hasattr(signal, "SIGHUP"):
    return  # Not POSIX; nothing to guard against.
  try:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
  except (OSError, ValueError) as error:
    # Only the main thread may install handlers; not fatal.
    logger.debug("Could not ignore SIGHUP: %s", error)
    return
  if not _is_interactive():
    console.info(
        "SIGHUP ignored: the campaign will keep running if the terminal "
        "disconnects."
    )


def _missing_upstream_checkpoints(config: CampaignConfig) -> List[str]:
  """Returns the upstream models a selected stage needs but cannot obtain.

  Catching this before launch matters: the alternative is discovering at
  hour eleven that the final stage has nothing to work with.

  Args:
    config: The campaign about to be launched.

  Returns:
    Flags the user must supply, empty when the plan is self-sufficient.
  """
  stages = theme_mod.iter_stage_names(config.stages)
  missing = []

  if "perl" in stages:
    if "sft" not in stages and (
        not config.perl.sft_model_path or config.perl.sft_model_path == "auto"
    ):
      missing.append("--sft-model")
    if "rm" not in stages and (
        not config.perl.reward_model_path
        or config.perl.reward_model_path == "auto"
    ):
      missing.append("--reward-model")

  # Evaluation scores the trained policies (SFT and PE-RL). It deliberately
  # never falls back to the base model, so a run with neither stage selected
  # and no explicit SFT checkpoint has nothing to evaluate.
  if "eval" in stages and not {"sft", "perl"} & set(stages):
    if not config.perl.sft_model_path or config.perl.sft_model_path == "auto":
      missing.append("--sft-model (evaluation has no policy to score)")

  return missing


def _print_launch_preview(config: CampaignConfig, console: UiConsole) -> None:
  """Shows what is about to happen before the campaign starts."""
  from src.orchestrator.cli.wizard import config_summary_lines  # pylint: disable=g-import-not-at-top

  console.blank()
  console.panel(
      config_summary_lines(config, console.theme),
      title="Launching campaign",
      style="accent",
  )


def _render_status(
    state: CampaignState,
    config: CampaignConfig,
    console: UiConsole,
    state_file: str,
) -> None:
  """Renders the status view for one campaign."""
  theme = console.theme
  done, total = state_progress(state, config)
  status_style = {
      "COMPLETED": "success",
      "FAILED": "error",
      "IN_PROGRESS": "running",
      "PAUSED": "warning",
      "STOPPED": "warning",
  }.get(str(state.status).upper(), "muted")

  console.blank()
  console.print(
      f"{theme.markup(state.campaign_id, 'heading')}  "
      f"{theme.markup(state.status, status_style)}  "
      f"{theme.markup(f'{done}/{total} stages', 'muted')}"
  )
  console.print(
      theme.markup(
          f"task {state.task_name} {theme.glyphs.bullet} updated "
          f"{str(state.updated_at)[:19].replace('T', ' ')} "
          f"{theme.glyphs.bullet} {state_file}",
          "muted",
      )
  )
  console.blank()
  if console.is_rich:
    console.print_renderable(
        renderables.status_table(config, state, theme, width=console.width)
    )
  else:
    console.print_lines(
        renderables.dag_lines(
            renderables.build_stage_views(config, state), theme, width=console.width
        )
    )
  eval_result = state.stages.get("eval")
  if eval_result is not None and eval_result.metrics:
    console.blank()
    console.print(theme.markup("Final metrics", "heading"))
    for name, value in eval_result.metrics.items():
      console.print(
          f"  {theme.markup(str(name).ljust(22), 'muted')} "
          f"{theme.markup(theme_mod.format_metric(value), 'metric')}"
      )


def _execute_campaign(
    config: CampaignConfig,
    console: UiConsole,
    state: Optional[CampaignState] = None,
    plain: bool = False,
) -> int:
  """Runs the campaign with the dashboard or the plain streamer.

  Args:
    config: The campaign to run.
    console: Output surface; replaced by a plain one when the TUI is off.
    state: Existing state to resume from, or None for a fresh campaign.
    plain: Force plain streaming regardless of the config.

  Returns:
    The process exit code.
  """
  from src.orchestrator.cli.dashboard import run_campaign_with_dashboard  # pylint: disable=g-import-not-at-top

  use_plain = plain or config.no_tui
  if use_plain:
    console = UiConsole(theme=console.theme, force_plain=not console.is_rich)

  # A failing campaign does not return a status - the engine re-raises out of
  # the dashboard - so the outcome is tracked here and the shutdown hook sits
  # in a `finally`. Otherwise the one case where an idle VM is most expensive
  # (a crash at hour two of twelve) would be the one case that never powers
  # off. Assume FAILED until proven otherwise for the same reason.
  status = "FAILED"
  try:
    result = run_campaign_with_dashboard(
        config=config,
        state=state,
        console=console,
        enable_keys=not use_plain and _is_interactive(),
    )
    _print_outcome(result, config, console)
    status = str(result.get("status", "")).upper()
    return EXIT_OK if status in ("COMPLETED", "STOPPED", "") else EXIT_ERROR
  except KeyboardInterrupt:
    # Somebody is at the keyboard; taking their machine down is never what
    # Ctrl-C meant.
    status = "STOPPED"
    raise
  finally:
    shutdown.maybe_shutdown(config, status, console)



def _print_outcome(
    result: Dict[str, Any], config: CampaignConfig, console: UiConsole
) -> None:
  """Prints the closing summary and the natural next steps."""
  theme = console.theme
  status = str(result.get("status", "DETACHED")).upper()
  console.blank()
  if status == "COMPLETED":
    console.success("Campaign completed.")
  elif status == "STOPPED":
    console.warn("Campaign stopped early; progress was saved.")
  else:
    console.info(f"Campaign finished with status {status}.")

  artifacts = result.get("artifacts") or {}
  if artifacts:
    console.blank()
    console.print(theme.markup("Artifacts", "heading"))
    for key, value in artifacts.items():
      console.print(
          f"  {theme.markup(key.ljust(18), 'muted')} "
          f"{theme.markup(str(value), 'accent')}"
      )
  console.blank()
  console.print(theme.markup("Next steps", "heading"))
  console.hint(f"{PROGRAM} status --task {config.task_name}")
  if status != "COMPLETED":
    console.hint(f"{PROGRAM} resume --task {config.task_name}")
  console.hint(f"{PROGRAM} report --task {config.task_name}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None, console: Optional[UiConsole] = None) -> int:
  """CLI entry point. Returns the process exit code."""
  parser = build_parser()
  args = _apply_global_defaults(
      parser.parse_args(list(argv) if argv is not None else None)
  )

  if console is None:
    theme = theme_mod.detect_theme(
        force_color=False if args.no_color else None,
        force_ascii=True if args.ascii else None,
    )
    console = UiConsole(theme=theme, force_plain=bool(args.plain and args.no_color))

  if not args.command:
    console.blank()
    console.print(renderables.banner(console.theme, "Automated PE-RL campaigns"))
    console.blank()
    parser.print_help()
    console.blank()
    console.hint(f"Tip: start with '{PROGRAM} doctor' then '{PROGRAM} wizard'.")
    return EXIT_OK

  handlers = {
      "run": cmd_run,
      "wizard": cmd_wizard,
      "status": cmd_status,
      "resume": cmd_resume,
      "report": cmd_report,
      "list": cmd_list,
      "ls": cmd_list,
      "doctor": cmd_doctor,
  }
  handler = handlers.get(args.command)
  if handler is None:
    parser.print_help()
    return EXIT_USAGE

  if args.command in ("run", "resume"):
    _survive_terminal_hangup(console)

  try:
    return handler(args, console)
  except KeyboardInterrupt:
    console.blank()
    console.warn("Interrupted by user.")
    return EXIT_ERROR
  except Exception as exc:  # pylint: disable=broad-except
    if getattr(args, "debug", False):
      raise
    console.blank()
    console.error(f"{type(exc).__name__}: {exc}")
    console.hint("Re-run with --debug to see the full traceback.")
    return EXIT_ERROR
