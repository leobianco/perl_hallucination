"""Entry point for the interactive Mission Control terminal UI.

Historically this module hosted a full-screen ``textual`` application. It now
delegates to :mod:`src.orchestrator.cli.dashboard`, which renders a *pinned*
live block instead of taking over the alternate screen buffer.

Why the change: on a GCP VM the orchestrator is driven through tmux over SSH.
A full-screen TUI hides the log history from tmux scrollback and copy-mode,
repaints poorly on re-attach, and cannot be piped to a file. The pinned
dashboard keeps every log line in the scrollback while still showing a live
DAG, leaderboard and hotkey bar, and it collapses gracefully to plain text
when the output is redirected.

The public function :func:`launch_interactive_tui` keeps its original
signature for backwards compatibility.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

from src.orchestrator.cli.console import UiConsole
from src.orchestrator.cli.dashboard import run_campaign_with_dashboard
from src.orchestrator.config import CampaignConfig

logger = logging.getLogger(__name__)


def launch_interactive_tui(
    config: CampaignConfig,
    state: Any = None,
    console: Optional[UiConsole] = None,
    enable_keys: Optional[bool] = None,
) -> Dict[str, Any]:
  """Runs a campaign with the live dashboard attached.

  Args:
    config: Campaign configuration to execute.
    state: Optional pre-loaded ``CampaignState`` for resumption.
    console: Output surface; auto-detected when omitted.
    enable_keys: Force hotkey capture on/off. ``None`` enables it only when
      both stdin and stdout are attached to a TTY.

  Returns:
    The engine result dictionary.
  """
  console = console or UiConsole()
  if enable_keys is None:
    enable_keys = bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )
  return run_campaign_with_dashboard(
      config=config,
      state=state,
      console=console,
      enable_keys=enable_keys,
  )
