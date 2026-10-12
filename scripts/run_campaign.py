#!/usr/bin/env python3
"""Auto-PERL Campaign Runner CLI.

Thin entry point: all the logic lives in ``src.orchestrator.cli.app`` so it
can be imported and unit tested.

Usage:
  # Guided setup
  python3 scripts/run_campaign.py wizard

  # Pre-flight environment check
  python3 scripts/run_campaign.py doctor --task npov

  # Quick run with a budget preset
  python3 scripts/run_campaign.py run --task npov --preset quick

  # Watch progress from a second tmux pane
  python3 scripts/run_campaign.py status --task npov --watch

  # Resume an interrupted campaign
  python3 scripts/run_campaign.py resume --task npov
"""

from __future__ import annotations

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator.cli.app import main  # pylint: disable=g-import-not-at-top


if __name__ == "__main__":
  sys.exit(main())
