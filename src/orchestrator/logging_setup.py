"""Campaign-wide logging: a durable transcript of an unattended run.

Before this module the orchestrator configured no logging handler at all, so
every ``logger.info`` call was dropped and ``logger.warning`` went to stderr
through ``logging.lastResort`` - straight through the Rich live region. The
only record of a twelve hour campaign was the tmux scrollback, which is
capped and disappears with the session.

:func:`configure_campaign_logging` sends everything to
``logs/campaign/{campaign_id}.log`` instead, and :func:`tee_line_callback`
mirrors the raw subprocess output (training curves, ``wandb agent`` chatter)
into the same file so failures can be diagnosed the morning after.
"""

from __future__ import annotations

import datetime
import logging
import logging.handlers
import os
import threading
from typing import Callable, Optional

LOG_DIR = os.path.join("logs", "campaign")

#: Overrides the log directory. Set by the test suite so unit tests do not
#: litter the repository with campaign logs.
LOG_DIR_ENV = "PERL_CAMPAIGN_LOG_DIR"

#: Keep a full run plus history; training logs are verbose.
MAX_BYTES = 64 * 1024 * 1024
BACKUP_COUNT = 3

_handler_lock = threading.Lock()


def resolve_log_dir(log_dir: Optional[str] = None) -> str:
  """Resolves the effective log directory (env override wins)."""
  return os.environ.get(LOG_DIR_ENV) or log_dir or LOG_DIR


def campaign_log_path(
    campaign_id: str, log_dir: Optional[str] = None
) -> str:
  """Returns the log file path for ``campaign_id``."""
  return os.path.join(resolve_log_dir(log_dir), f"{campaign_id}.log")


def configure_campaign_logging(
    campaign_id: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    quiet_console: bool = True,
) -> str:
  """Attaches a rotating file handler for the whole process.

  Args:
    campaign_id: Campaign name; used as the log file stem.
    log_dir: Directory holding campaign logs.
    level: Log level for the orchestrator's own loggers.
    quiet_console: When True, stray console handlers are removed so library
      warnings cannot corrupt the live dashboard. The file still receives
      everything.

  Returns:
    The absolute path of the log file.
  """
  path = os.path.abspath(campaign_log_path(campaign_id, log_dir))
  os.makedirs(os.path.dirname(path), exist_ok=True)

  root = logging.getLogger()
  root.setLevel(min(level, logging.INFO))

  with _handler_lock:
    for existing in list(root.handlers):
      if getattr(existing, "_perl_campaign_log", None) == path:
        return path
      if quiet_console and isinstance(existing, logging.StreamHandler):
        root.removeHandler(existing)

    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    # Tagged so a second call (resume in the same process) is a no-op.
    handler._perl_campaign_log = path  # pylint: disable=protected-access
    root.addHandler(handler)

  # ``logging.lastResort`` prints WARNING+ to stderr when no handler exists,
  # which would smear the live region. A real handler now exists, but silence
  # it explicitly in case another library resets the root logger.
  logging.lastResort = logging.NullHandler()

  logging.getLogger(__name__).info(
      "=== Campaign log opened for %s at %s ===",
      campaign_id,
      datetime.datetime.now().isoformat(timespec="seconds"),
  )
  return path


def tee_line_callback(
    path: str,
    downstream: Optional[Callable[[str], None]] = None,
) -> Callable[[str], None]:
  """Wraps a live-line callback so every line is also appended to ``path``.

  Subprocess output is the only evidence of what a trial actually did; the
  dashboard keeps a bounded ring buffer in memory, so without this tee the
  detail is gone as soon as it scrolls.

  Args:
    path: Log file to append to.
    downstream: Original callback (dashboard/plain printer), may be None.

  Returns:
    A callback writing to both sinks. Write failures are swallowed: a full
    disk must not take the campaign down with it.
  """
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
  lock = threading.Lock()

  def _tee(line: str) -> None:
    if downstream is not None:
      try:
        downstream(line)
      except Exception:  # pylint: disable=broad-except
        pass
    try:
      with lock:
        with open(path, "a", encoding="utf-8") as handle:
          handle.write(f"{datetime.datetime.now():%H:%M:%S} {line}\n")
    except Exception:  # pylint: disable=broad-except
      pass

  return _tee
