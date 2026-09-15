"""Pytest fixtures shared by the orchestrator test suite."""

import os
import tempfile

import pytest

from src.orchestrator import logging_setup


@pytest.fixture(autouse=True, scope="session")
def _isolate_campaign_logs():
  """Keeps campaign logs out of the repository during tests.

  The engine opens ``logs/campaign/{campaign_id}.log`` on construction, and
  the suite builds dozens of engines; without this every run would leave
  megabytes of stray logs in the working copy.
  """
  with tempfile.TemporaryDirectory(prefix="perl_test_logs_") as temp_dir:
    previous = os.environ.get(logging_setup.LOG_DIR_ENV)
    os.environ[logging_setup.LOG_DIR_ENV] = temp_dir
    try:
      yield temp_dir
    finally:
      if previous is None:
        os.environ.pop(logging_setup.LOG_DIR_ENV, None)
      else:
        os.environ[logging_setup.LOG_DIR_ENV] = previous
