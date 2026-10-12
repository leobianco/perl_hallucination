"""Retry helpers for the flaky, network-bound parts of a campaign.

A campaign spends the whole day talking to the W&B API and the Hugging Face
Hub. Both return transient 5xx/timeout errors often enough that, over a
12 hour run, hitting one is close to certain. Without a retry the campaign
dies at 3am and the GPU idles until morning, so every network boundary in the
orchestrator goes through :func:`run_with_retries`.

Retries are deliberately *not* applied to:
  * user interruptions (stop/abort), which must take effect immediately;
  * argument or configuration errors, which are deterministic and would just
    burn the same failure N times.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Substrings identifying errors that will never succeed on a second attempt.
NON_RETRYABLE_MARKERS: Tuple[str, ...] = (
    "unrecognized arguments",
    "interrupted",
    "permission denied",
    "401",
    "403",
    "not a valid",
    # Sweep produced no usable metric: retrying the query cannot change that.
    "logged the metric",
    "no runs found in sweep",
    "has no trained policy",
)


class RetryError(RuntimeError):
  """Raised when every attempt of a retried operation failed."""


def is_retryable(error: BaseException) -> bool:
  """Heuristically decides whether ``error`` is worth another attempt.

  Args:
    error: The exception raised by the operation.

  Returns:
    False for interruptions and errors that are clearly deterministic.
  """
  if isinstance(error, (KeyboardInterrupt, SystemExit)):
    return False
  text = str(error).lower()
  return not any(marker in text for marker in NON_RETRYABLE_MARKERS)


def run_with_retries(
    operation: Callable[[], T],
    description: str,
    attempts: int = 4,
    base_delay_s: float = 5.0,
    max_delay_s: float = 300.0,
    on_notice: Optional[Callable[[str], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    retry_on: Optional[Iterable[Type[BaseException]]] = None,
) -> T:
  """Runs ``operation`` with exponential backoff.

  Args:
    operation: Zero-argument callable to execute.
    description: Human readable name used in log and notice messages.
    attempts: Total number of attempts (1 disables retrying).
    base_delay_s: Delay before the second attempt; doubles each time.
    max_delay_s: Upper bound on the backoff delay.
    on_notice: Optional sink receiving a human readable line per retry.
    stop_requested: Predicate; when it returns True the retry loop gives up
      immediately so a user stop is not delayed by a long backoff.
    retry_on: Optional whitelist of exception types to retry. When omitted,
      :func:`is_retryable` decides.

  Returns:
    Whatever ``operation`` returns.

  Raises:
    BaseException: The last error raised by ``operation`` once the attempts
      are exhausted or the error is judged non-retryable.
  """
  total = max(1, int(attempts))
  delay = max(0.0, base_delay_s)
  last_error: Optional[BaseException] = None

  for attempt in range(1, total + 1):
    try:
      return operation()
    except BaseException as error:  # pylint: disable=broad-except
      last_error = error
      retryable = (
          isinstance(error, tuple(retry_on))
          if retry_on is not None
          else is_retryable(error)
      )
      if not retryable or attempt == total:
        raise
      if stop_requested is not None and stop_requested():
        logger.info("Stop requested; not retrying %s.", description)
        raise

      message = (
          f"{description} failed (attempt {attempt}/{total}): {error}. "
          f"Retrying in {delay:.0f}s."
      )
      logger.warning(message)
      if on_notice:
        on_notice(f"[RETRY] {message}")

      # Sleep in slices so a stop request is honored during a long backoff.
      waited = 0.0
      while waited < delay:
        if stop_requested is not None and stop_requested():
          logger.info("Stop requested during backoff of %s.", description)
          raise
        time.sleep(min(1.0, delay - waited))
        waited += 1.0
      delay = min(max_delay_s, delay * 2.0)

  # Unreachable: the loop either returns or raises.
  raise RetryError(f"{description} failed") from last_error
