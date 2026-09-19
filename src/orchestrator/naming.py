"""Naming of the W&B sweeps a campaign registers.

Sweeps used to be numbered ``Sweep #1``, ``Sweep #2``, ... The number was
recovered by listing every state file on disk *and* every sweep in the W&B
project and taking ``max + 1``. That is expensive (a W&B round trip per sweep
registration), fragile (a campaign run from a different directory, or one whose
sweeps were deleted from the UI, restarts the count) and it collides silently:
two campaigns started on two machines both call their sweep ``#1``, and the
project ends up with several ``RAGTRUTH-QA gemma-4-E4B-it SFT Sweep #1`` that
nothing distinguishes.

The number is replaced by a short hex token derived from *when the campaign
started*:

    RAGTRUTH-QA gemma-4-E4B-it SFT Sweep 35d2a7
    RAGTRUTH-QA gemma-3-1b-it RM Organic Sweep 35d2a7
    RAGTRUTH-QA gemma-4-E4B-it PERL Organic Sweep 35d2a7

Deliberately campaign-scoped rather than per-sweep: all the sweeps of one
campaign carry the same token, which is what makes them groupable in the W&B
UI. The token also sorts chronologically, so an alphabetical sweep list is a
timeline.
"""

from __future__ import annotations

import datetime
import re
from typing import Optional

#: Epoch the token counts minutes from. Fixed rather than "the Unix epoch" so
#: the token stays 6 characters wide: 0xFFFFFF minutes from here is the year
#: 2051, by which point this code will be somebody else's problem.
TOKEN_EPOCH = datetime.datetime(2020, 1, 1)

#: Width of the zero-padded hex token.
TOKEN_WIDTH = 6

#: Matches the ``%y%m%d%H%M`` stamp ``CampaignConfig`` appends to a campaign
#: name (``ragtruth_campaign_2609150825``).
_CAMPAIGN_STAMP_RE = re.compile(r"(?:^|[_-])(\d{10})$")

#: Recognises an already-tokenised sweep name, for tests and for readers that
#: want to group sweeps by campaign.
SWEEP_TOKEN_RE = re.compile(r"\bSweep ([0-9a-f]{%d})\b" % TOKEN_WIDTH)


def token_from_datetime(moment: datetime.datetime) -> str:
  """Returns the hex token identifying the minute ``moment`` falls in.

  Args:
    moment: Naive local datetime, normally a campaign's start time.

  Returns:
    A lowercase, zero-padded hex string of :data:`TOKEN_WIDTH` characters.
  """
  minutes = int((moment - TOKEN_EPOCH).total_seconds() // 60)
  minutes = max(0, minutes) & 0xFFFFFF
  return format(minutes, f"0{TOKEN_WIDTH}x")


def _parse_campaign_stamp(campaign_id: Optional[str]) -> Optional[datetime.datetime]:
  """Recovers the start time encoded in a campaign id, when there is one."""
  if not campaign_id:
    return None
  match = _CAMPAIGN_STAMP_RE.search(str(campaign_id))
  if not match:
    return None
  try:
    return datetime.datetime.strptime(match.group(1), "%y%m%d%H%M")
  except ValueError:
    return None


def _parse_iso(value: Optional[str]) -> Optional[datetime.datetime]:
  """Parses an ISO timestamp, returning None instead of raising."""
  if not value:
    return None
  try:
    return datetime.datetime.fromisoformat(str(value))
  except (TypeError, ValueError):
    return None


def campaign_token(
    campaign_id: Optional[str] = None,
    created_at: Optional[str] = None,
    now: Optional[datetime.datetime] = None,
) -> str:
  """Returns the token shared by every sweep of one campaign.

  The campaign id is the preferred source: it is stable across resumes, so a
  campaign that crashes and is restarted keeps naming its sweeps the same way
  instead of scattering them under two tokens. ``created_at`` (the state
  file's creation timestamp) is the fallback for campaigns named by hand, and
  the clock is the last resort.

  Args:
    campaign_id: Campaign name, normally ``{task}_campaign_{%y%m%d%H%M}``.
    created_at: ISO timestamp of the campaign state, used when the id carries
      no stamp.
    now: Clock override, for tests.

  Returns:
    A lowercase hex token of :data:`TOKEN_WIDTH` characters.
  """
  moment = (
      _parse_campaign_stamp(campaign_id)
      or _parse_iso(created_at)
      or now
      or datetime.datetime.now()
  )
  return token_from_datetime(moment)
