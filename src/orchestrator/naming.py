"""Names a campaign gives to the artifacts it creates.

Two families live here: the W&B sweeps a campaign registers, and the Hugging
Face repositories it publishes checkpoints to. Both are generated names that
have to stay unique and stay readable, and both have run into the same class
of bug - a name that collides with another campaign's, or one that a length
limit quietly truncated into another name's twin.

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
from typing import Optional, Tuple

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


# --- Hugging Face repository names --------------------------------------

#: Hugging Face caps a repo id at 96 characters, namespace included.
HF_REPO_ID_MAX = 96

#: How short the task name may get before the fitter gives up on it. At 10
#: characters every task this repository defines is still distinguishable
#: (``ragtruth``, ``ragtruth-q``, ``ragtruth-s``), which is the property that
#: matters; ``TestRepoIdFitting`` pins it against ``VALID_TASKS``.
TASK_MIN_LEN = 10

#: How short the base-model name may get. Shrunk only after the task has hit
#: its own floor, because the model is what distinguishes checkpoints being
#: compared against each other, while the task is usually already known from
#: the context the name is read in.
MODEL_MIN_LEN = 12


def _shrink_token(token: str, target_len: int, keep_head: bool = True) -> str:
  """Shortens a hyphenated token, spending the last segments first.

  A plain ``token[:target_len]`` turns ``ragtruth-summarization`` into
  ``ragtruth-s`` only by luck, and turns ``ragtruth-qa`` into ``ragtruth-q``
  or into ``ragtruth`` depending on one character - at which point it is no
  longer distinguishable from the ``ragtruth`` task. Segments are therefore
  consumed from the right, and no segment is ever consumed entirely: the
  leading segment is the family name and the trailing ones are the variant,
  so keeping one character of each variant keeps the whole token unique.

  Args:
    token: The name to shorten, e.g. ``ragtruth-summarization``.
    target_len: Length to fit into. A token already at or below it is
      returned unchanged.
    keep_head: Leave the leading segment at full length. On by default
      because that segment is what a reader recognises the token by; only
      the last-resort path in :func:`fit_model_repo_id` turns it off, and
      then only because the alternative is an invalid name. A single-segment
      token is shrunk either way - there is no head to protect it from.

  Returns:
    The shortened token, which may still exceed ``target_len`` when every
    shrinkable segment is down to its last character.
  """
  if len(token) <= target_len:
    return token
  segments = token.split("-")
  floor = 1 if (len(segments) > 1 and keep_head) else 0
  # Walk right to left, taking as much as each segment can spare.
  for index in range(len(segments) - 1, floor - 1, -1):
    excess = len("-".join(segments)) - target_len
    if excess <= 0:
      break
    spare = len(segments[index]) - 1
    if spare <= 0:
      continue
    keep = len(segments[index]) - min(spare, excess)
    segments[index] = segments[index][:keep]
  return "-".join(segments)


def fit_model_repo_id(
    user: str,
    task: str,
    stage: str,
    model: str,
    details: str,
    timestamp: str,
    flavor_slug: str = "",
    max_length: int = HF_REPO_ID_MAX,
) -> Tuple[str, bool]:
  """Assembles a checkpoint repo id that fits the Hub's length limit.

  The naive approach - build the full name, then hand it to
  ``sanitize_hf_repo_id`` - does fit the limit, but it fits it by cutting
  characters off the *right*, and the right is where the timestamp lives.
  A ``ragtruth-summarization`` PE-RL name is 108 characters, so the trim
  removed the timestamp entirely and two campaigns a week apart produced the
  same repo id and silently stacked commits on each other.

  What actually has to survive, in order: the namespace, the stage, the
  flavor tag and the timestamp, all of which are short and none of which can
  be shortened without losing meaning. The elastic parts are spent in
  increasing order of how much a reader misses them: the task first (it is
  usually already known from the context the name is read in), then the base
  model, then - only if the first two have bottomed out - the hyperparameter
  block, which is recoverable from W&B.

  Args:
    user: Hub namespace.
    task: Campaign task, e.g. ``ragtruth-summarization``.
    stage: ``SFT``, ``RM`` or ``PERL``.
    model: Base model short name, e.g. ``gemma-4-E4B-it``.
    details: The hyperparameter block, e.g. ``S12345_epo3_lr1_0e-04_r16``.
    timestamp: The ``%y%m%d%H%M`` stamp that makes the name unique.
    flavor_slug: Reward-model dataset tag, empty for stages that have none.
    max_length: The Hub's cap, exposed for tests.

  Returns:
    The repo id, and whether anything had to be shortened to fit. The id is
    guaranteed to be at most ``max_length`` characters.

  Raises:
    ValueError: When the parts that may not be shortened do not themselves
      fit. Raising beats returning an over-long name that the caller's
      sanitizer would then truncate back into a collision.
  """
  flavor_tag = f"{flavor_slug}_" if flavor_slug else ""

  def _assemble(task_part: str, model_part: str, detail_part: str) -> str:
    joined = "_".join(p for p in (detail_part, timestamp) if p)
    return (
        f"{user}/{task_part}_{stage}_{flavor_tag}{model_part}_{joined}"
    )

  full = _assemble(task, model, details)
  if len(full) <= max_length:
    return full, False

  # 1. The task gives way first, down to the floor that keeps every known
  #    task distinguishable from every other.
  overflow = len(full) - max_length
  short_task = _shrink_token(task, max(TASK_MIN_LEN, len(task) - overflow))

  # 2. Then the base model.
  short_model = model
  overflow = len(_assemble(short_task, model, details)) - max_length
  if overflow > 0:
    short_model = _shrink_token(
        model, max(MODEL_MIN_LEN, len(model) - overflow)
    )

  # 3. Below here the name is already unusually long - no task or model in
  #    this repository reaches this point - so the floors are abandoned and
  #    both tokens are squeezed to one character per segment.
  overflow = len(_assemble(short_task, short_model, details)) - max_length
  if overflow > 0:
    short_task = _shrink_token(short_task, 1, keep_head=False)
    short_model = _shrink_token(short_model, 1, keep_head=False)

  # 4. Last resort: spend the hyperparameters, which W&B still has, rather
  #    than the timestamp, which nothing else records.
  short_details = details
  overflow = len(_assemble(short_task, short_model, details)) - max_length
  if overflow > 0:
    short_details = details[: max(0, len(details) - overflow)].rstrip("_-")

  fitted = _assemble(short_task, short_model, short_details)
  if len(fitted) > max_length:
    raise ValueError(
        f"Cannot build a repo id of at most {max_length} characters for "
        f"task '{task}', stage '{stage}', model '{model}': the namespace, "
        "stage, flavor tag and timestamp alone need "
        f"{len(fitted)}. Shorten the Hugging Face user name."
    )
  return fitted, True
