"""Publishing a built site to a host that outlives the VM.

The dashboard exists to be readable when the research VM is powered off, so
the build is only half the job: something has to push the snapshot somewhere
persistent. That is this module, and it is deliberately the *only* part of the
dashboard that talks to the network.

Backends:

``hf_space``
    A Hugging Face Space with ``sdk: static``. The default, because the Hub
    account, the token and ``huggingface_hub`` already exist in this
    repository, private Spaces are free, and the models and datasets the
    dashboard links to live in the same namespace.
``dir``
    A plain directory copy. Useful for testing, for an alternative host that
    serves from a filesystem, and for keeping a local archive of snapshots.

Two behaviours matter more than the backends:

* **A snapshot whose content is unchanged is never published.** A Space is a
  git repository; publishing on a timer rather than on a change would grow it
  without bound and say nothing new.
* **Publishing never raises into the watcher.** A failed upload is reported
  and retried on the next change; it must not be able to stop a campaign's
  snapshots from being taken.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
import os
import shutil
from typing import Any, Dict, Optional

#: Where the last published fingerprint is remembered, relative to the
#: repository root. Kept outside the site directory so that rebuilding (which
#: wipes the site) does not lose the publishing history.
DEFAULT_STATE_FILE = ".dashboard_publish.json"

#: The Space is recreated from scratch by every publish, so its whole content
#: is replaced rather than merged: a campaign page that no longer exists
#: locally must stop being served.
_DELETE_PATTERNS = ["*"]


@dataclass
class PublishResult:
  """Outcome of a publish attempt."""

  backend: str
  target: str
  published: bool
  reason: str = ""
  url: str = ""
  error: Optional[str] = None

  @property
  def ok(self) -> bool:
    """Whether the attempt completed without an error.

    A skipped publish is a success: it means there was nothing to say.
    """
    return self.error is None


def _load_state(path: str) -> Dict[str, Any]:
  """Reads the publish bookkeeping file.

  Args:
    path: Path to the state file.

  Returns:
    The parsed state, or an empty mapping when absent or unreadable.
  """
  try:
    with open(path, "r", encoding="utf-8") as handle:
      data = json.load(handle)
    return data if isinstance(data, dict) else {}
  except (OSError, ValueError):
    return {}


def _save_state(path: str, state: Dict[str, Any]) -> None:
  """Writes the publish bookkeeping file atomically enough for its purpose.

  Args:
    path: Path to the state file.
    state: Mapping to persist.
  """
  directory = os.path.dirname(os.path.abspath(path))
  os.makedirs(directory, exist_ok=True)
  temporary = f"{path}.tmp"
  with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2)
  os.replace(temporary, path)


def last_published_hash(state_file: str = DEFAULT_STATE_FILE) -> Optional[str]:
  """Returns the content fingerprint of the last successful publish.

  Args:
    state_file: Path to the publish state file.

  Returns:
    The stored hash, or None when nothing has been published yet.
  """
  return _load_state(state_file).get("content_hash")


def record_published(
    content_hash: str,
    target: str,
    backend: str,
    state_file: str = DEFAULT_STATE_FILE,
) -> None:
  """Remembers a successful publish.

  Args:
    content_hash: Fingerprint that was published.
    target: Where it went.
    backend: Which backend sent it.
    state_file: Path to the publish state file.
  """
  _save_state(
      state_file,
      {
          "content_hash": content_hash,
          "target": target,
          "backend": backend,
          "published_at": datetime.datetime.now().astimezone().isoformat(),
      },
  )


def _space_url(repo_id: str) -> str:
  """Returns the browsable URL of a Space.

  Args:
    repo_id: ``user/space`` identifier.

  Returns:
    The Space page URL. Private Spaces are viewed here rather than on their
    ``*.hf.space`` subdomain, which is the public serving path.
  """
  return f"https://huggingface.co/spaces/{repo_id}"


def publish_to_directory(
    site_dir: str, destination: str
) -> PublishResult:
  """Copies a built site into a local directory.

  Args:
    site_dir: Directory produced by :func:`src.dashboard.build.build_site`.
    destination: Directory to replace with the site's contents.

  Returns:
    The publish outcome.
  """
  try:
    if os.path.isdir(destination):
      shutil.rmtree(destination)
    shutil.copytree(site_dir, destination)
  except OSError as error:
    return PublishResult(
        backend="dir",
        target=destination,
        published=False,
        error=str(error),
    )
  return PublishResult(
      backend="dir",
      target=destination,
      published=True,
      url=f"file://{os.path.abspath(destination)}/index.html",
  )


def publish_to_space(
    site_dir: str,
    repo_id: str,
    private: bool = True,
    commit_message: str = "Update Auto-PERL dashboard snapshot",
    api: Any = None,
    create: bool = True,
) -> PublishResult:
  """Uploads a built site to a static Hugging Face Space.

  The Space is created on first use with ``sdk: static`` and the requested
  visibility. Subsequent publishes replace its whole contents in a single
  commit.

  Args:
    site_dir: Directory produced by :func:`src.dashboard.build.build_site`.
    repo_id: ``user/space`` identifier, e.g. ``leobianco/auto-perl``.
    private: Whether to create the Space private. Ignored if it exists: this
      function never changes the visibility of an existing Space, because
      silently making a private page public is not a thing a build tool
      should be able to do.
    commit_message: Commit message for the upload.
    api: An ``HfApi``-like object, injected by tests.
    create: Whether to create the Space when it does not exist.

  Returns:
    The publish outcome. Missing credentials and network failures are
    reported, never raised.
  """
  if api is None:
    try:
      from huggingface_hub import HfApi  # pylint: disable=g-import-not-at-top
    except ImportError:
      return PublishResult(
          backend="hf_space",
          target=repo_id,
          published=False,
          error=(
              "huggingface_hub is not installed; "
              "run pip install -r requirements.txt"
          ),
      )
    api = HfApi()

  try:
    if create:
      api.create_repo(
          repo_id=repo_id,
          repo_type="space",
          space_sdk="static",
          private=private,
          exist_ok=True,
      )
    api.upload_folder(
        folder_path=site_dir,
        repo_id=repo_id,
        repo_type="space",
        commit_message=commit_message,
        delete_patterns=_DELETE_PATTERNS,
    )
  except Exception as error:  # pylint: disable=broad-except
    # Anything from an expired token to a 503. The campaign that produced
    # this snapshot is long gone; the next change will try again.
    return PublishResult(
        backend="hf_space",
        target=repo_id,
        published=False,
        error=f"{type(error).__name__}: {error}",
    )

  return PublishResult(
      backend="hf_space",
      target=repo_id,
      published=True,
      url=_space_url(repo_id),
  )


def publish(
    site_dir: str,
    content_hash: str,
    backend: str = "hf_space",
    repo_id: Optional[str] = None,
    destination: Optional[str] = None,
    private: bool = True,
    force: bool = False,
    state_file: str = DEFAULT_STATE_FILE,
    commit_message: Optional[str] = None,
    api: Any = None,
) -> PublishResult:
  """Publishes a built site, unless its content is already published.

  Args:
    site_dir: Directory produced by :func:`src.dashboard.build.build_site`.
    content_hash: The build's content fingerprint.
    backend: ``hf_space`` or ``dir``.
    repo_id: Target Space, required by the ``hf_space`` backend.
    destination: Target directory, required by the ``dir`` backend.
    private: Visibility used when creating a new Space.
    force: Publish even when the content is unchanged.
    state_file: Where the last published fingerprint is remembered.
    commit_message: Overrides the default commit message.
    api: An ``HfApi``-like object, injected by tests.

  Returns:
    The publish outcome. A skipped publish has ``published=False`` and no
    error.
  """
  target = repo_id if backend == "hf_space" else (destination or "")
  if not force and content_hash and content_hash == last_published_hash(
      state_file
  ):
    return PublishResult(
        backend=backend,
        target=target or "",
        published=False,
        reason="unchanged since the last publish",
        url=_space_url(repo_id) if backend == "hf_space" and repo_id else "",
    )

  if backend == "dir":
    if not destination:
      return PublishResult(
          backend=backend,
          target="",
          published=False,
          error="the dir backend needs --destination",
      )
    result = publish_to_directory(site_dir, destination)
  elif backend == "hf_space":
    if not repo_id:
      return PublishResult(
          backend=backend,
          target="",
          published=False,
          error="the hf_space backend needs --repo-id",
      )
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    result = publish_to_space(
        site_dir,
        repo_id,
        private=private,
        commit_message=commit_message
        or f"Auto-PERL dashboard snapshot {stamp}",
        api=api,
    )
  else:
    return PublishResult(
        backend=backend,
        target=target or "",
        published=False,
        error=f"unknown backend '{backend}'",
    )

  if result.published and content_hash:
    record_published(content_hash, result.target, backend, state_file)
  return result
