"""Tests for publishing and for the change watcher.

Nothing here touches the network: the Hugging Face API is replaced by a
recording double, so the ``hf_space`` backend's *arguments* are asserted
(which is where the bugs live - wrong ``repo_type``, a missing
``delete_patterns``, an accidental public Space) without a real upload.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from src.dashboard import build as build_mod
from src.dashboard import index as index_mod
from src.dashboard import publish as publish_mod
from src.dashboard import watcher as watcher_mod
from src.dashboard.tests import fixtures


class _FakeApi:
  """Records the calls a publish would make to the Hub."""

  def __init__(self, fail_on: str = ""):
    """Initializes the double.

    Args:
      fail_on: Method name that should raise, to test error handling.
    """
    self.created = []
    self.uploaded = []
    self.fail_on = fail_on

  def create_repo(self, **kwargs):
    """Records a repo creation."""
    if self.fail_on == "create_repo":
      raise RuntimeError("boom")
    self.created.append(kwargs)

  def upload_folder(self, **kwargs):
    """Records a folder upload."""
    if self.fail_on == "upload_folder":
      raise RuntimeError("503 Service Unavailable")
    self.uploaded.append(kwargs)


class SpacePublishTest(unittest.TestCase):
  """Arguments passed to the Hub, and failures coming back from it."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.site = os.path.join(self.root, "site")
    os.makedirs(self.site)
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_creates_a_private_static_space(self):
    api = _FakeApi()
    result = publish_mod.publish_to_space(
        self.site, "leobianco/auto-perl", api=api
    )
    self.assertTrue(result.published)
    self.assertEqual(api.created[0]["repo_type"], "space")
    self.assertEqual(api.created[0]["space_sdk"], "static")
    self.assertTrue(api.created[0]["private"])
    self.assertTrue(api.created[0]["exist_ok"])

  def test_public_is_opt_in(self):
    api = _FakeApi()
    publish_mod.publish_to_space(
        self.site, "leobianco/auto-perl", private=False, api=api
    )
    self.assertFalse(api.created[0]["private"])

  def test_upload_replaces_the_whole_space(self):
    # Without delete_patterns, a campaign page removed locally would keep
    # being served from the Space forever.
    api = _FakeApi()
    publish_mod.publish_to_space(self.site, "leobianco/auto-perl", api=api)
    self.assertEqual(api.uploaded[0]["delete_patterns"], ["*"])
    self.assertEqual(api.uploaded[0]["repo_type"], "space")
    self.assertEqual(api.uploaded[0]["folder_path"], self.site)

  def test_url_points_at_the_space_page(self):
    result = publish_mod.publish_to_space(
        self.site, "leobianco/auto-perl", api=_FakeApi()
    )
    self.assertEqual(
        result.url, "https://huggingface.co/spaces/leobianco/auto-perl"
    )

  def test_hub_failure_is_reported_not_raised(self):
    result = publish_mod.publish_to_space(
        self.site, "leobianco/auto-perl", api=_FakeApi(fail_on="upload_folder")
    )
    self.assertFalse(result.published)
    self.assertFalse(result.ok)
    self.assertIn("503", result.error)


class DirectoryPublishTest(unittest.TestCase):
  """The filesystem backend."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.site = os.path.join(self.root, "site")
    os.makedirs(self.site)
    with open(os.path.join(self.site, "index.html"), "w", encoding="utf-8") as f:
      f.write("<html></html>")

  def test_copies_the_site(self):
    destination = os.path.join(self.root, "published")
    result = publish_mod.publish_to_directory(self.site, destination)
    self.assertTrue(result.published)
    self.assertTrue(os.path.isfile(os.path.join(destination, "index.html")))

  def test_replaces_a_previous_snapshot(self):
    destination = os.path.join(self.root, "published")
    os.makedirs(destination)
    stale = os.path.join(destination, "stale.html")
    with open(stale, "w", encoding="utf-8") as handle:
      handle.write("old")
    publish_mod.publish_to_directory(self.site, destination)
    self.assertFalse(os.path.isfile(stale))


class SkipUnchangedTest(unittest.TestCase):
  """A Space is a git repository; identical snapshots are not committed."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.site = os.path.join(self.root, "site")
    os.makedirs(self.site)
    self.state_file = os.path.join(self.root, "publish.json")
    self.api = _FakeApi()

  def _publish(self, content_hash: str, force: bool = False):
    """Publishes with the recording double.

    Args:
      content_hash: Fingerprint to publish.
      force: Whether to bypass the unchanged check.

    Returns:
      The publish result.
    """
    return publish_mod.publish(
        site_dir=self.site,
        content_hash=content_hash,
        backend="hf_space",
        repo_id="leobianco/auto-perl",
        force=force,
        state_file=self.state_file,
        api=self.api,
    )

  def test_first_publish_uploads(self):
    result = self._publish("hash-a")
    self.assertTrue(result.published)
    self.assertEqual(len(self.api.uploaded), 1)

  def test_identical_content_is_skipped(self):
    self._publish("hash-a")
    result = self._publish("hash-a")
    self.assertFalse(result.published)
    self.assertTrue(result.ok)
    self.assertIn("unchanged", result.reason)
    self.assertEqual(len(self.api.uploaded), 1)

  def test_changed_content_uploads_again(self):
    self._publish("hash-a")
    self._publish("hash-b")
    self.assertEqual(len(self.api.uploaded), 2)

  def test_force_overrides_the_skip(self):
    self._publish("hash-a")
    self._publish("hash-a", force=True)
    self.assertEqual(len(self.api.uploaded), 2)

  def test_a_failed_publish_is_not_recorded(self):
    # Otherwise a 503 would mark the snapshot as published and the next
    # attempt would skip it as "unchanged", losing it entirely.
    self.api = _FakeApi(fail_on="upload_folder")
    self._publish("hash-a")
    self.assertIsNone(publish_mod.last_published_hash(self.state_file))

  def test_state_file_records_the_target(self):
    self._publish("hash-a")
    with open(self.state_file, "r", encoding="utf-8") as handle:
      state = json.load(handle)
    self.assertEqual(state["content_hash"], "hash-a")
    self.assertEqual(state["target"], "leobianco/auto-perl")
    self.assertEqual(state["backend"], "hf_space")


class MisconfigurationTest(unittest.TestCase):
  """Missing arguments are reported, not crashed on."""

  def test_space_backend_needs_a_repo_id(self):
    result = publish_mod.publish(
        site_dir=".", content_hash="x", backend="hf_space", repo_id=None
    )
    self.assertIn("--repo-id", result.error)

  def test_dir_backend_needs_a_destination(self):
    result = publish_mod.publish(
        site_dir=".", content_hash="x", backend="dir", destination=None
    )
    self.assertIn("--destination", result.error)

  def test_unknown_backend(self):
    result = publish_mod.publish(
        site_dir=".", content_hash="x", backend="carrier-pigeon"
    )
    self.assertIn("unknown backend", result.error)


class _Clock:
  """A monotonic clock the tests advance by hand."""

  def __init__(self):
    self.now = 1000.0

  def __call__(self) -> float:
    return self.now

  def advance(self, seconds: float) -> None:
    """Moves the clock forward.

    Args:
      seconds: How far to advance.
    """
    self.now += seconds


class _RecordingPublisher:
  """Captures publish calls made by the watcher."""

  def __init__(self, error: str = ""):
    """Initializes the double.

    Args:
      error: When set, every publish reports this error.
    """
    self.calls = []
    self.error = error

  def __call__(self, **kwargs):
    """Records a publish call and returns a synthetic result."""
    self.calls.append(kwargs)
    return publish_mod.PublishResult(
        backend=kwargs.get("backend", "dir"),
        target=kwargs.get("repo_id") or kwargs.get("destination") or "",
        published=not self.error,
        error=self.error or None,
    )


class WatcherTest(unittest.TestCase):
  """When the watcher decides to publish."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
    self.clock = _Clock()
    self.publisher = _RecordingPublisher()
    self.config = watcher_mod.WatchConfig(
        root=self.root,
        out_dir=os.path.join(self.root, "site"),
        backend="dir",
        destination=os.path.join(self.root, "published"),
        debounce_seconds=60.0,
        state_file=os.path.join(self.root, "publish.json"),
    )
    self.watcher = watcher_mod.Watcher(
        self.config, clock=self.clock, publisher=self.publisher
    )

  def _write(self, name="npov_campaign_2609211206", status="IN_PROGRESS",
             updated="2026-09-21T12:00:00", trials=1):
    """Writes a campaign state file.

    Args:
      name: Campaign id.
      status: Campaign status.
      updated: Last-update timestamp.
      trials: Trials recorded for the SFT stage.
    """
    state = fixtures.campaign_state(
        name=name, status=status, updated_at=updated
    )
    state["stages"]["sft"]["trials_done"] = trials
    fixtures.write_campaign(self.root, state=state, with_report=False)

  def test_first_tick_with_campaigns_publishes(self):
    self._write()
    outcome = self.watcher.tick()
    self.assertTrue(outcome.changed)
    self.assertEqual(len(self.publisher.calls), 1)

  def test_empty_repository_publishes_nothing(self):
    outcome = self.watcher.tick()
    self.assertFalse(outcome.changed)
    self.assertEqual(self.publisher.calls, [])

  def test_unchanged_tree_does_not_publish_again(self):
    self._write()
    self.watcher.tick()
    self.clock.advance(300)
    self.watcher.tick()
    self.assertEqual(len(self.publisher.calls), 1)

  def test_progress_within_the_debounce_is_held(self):
    self._write()
    self.watcher.tick()
    self.clock.advance(10)
    self._write(updated="2026-09-21T12:05:00", trials=2)
    outcome = self.watcher.tick()
    self.assertTrue(outcome.changed)
    self.assertEqual(outcome.reason, "debounced")
    self.assertEqual(len(self.publisher.calls), 1)

  def test_held_progress_publishes_once_the_debounce_elapses(self):
    self._write()
    self.watcher.tick()
    self.clock.advance(10)
    self._write(updated="2026-09-21T12:05:00", trials=2)
    self.watcher.tick()
    self.clock.advance(60)
    self.watcher.tick()
    self.assertEqual(len(self.publisher.calls), 2)

  def test_terminal_transition_bypasses_the_debounce(self):
    # The shutdown_when_done case: the VM powers off about a minute after
    # this transition, so the final snapshot cannot wait for a timer.
    self._write()
    self.watcher.tick()
    self.clock.advance(1)
    self._write(status="COMPLETED", updated="2026-09-21T12:06:00")
    outcome = self.watcher.tick()
    self.assertEqual(outcome.finished, ["npov-campaign-2609211206"])
    self.assertEqual(len(self.publisher.calls), 2)

  def test_every_terminal_status_counts_as_finished(self):
    for status in ("COMPLETED", "FAILED", "STOPPED", "ABORTED"):
      with self.subTest(status=status):
        previous = {"a": ("IN_PROGRESS", "t0", 0.0)}
        current = {"a": (status, "t1", 0.0)}
        self.assertEqual(watcher_mod.newly_finished(previous, current), ["a"])

  def test_paused_is_not_a_terminal_transition(self):
    previous = {"a": ("IN_PROGRESS", "t0", 0.0)}
    current = {"a": ("PAUSED", "t1", 0.0)}
    self.assertEqual(watcher_mod.newly_finished(previous, current), [])

  def test_a_campaign_first_seen_finished_is_not_a_transition(self):
    # Starting the watcher next to last week's campaigns must not be read
    # as fifteen campaigns finishing at once.
    current = {"a": ("COMPLETED", "t1", 0.0)}
    self.assertEqual(watcher_mod.newly_finished({}, current), [])

  def test_a_failed_publish_is_retried_on_the_next_tick(self):
    self.publisher = _RecordingPublisher(error="503")
    self.watcher = watcher_mod.Watcher(
        self.config, clock=self.clock, publisher=self.publisher
    )
    self._write()
    with self.assertLogs(watcher_mod.logger, level="WARNING"):
      self.watcher.tick()
    self.assertEqual(len(self.publisher.calls), 1)
    self.clock.advance(120)
    with self.assertLogs(watcher_mod.logger, level="WARNING"):
      self.watcher.tick()
    self.assertEqual(len(self.publisher.calls), 2)

  def test_dry_run_never_publishes(self):
    self.config.dry_run = True
    self._write()
    self.watcher.tick()
    self.assertEqual(self.publisher.calls, [])

  def test_a_new_report_counts_as_a_change(self):
    self._write(status="COMPLETED")
    self.watcher.tick()
    self.clock.advance(120)
    fixtures.write_campaign(
        self.root,
        state=fixtures.campaign_state(status="COMPLETED"),
        with_report=True,
    )
    outcome = self.watcher.tick()
    self.assertTrue(outcome.changed)
    self.assertEqual(len(self.publisher.calls), 2)

  def test_run_stops_after_max_ticks(self):
    self.config.poll_seconds = 0.0
    self._write()
    self.assertEqual(self.watcher.run(max_ticks=3), 3)

  def test_run_survives_a_broken_tick(self):
    def _explode(**_):
      raise RuntimeError("disk on fire")

    self.config.poll_seconds = 0.0
    watcher = watcher_mod.Watcher(
        self.config, clock=self.clock, builder=_explode,
        publisher=self.publisher,
    )
    self._write()
    with self.assertLogs(watcher_mod.logger, level="WARNING"):
      self.assertEqual(watcher.run(max_ticks=2), 2)

  def test_request_stop_ends_the_loop(self):
    self.config.poll_seconds = 0.0
    self.watcher.request_stop()
    self.assertEqual(self.watcher.run(max_ticks=100), 0)


class WatcherSignatureTest(unittest.TestCase):
  """What the watcher considers a change."""

  def setUp(self):
    super().setUp()
    self.root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

  def test_signature_tracks_status_and_update_time(self):
    fixtures.write_campaign(self.root, with_report=False)
    campaigns = index_mod.build_index(self.root)
    signature = watcher_mod.signature_of(campaigns, self.root)
    status, updated, mtime = signature["npov-campaign-2609211206"]
    self.assertEqual(status, "COMPLETED")
    self.assertTrue(updated)
    self.assertEqual(mtime, 0.0)

  def test_report_mtime_is_part_of_the_signature(self):
    fixtures.write_campaign(self.root, with_report=True)
    campaigns = index_mod.build_index(self.root)
    signature = watcher_mod.signature_of(campaigns, self.root)
    self.assertGreater(signature["npov-campaign-2609211206"][2], 0.0)


class WatcherIntegrationTest(unittest.TestCase):
  """The watcher with the real builder and the directory backend."""

  def test_publishes_a_real_site_to_a_directory(self):
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    destination = os.path.join(root, "published")
    fixtures.write_campaign(root)
    config = watcher_mod.WatchConfig(
        root=root,
        out_dir=os.path.join(root, "site"),
        backend="dir",
        destination=destination,
        state_file=os.path.join(root, "publish.json"),
    )
    watcher = watcher_mod.Watcher(config)
    outcome = watcher.tick()
    self.assertTrue(outcome.published.published)
    self.assertTrue(os.path.isfile(os.path.join(destination, "index.html")))
    self.assertTrue(
        os.path.isfile(
            os.path.join(
                destination, "campaign", "npov-campaign-2609211206.html"
            )
        )
    )

  def test_build_result_hash_is_stable_across_rebuilds(self):
    root = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, root, ignore_errors=True)
    fixtures.write_campaign(root)
    first = build_mod.build_site(root=root, out_dir=os.path.join(root, "a"))
    second = build_mod.build_site(root=root, out_dir=os.path.join(root, "b"))
    # Only the clock differs between these two builds, and the fingerprint
    # deliberately ignores it - otherwise every poll would be a commit.
    self.assertEqual(first.content_hash, second.content_hash)
    self.assertNotEqual(first.generated_at, second.generated_at)


if __name__ == "__main__":
  unittest.main()
