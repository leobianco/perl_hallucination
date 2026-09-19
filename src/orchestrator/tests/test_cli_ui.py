"""Tests for the Auto-PERL terminal UI layer (theme, events, rendering).

These tests are dependency-free: everything that needs ``rich`` is skipped
when the package is absent, and the plain-text code paths are always
exercised. That mirrors the production requirement that the CLI must work on
a bare GCP VM before ``pip install -r requirements.txt`` has finished.
"""

from __future__ import annotations

import io
import os
import threading
import time
import unittest
from unittest import mock

from src.orchestrator.cli import console as console_mod
from src.orchestrator.cli import dashboard as dashboard_mod
from src.orchestrator.cli import events as events_mod
from src.orchestrator.cli import renderables
from src.orchestrator.cli import theme as theme_mod
from src.orchestrator.config import CampaignConfig
from src.orchestrator.state import CampaignState, StageResult, StageStatus

HAS_RICH = renderables.rich_available()
requires_rich = unittest.skipUnless(HAS_RICH, "rich is not installed")


def make_state(**stages) -> CampaignState:
  """Builds a campaign state pre-populated with stage results."""
  state = CampaignState(campaign_id="test_camp", task_name="npov")
  for name, result in stages.items():
    state.stages[name] = result
  return state


class ThemeDetectionTest(unittest.TestCase):
  """Terminal capability detection and overrides."""

  def test_no_color_env_disables_color(self):
    with mock.patch.dict("os.environ", {"NO_COLOR": "1"}, clear=False):
      theme = theme_mod.detect_theme(stream=io.StringIO())
      self.assertFalse(theme.use_color)
      self.assertEqual(theme.markup("hello", "accent"), "hello")

  def test_force_color_env_enables_color(self):
    with mock.patch.dict("os.environ", {"FORCE_COLOR": "1"}, clear=False):
      theme = theme_mod.detect_theme(stream=io.StringIO())
      self.assertTrue(theme.use_color)

  def test_explicit_overrides_win(self):
    theme = theme_mod.detect_theme(force_color=True, force_ascii=True)
    self.assertTrue(theme.use_color)
    self.assertFalse(theme.use_unicode)
    self.assertEqual(theme.glyphs.completed, "+")

  def test_non_tty_stream_has_no_color(self):
    with mock.patch.dict("os.environ", {}, clear=True):
      theme = theme_mod.detect_theme(stream=io.StringIO())
      self.assertFalse(theme.use_color)

  def test_ascii_glyphs_are_encodable_in_latin1(self):
    glyphs = theme_mod.Glyphs.ascii()
    for value in (glyphs.completed, glyphs.failed, glyphs.bar_full, glyphs.arrow):
      value.encode("ascii")  # Must not raise.

  def test_unicode_detection_respects_encoding(self):
    stream = io.StringIO()
    self.assertTrue(theme_mod.supports_unicode(stream))

    class AsciiStream(io.StringIO):
      encoding = "ascii"

    self.assertFalse(theme_mod.supports_unicode(AsciiStream()))


class FormattingTest(unittest.TestCase):
  """Number/duration/markup formatting helpers."""

  def test_format_duration_ranges(self):
    cases = {
        None: "-",
        0: "0s",
        9.4: "9s",
        65: "1m 05s",
        3600: "1h 00m",
        7625: "2h 07m",
        90000: "1d 01h",
        float("nan"): "-",
        float("inf"): "-",
        -5: "0s",
    }
    for value, expected in cases.items():
      self.assertEqual(theme_mod.format_duration(value), expected, msg=str(value))

  def test_format_metric_precision(self):
    self.assertEqual(theme_mod.format_metric(None), "-")
    self.assertEqual(theme_mod.format_metric(0.3123456), "0.3123")
    self.assertEqual(theme_mod.format_metric(0.31235), "0.3124")
    self.assertEqual(theme_mod.format_metric(1234.5), "1234.50")
    self.assertEqual(theme_mod.format_metric(0.00001), "1.000e-05")
    self.assertEqual(theme_mod.format_metric(7), "7")
    self.assertEqual(theme_mod.format_metric("n/a"), "n/a")
    self.assertEqual(theme_mod.format_metric(float("nan")), "nan")

  def test_format_count_pads_consistently(self):
    self.assertEqual(theme_mod.format_count(3, 30), "03/30")
    self.assertEqual(theme_mod.format_count(0, 5), "0/5")
    self.assertEqual(theme_mod.format_count(99, 5), "5/5")
    self.assertEqual(theme_mod.format_count(None, 0), "-")

  def test_progress_bar_width_and_clamping(self):
    theme = theme_mod.detect_theme(force_color=False)
    bar = theme_mod.progress_bar(0, 10, 20, theme)
    self.assertEqual(len(theme_mod.strip_markup(bar)), 20)
    self.assertNotIn(theme.glyphs.bar_full, bar)

    full = theme_mod.strip_markup(theme_mod.progress_bar(10, 10, 20, theme))
    self.assertEqual(full, theme.glyphs.bar_full * 20)

    # A single finished trial must be visible, and an unfinished sweep must
    # never look 100% complete.
    tiny = theme_mod.strip_markup(theme_mod.progress_bar(1, 1000, 10, theme))
    self.assertEqual(tiny.count(theme.glyphs.bar_full), 1)
    almost = theme_mod.strip_markup(theme_mod.progress_bar(999, 1000, 10, theme))
    self.assertEqual(almost.count(theme.glyphs.bar_empty), 1)

  def test_progress_bar_zero_total(self):
    theme = theme_mod.detect_theme(force_color=False)
    bar = theme_mod.strip_markup(theme_mod.progress_bar(0, 0, 8, theme))
    self.assertEqual(len(bar), 8)

  def test_sparkline(self):
    theme = theme_mod.detect_theme(force_color=False)
    self.assertEqual(theme_mod.sparkline([], theme), "")
    spark = theme_mod.sparkline([1, 2, 3, 4], theme)
    self.assertEqual(len(spark), 4)
    self.assertEqual(spark[0], theme.glyphs.spark[0])
    self.assertEqual(spark[-1], theme.glyphs.spark[-1])
    flat = theme_mod.sparkline([2, 2, 2], theme)
    self.assertEqual(len(set(flat)), 1)
    windowed = theme_mod.sparkline(list(range(50)), theme, width=10)
    self.assertEqual(len(windowed), 10)
    self.assertEqual(theme_mod.sparkline([float("nan"), None], theme), "")

  def test_truncate_keeps_head_and_tail(self):
    text = "leobianco/npov_SFT_gemma-4-E2B-it_S130104_lr0.003"
    short = theme_mod.truncate(text, 20)
    self.assertEqual(len(short), 20)
    self.assertTrue(short.startswith("leobianc"))
    self.assertTrue(short.endswith("lr0.003"))
    self.assertEqual(theme_mod.truncate("abc", 10), "abc")

  def test_markup_escaping_preserves_literal_brackets(self):
    theme = theme_mod.detect_theme(force_color=True)
    rendered = theme.markup("[DRY-RUN] Trial 1/3", "accent")
    self.assertIn("\\[DRY-RUN]", rendered)
    self.assertEqual(
        theme_mod.strip_markup(rendered), "[DRY-RUN] Trial 1/3"
    )

  def test_strip_markup_removes_style_tags_only(self):
    self.assertEqual(theme_mod.strip_markup("[bold red]hi[/bold red]"), "hi")
    self.assertEqual(theme_mod.strip_markup("[RESUME] skipping"), "[RESUME] skipping")
    self.assertEqual(theme_mod.strip_markup("a [1, 2] b"), "a [1, 2] b")

  def test_iter_stage_names_normalizes(self):
    self.assertEqual(
        theme_mod.iter_stage_names([" SFT ", "rm", "sft", "", None]),
        ["sft", "rm"],
    )


class EventBusTest(unittest.TestCase):
  """Publish/subscribe semantics of the campaign event bus."""

  def test_emit_and_history(self):
    bus = events_mod.EventBus()
    seen = []
    bus.subscribe(seen.append)
    bus.publish(events_mod.EventType.LOG, message="hello", stage="sft")
    self.assertEqual(len(seen), 1)
    self.assertEqual(seen[0].stage, "sft")
    self.assertEqual(bus.history[0].message, "hello")
    self.assertRegex(seen[0].clock, r"^\d{2}:\d{2}:\d{2}$")

  def test_replay_and_unsubscribe(self):
    bus = events_mod.EventBus()
    bus.log("first")
    late = []
    unsubscribe = bus.subscribe(late.append, replay=True)
    self.assertEqual(len(late), 1)
    unsubscribe()
    bus.log("second")
    self.assertEqual(len(late), 1)

  def test_subscriber_exception_does_not_break_bus(self):
    bus = events_mod.EventBus()
    good = []
    bus.subscribe(lambda event: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(good.append)
    bus.log("still delivered")
    self.assertEqual(len(good), 1)

  def test_history_is_bounded(self):
    bus = events_mod.EventBus(history_limit=5)
    for index in range(20):
      bus.log(str(index))
    history = bus.history
    self.assertEqual(len(history), 5)
    self.assertEqual(history[-1].message, "19")

  def test_line_callback_adapter(self):
    bus = events_mod.EventBus()
    seen = []
    bus.subscribe(seen.append)
    callback = bus.as_line_callback(stage="rm")
    callback("agent output")
    self.assertEqual(seen[0].type, events_mod.EventType.LOG)
    self.assertEqual(seen[0].stage, "rm")

  def test_event_serialization(self):
    event = events_mod.CampaignEvent(
        type=events_mod.EventType.STAGE_COMPLETED, stage="sft", message="done"
    )
    payload = event.to_dict()
    self.assertEqual(payload["type"], "STAGE_COMPLETED")
    self.assertEqual(payload["stage"], "sft")

  def test_thread_safety(self):
    bus = events_mod.EventBus(history_limit=10000)
    received = []
    lock = threading.Lock()

    def collect(event):
      with lock:
        received.append(event)

    bus.subscribe(collect)

    def worker(index):
      for step in range(50):
        bus.log(f"{index}-{step}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    self.assertEqual(len(received), 200)
    self.assertEqual(len(bus.history), 200)


class ControlSignalsTest(unittest.TestCase):
  """Pause / advance / stop are independent intents."""

  def test_pause_does_not_stop(self):
    controls = events_mod.ControlSignals()
    controls.pause()
    self.assertTrue(controls.is_paused)
    self.assertFalse(controls.stop_requested)
    # A paused campaign must not look like an interrupted sweep.
    self.assertFalse(controls.should_interrupt_stage())

  def test_toggle_pause(self):
    controls = events_mod.ControlSignals()
    self.assertTrue(controls.toggle_pause())
    self.assertFalse(controls.toggle_pause())

  def test_advance_interrupts_stage_only(self):
    controls = events_mod.ControlSignals()
    controls.request_advance()
    self.assertTrue(controls.should_interrupt_stage())
    self.assertFalse(controls.stop_requested)
    controls.clear_advance()
    self.assertFalse(controls.should_interrupt_stage())

  def test_stop_releases_pause(self):
    controls = events_mod.ControlSignals()
    controls.pause()
    controls.request_stop()
    self.assertFalse(controls.wait_while_paused(poll_seconds=0.01))
    self.assertTrue(controls.stop_requested)

  def test_wait_while_paused_unblocks_on_resume(self):
    controls = events_mod.ControlSignals()
    controls.pause()
    result = {}

    def waiter():
      result["ok"] = controls.wait_while_paused(poll_seconds=0.01)

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.05)
    self.assertTrue(thread.is_alive())
    controls.resume()
    thread.join(timeout=2)
    self.assertFalse(thread.is_alive())
    self.assertTrue(result["ok"])

  def test_abort_implies_stop(self):
    controls = events_mod.ControlSignals()
    controls.request_abort()
    self.assertTrue(controls.abort_requested)
    self.assertTrue(controls.stop_requested)
    self.assertEqual(
        controls.snapshot(),
        {"paused": False, "advance": False, "stop": True, "abort": True},
    )

  def test_listeners_are_notified(self):
    controls = events_mod.ControlSignals()
    actions = []
    controls.on_change(actions.append)
    controls.pause()
    controls.resume()
    controls.request_stop()
    self.assertEqual(actions, ["pause", "resume", "resume", "stop"])


class SweepProgressParserTest(unittest.TestCase):
  """Rebuilding a leaderboard from raw agent output."""

  AGENT_OUTPUT = [
      "wandb: Agent Starting Run: abc123 with config:",
      "wandb: \tlearning_rate: 0.0003",
      "wandb: \tlora_r: 8",
      "some training noise",
      "eval/loss: 0.42",
      "wandb: Agent Finished Run: abc123",
      "wandb: Agent Starting Run: def456 with config:",
      "wandb: \tlearning_rate: 0.001",
      "eval/loss: 0.31",
      "wandb: Agent Finished Run: def456",
  ]

  def test_parses_trials_params_and_metrics(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss", max_trials=30)
    for line in self.AGENT_OUTPUT:
      parser.feed(line)
    self.assertEqual(len(parser.trials), 2)
    self.assertEqual(parser.completed_count, 2)
    first = parser.trials[0]
    self.assertEqual(first.run_id, "abc123")
    self.assertAlmostEqual(first.params["learning_rate"], 0.0003)
    self.assertEqual(first.params["lora_r"], 8)
    self.assertAlmostEqual(first.metric, 0.42)
    self.assertIsNone(parser.active_trial)

  def test_leaderboard_respects_goal(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    for line in self.AGENT_OUTPUT:
      parser.feed(line)
    best_min = parser.best_trial("minimize")
    best_max = parser.best_trial("maximize")
    self.assertEqual(best_min.run_id, "def456")
    self.assertEqual(best_max.run_id, "abc123")
    board = parser.leaderboard(goal="minimize", limit=1)
    self.assertEqual(len(board), 1)
    self.assertEqual(board[0].run_id, "def456")

  def test_running_trial_tracked(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: zzz with config:")
    self.assertIsNotNone(parser.active_trial)
    self.assertEqual(parser.completed_count, 0)
    self.assertEqual(parser.active_trial.state, "running")

  def test_dry_run_lines(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("[DRY-RUN] Trial 1/3 executed successfully (metric simulated).")
    parser.feed("[DRY-RUN] Trial 2/3 executed successfully (metric simulated).")
    self.assertEqual(parser.completed_count, 2)
    self.assertEqual(parser.max_trials, 3)

  def test_unknown_lines_are_ignored(self):
    parser = events_mod.SweepProgressParser()
    for line in ["", "random noise", "==== separator ===="]:
      self.assertIsNone(parser.feed(line))
    self.assertEqual(parser.trials, [])

  def test_unscored_trials_sort_last(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: a with config:")
    parser.feed("eval/loss: 0.5")
    parser.feed("wandb: Agent Finished Run: a")
    parser.feed("wandb: Agent Starting Run: b with config:")
    board = parser.leaderboard(goal="minimize")
    self.assertEqual([t.run_id for t in board], ["a", "b"])


class RealAgentOutputParserTest(unittest.TestCase):
  """The output a piped, non-interactive ``wandb agent`` actually produces.

  This is the shape that froze the dashboard at ``00/N``: the agent logs
  through the ``wandb.wandb_agent`` logger, announces runs without an id
  ("Agent starting run with config:") and closes them with "Cleaning up
  finished run: <id>" rather than "Agent Finished Run: <id>".
  """

  #: One complete two-trial SFT sweep, interleaving the logger shape, the
  #: termlog shape and HF Trainer output, exactly as it reaches the parser.
  AGENT_OUTPUT = [
      "wandb: Starting wandb agent",
      "2026-09-15 11:30:46,123 - wandb.wandb_agent - INFO - Running runs: []",
      "2026-09-15 11:30:46,456 - wandb.wandb_agent - INFO - Agent received"
      " command: run",
      "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - Agent starting run"
      " with config:",
      "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - \tlearning_rate:"
      " 0.0003",
      "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - \tlora_r: 8",
      "2026-09-15 11:30:46,458 - wandb.wandb_agent - INFO - About to run"
      " command: /usr/bin/env python train_sft.py",
      "wandb: Agent Starting Run: k8jd92la with config:",
      "wandb: \tlearning_rate: 0.0003",
      "{'eval_loss': 0.42, 'eval_runtime': 12.3, 'epoch': 1.0}",
      "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
      " finished run: k8jd92la",
      "2026-09-15 11:45:11,000 - wandb.wandb_agent - INFO - Agent received"
      " command: run",
      "2026-09-15 11:45:11,001 - wandb.wandb_agent - INFO - Agent starting run"
      " with config:",
      "2026-09-15 11:45:11,001 - wandb.wandb_agent - INFO - \tlearning_rate:"
      " 0.001",
      "wandb: Agent Starting Run: p2mx77qq with config:",
      "{'eval_loss': 0.31, 'epoch': 1.0}",
      "2026-09-15 12:02:00,000 - wandb.wandb_agent - INFO - Cleaning up"
      " finished run: p2mx77qq",
  ]

  def feed_all(self, metric_name="eval/loss", max_trials=15):
    parser = events_mod.SweepProgressParser(
        metric_name=metric_name, max_trials=max_trials
    )
    for line in self.AGENT_OUTPUT:
      parser.feed(line)
    return parser

  def test_trial_counter_advances(self):
    parser = self.feed_all()
    # The regression: this used to stay at 0 for the whole sweep.
    self.assertEqual(parser.completed_count, 2)
    self.assertEqual(parser.max_trials, 15)

  def test_logger_and_termlog_announcements_are_one_trial(self):
    parser = self.feed_all()
    self.assertEqual(len(parser.trials), 2)
    self.assertEqual(
        [t.run_id for t in parser.trials], ["k8jd92la", "p2mx77qq"]
    )
    self.assertFalse(any(t.provisional for t in parser.trials))

  def test_metrics_and_params_populate_the_leaderboard(self):
    parser = self.feed_all()
    board = parser.leaderboard(goal="minimize")
    self.assertEqual([t.run_id for t in board], ["p2mx77qq", "k8jd92la"])
    self.assertAlmostEqual(board[0].metric, 0.31)
    self.assertAlmostEqual(board[1].metric, 0.42)
    first = parser.trials[0]
    self.assertAlmostEqual(first.params["learning_rate"], 0.0003)
    self.assertEqual(first.params["lora_r"], 8)

  def test_best_trial_tracks_goal(self):
    parser = self.feed_all()
    self.assertEqual(parser.best_trial("minimize").run_id, "p2mx77qq")
    self.assertEqual(parser.best_trial("maximize").run_id, "k8jd92la")

  def test_trial_is_counted_only_once_finished(self):
    parser = events_mod.SweepProgressParser(
        metric_name="eval/loss", max_trials=15
    )
    for line in self.AGENT_OUTPUT[:9]:
      parser.feed(line)
    self.assertEqual(parser.completed_count, 0)
    self.assertIsNotNone(parser.active_trial)
    self.assertEqual(parser.active_trial.run_id, "k8jd92la")

  def test_anonymous_run_survives_without_a_termlog_line(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed(
        "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - Agent starting"
        " run with config:"
    )
    self.assertEqual(parser.completed_count, 0)
    parser.feed(
        "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
        " finished run: abc999"
    )
    self.assertEqual(parser.completed_count, 1)
    self.assertEqual(parser.trials[0].run_id, "abc999")

  def test_failed_run_is_terminal(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: bad001 with config:")
    parser.feed("wandb: Run bad001 failed with exit code 1")
    self.assertEqual(parser.completed_count, 1)
    self.assertEqual(parser.trials[0].state, "failed")
    self.assertIsNone(parser.active_trial)

  def test_wandb_summary_block_metric(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/roc_auc")
    parser.feed("wandb: Agent Starting Run: rm001 with config:")
    parser.feed("wandb: Run summary:")
    parser.feed("wandb:   eval/roc_auc 0.9651")
    self.assertAlmostEqual(parser.trials[0].metric, 0.9651)

  def test_metric_alias_does_not_match_a_longer_key(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: x1 with config:")
    parser.feed("wandb: \teval_loss_weight: 3")
    self.assertIsNone(parser.trials[0].metric)

  def test_ansi_colored_lines_are_parsed(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("\x1b[34m\x1b[1mwandb\x1b[0m: Agent Starting Run: c1 with"
                " config:")
    parser.feed("\x1b[34mwandb\x1b[0m: Agent Finished Run: c1")
    self.assertEqual(parser.completed_count, 1)

  def test_strip_log_prefix_keeps_indentation(self):
    stripped = events_mod.strip_log_prefix(
        "2026-09-15 11:30:46,457 - wandb.wandb_agent - INFO - \tlora_r: 8"
    )
    self.assertEqual(stripped, "\tlora_r: 8")
    self.assertEqual(
        events_mod.strip_log_prefix("wandb: \tlora_r: 8"), "\tlora_r: 8"
    )


class MetricExtractionTest(unittest.TestCase):
  """Why the leaderboard and the DAG's "Best" cell used to stay empty.

  Three independent defects all surfaced as a ``-``: an alias that matched
  the wrong number, a metric flushed after the trial was already closed, and
  an alias expansion that mangled multi-word metric keys.
  """

  def test_training_loss_is_not_mistaken_for_eval_loss(self):
    # The HF Trainer interleaves {'loss': ...} (training) with
    # {'eval_loss': ...}. Degrading `eval/loss` to a bare `loss` alias made
    # the leaderboard rank trials on the training loss instead.
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: t1 with config:")
    parser.feed("{'loss': 1.9012, 'grad_norm': 3.2, 'epoch': 0.02}")
    self.assertIsNone(parser.trials[0].metric)
    parser.feed("{'eval_loss': 0.4231, 'eval_runtime': 11.8, 'epoch': 0.33}")
    self.assertAlmostEqual(parser.trials[0].metric, 0.4231)

  def test_unqualified_metric_still_matches_bare_key(self):
    parser = events_mod.SweepProgressParser(metric_name="accuracy")
    parser.feed("wandb: Agent Starting Run: t1 with config:")
    parser.feed("{'accuracy': 0.87}")
    self.assertAlmostEqual(parser.trials[0].metric, 0.87)

  def test_metric_flushed_after_cleanup_still_scores_the_trial(self):
    # A piped child is block buffered: its run-summary block can drain after
    # the agent has already logged the cleanup line.
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: t1 with config:")
    parser.feed(
        "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
        " finished run: t1"
    )
    self.assertIsNone(parser.active_trial)
    parser.feed("wandb: Run summary:")
    parser.feed("wandb:   eval/loss 0.4231")
    self.assertAlmostEqual(parser.trials[0].metric, 0.4231)
    self.assertAlmostEqual(parser.best_trial("minimize").metric, 0.4231)

  def test_late_metric_never_overwrites_a_scored_trial(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: t1 with config:")
    parser.feed("{'eval_loss': 0.40}")
    parser.feed("wandb: Agent Finished Run: t1")
    parser.feed("wandb:   eval/loss 9.99")
    self.assertAlmostEqual(parser.trials[0].metric, 0.40)

  def test_late_metric_does_not_leak_into_the_next_trial(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    parser.feed("wandb: Agent Starting Run: t1 with config:")
    parser.feed("wandb: Agent Finished Run: t1")
    parser.feed("wandb: Agent Starting Run: t2 with config:")
    parser.feed("{'eval_loss': 0.22}")
    self.assertIsNone(parser.trials[0].metric)
    self.assertAlmostEqual(parser.trials[1].metric, 0.22)

  def test_multi_word_segments_survive_alias_expansion(self):
    aliases = events_mod._metric_aliases(  # pylint: disable=protected-access
        "train/rewards/reward_fn/mean"
    )
    self.assertIn("train/rewards/reward_fn/mean", aliases)
    self.assertIn("rewards/reward_fn/mean", aliases)
    # `reward_fn` must not be split into `reward/fn`, and no alias may be
    # short enough to match an unrelated `mean`.
    self.assertNotIn("train/rewards/reward/fn/mean", aliases)
    self.assertNotIn("mean", aliases)

  def test_perl_reward_metric_is_extracted(self):
    parser = events_mod.SweepProgressParser(
        metric_name="train/rewards/reward_fn/mean"
    )
    parser.feed("wandb: Agent Starting Run: p1 with config:")
    parser.feed("{'rewards/reward_fn/mean': 0.6123, 'epoch': 0.5}")
    self.assertAlmostEqual(parser.trials[0].metric, 0.6123)


class StageViewTest(unittest.TestCase):
  """Derivation of DAG rows from config + state."""

  def setUp(self):
    self.config = CampaignConfig.create_default(
        task_name="npov", sft_runs=10, rm_runs=8, perl_runs=4
    )

  def test_pending_campaign(self):
    views = renderables.build_stage_views(self.config, make_state())
    self.assertEqual([v.key for v in views], ["sft", "rm", "perl", "eval"])
    self.assertTrue(all(v.status == "PENDING" for v in views))
    self.assertEqual(views[0].trials_total, 10)
    self.assertEqual(views[0].fraction, 0.0)

  def test_completed_stage_without_accounting_is_drawn_full(self):
    # State files written before trial accounting existed carry no counters.
    # Drawing such a stage as 00/10 would be worse than assuming it ran, so
    # the bar is full - but only when nothing at all was recorded.
    state = make_state(
        sft=StageResult(
            status=StageStatus.COMPLETED,
            best_metric_val=0.31,
            model_repo_id="leobianco/npov_SFT",
            start_time="2026-09-14T10:00:00",
            end_time="2026-09-14T10:30:00",
        )
    )
    views = renderables.build_stage_views(self.config, state)
    sft = views[0]
    self.assertEqual(sft.status, "COMPLETED")
    self.assertEqual(sft.fraction, 1.0)
    self.assertFalse(sft.is_partial)
    done, total, _ = renderables._bar_values(sft)  # pylint: disable=protected-access
    self.assertEqual((done, total), (10, 10))
    self.assertAlmostEqual(sft.elapsed_s, 1800.0)
    self.assertEqual(sft.model_repo_id, "leobianco/npov_SFT")

  def test_completed_stage_reports_the_trials_it_actually_ran(self):
    # The regression this file used to enshrine: a COMPLETED stage had its
    # counter overwritten with the budget, so a sweep that lost trials to a
    # crash rendered as a clean full bar.
    state = make_state(
        sft=StageResult(
            status=StageStatus.COMPLETED,
            best_metric_val=0.31,
            trials_done=6,
            trials_total=10,
            sweep_outcome="partial",
        )
    )
    sft = renderables.build_stage_views(self.config, state)[0]
    self.assertEqual(sft.trials_done, 6)
    self.assertTrue(sft.is_partial)
    self.assertAlmostEqual(sft.fraction, 0.6)

  def test_eval_metric_uses_hallucination_rate(self):
    state = make_state(
        eval=StageResult(
            status=StageStatus.COMPLETED,
            metrics={"hallucination_rate": 0.05, "bertscore_f1": 0.9},
        )
    )
    views = renderables.build_stage_views(self.config, state)
    self.assertAlmostEqual(views[-1].metric_value, 0.05)

  def test_live_overrides_applied(self):
    views = renderables.build_stage_views(
        self.config,
        make_state(),
        live={"rm": {"status": "RUNNING", "trials_done": 3, "metric_value": 0.9}},
    )
    rm = views[1]
    self.assertTrue(rm.is_active)
    self.assertEqual(rm.trials_done, 3)
    self.assertAlmostEqual(rm.fraction, 3 / 8)

  def test_failed_stage_keeps_error(self):
    state = make_state(
        perl=StageResult(status=StageStatus.FAILED, error_message="CUDA OOM")
    )
    views = renderables.build_stage_views(self.config, state)
    self.assertEqual(views[2].error, "CUDA OOM")

  def test_running_stage_elapsed_uses_now(self):
    state = CampaignState(campaign_id="c", task_name="npov")
    state.mark_stage_running("sft")
    views = renderables.build_stage_views(self.config, state)
    self.assertIsNotNone(views[0].elapsed_s)
    self.assertLess(views[0].elapsed_s, 5)

  def test_partial_stage_selection(self):
    self.config.stages = ["perl", "eval"]
    views = renderables.build_stage_views(self.config, make_state())
    self.assertEqual([v.key for v in views], ["perl", "eval"])


class TextRenderingTest(unittest.TestCase):
  """Plain-text rendering of the DAG and helpers."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov")
    self.theme = theme_mod.detect_theme(force_color=False)

  def test_dag_lines_contain_stage_titles(self):
    views = renderables.build_stage_views(self.config, make_state())
    lines = renderables.dag_lines(views, self.theme, width=120)
    self.assertEqual(len(lines), 4)
    self.assertIn("SFT Sweep", lines[0])
    self.assertIn("Final Evaluation", lines[3])

  def test_dag_lines_have_no_leftover_markup(self):
    views = renderables.build_stage_views(self.config, make_state())
    for line in renderables.dag_lines(views, self.theme, width=100):
      self.assertNotIn("[bold", line)
      self.assertNotIn("[/", line)

  def test_dag_lines_adapt_to_width(self):
    state = make_state(
        sft=StageResult(
            status=StageStatus.COMPLETED, model_repo_id="leobianco/npov_SFT_x"
        )
    )
    views = renderables.build_stage_views(self.config, state)
    narrow = renderables.dag_lines(views, self.theme, width=60)[0]
    wide = renderables.dag_lines(views, self.theme, width=140)[0]
    self.assertLess(len(narrow), len(wide))
    self.assertIn("leobianco/npov_SFT_x", wide)
    self.assertNotIn("leobianco/npov_SFT_x", narrow)

  def test_non_sweep_stage_shows_done(self):
    state = make_state(eval=StageResult(status=StageStatus.COMPLETED))
    views = renderables.build_stage_views(self.config, state)
    line = renderables.dag_lines(views, self.theme, width=100)[3]
    self.assertIn("done", line)
    self.assertIn(self.theme.glyphs.bar_full, line)

  def test_hotkey_hint_collapses_on_narrow_terminals(self):
    wide = renderables.hotkey_hint(self.theme, width=120)
    narrow = renderables.hotkey_hint(self.theme, width=70)
    self.assertIn("advance w/ best", wide)
    self.assertNotIn("advance w/ best", narrow)
    self.assertIn("[a]", narrow)
    self.assertLess(len(narrow), len(wide))

  def test_hotkey_hint_reflects_pause_state(self):
    self.assertIn("resume", renderables.hotkey_hint(self.theme, paused=True))
    self.assertIn("pause", renderables.hotkey_hint(self.theme, paused=False))

  def test_summary_lines(self):
    state = make_state(sft=StageResult(status=StageStatus.COMPLETED))
    state.status = "COMPLETED"
    lines = renderables.summary_lines(self.config, state, self.theme)
    self.assertIn("test_camp", lines[0])

  def test_banner_ascii_mode(self):
    ascii_theme = theme_mod.detect_theme(force_color=False, force_ascii=True)
    art = renderables.banner(ascii_theme, "subtitle")
    art.encode("ascii")  # Must not raise.
    self.assertIn("subtitle", art)


@requires_rich
class RichRenderingTest(unittest.TestCase):
  """Rendering through an in-memory rich console at a fixed width."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov", sft_runs=4)
    self.theme = theme_mod.detect_theme(force_color=False, force_ascii=False)

  def render(self, renderable, width=120) -> str:
    from rich.console import Console  # pylint: disable=g-import-not-at-top

    buffer = io.StringIO()
    Console(file=buffer, width=width, no_color=True, highlight=False).print(renderable)
    return buffer.getvalue()

  def test_dag_table(self):
    state = make_state(
        sft=StageResult(status=StageStatus.COMPLETED, best_metric_val=0.312)
    )
    views = renderables.build_stage_views(self.config, state)
    text = self.render(renderables.dag_table(views, self.theme, width=120))
    self.assertIn("SFT Sweep", text)
    self.assertIn("0.3120", text)
    self.assertIn("4/4", text)

  def test_status_table_shows_marker_on_narrow_terminal(self):
    state = make_state(sft=StageResult(status=StageStatus.COMPLETED))
    text = self.render(
        renderables.status_table(self.config, state, self.theme, width=80),
        width=80,
    )
    # Regression: cell padding used to squeeze the glyph column to zero.
    self.assertIn(self.theme.glyphs.completed, text)
    self.assertIn("done", text)

  def test_status_table_hides_artifact_column_when_narrow(self):
    state = make_state(
        sft=StageResult(
            status=StageStatus.COMPLETED, model_repo_id="leobianco/npov_SFT"
        )
    )
    narrow = self.render(
        renderables.status_table(self.config, state, self.theme, width=80), width=80
    )
    wide = self.render(
        renderables.status_table(self.config, state, self.theme, width=140), width=140
    )
    self.assertNotIn("Artifact", narrow)
    self.assertIn("Artifact", wide)
    self.assertIn("leobianco/npov_SFT", wide)

  def test_leaderboard_table(self):
    parser = events_mod.SweepProgressParser(metric_name="eval/loss")
    for line in SweepProgressParserTest.AGENT_OUTPUT:
      parser.feed(line)
    text = self.render(
        renderables.leaderboard_table(
            parser.leaderboard("minimize"), self.theme, "eval/loss", "minimize"
        )
    )
    self.assertIn("def456", text)
    self.assertIn("0.3100", text)
    self.assertIn("learning_rate=", text)

  def test_empty_leaderboard_is_friendly(self):
    text = self.render(renderables.leaderboard_table([], self.theme))
    self.assertIn("waiting for first trial", text)

  def test_header_panel_badges(self):
    state = make_state()
    state.status = "IN_PROGRESS"
    text = self.render(
        renderables.header_panel(
            self.config, state, self.theme, elapsed_s=3700, paused=True, dry_run=True
        )
    )
    self.assertIn("DRY-RUN", text)
    self.assertIn("PAUSED", text)
    self.assertIn("1h 01m", text)
    self.assertIn("npov", text)

  def test_help_panel_lists_all_keys(self):
    text = self.render(renderables.help_panel(self.theme))
    for key, _ in renderables.HELP_TEXT:
      self.assertIn(key, text)


class UiConsoleTest(unittest.TestCase):
  """Output surface behaviour in both plain and rich modes."""

  def test_plain_console_strips_markup(self):
    buffer = io.StringIO()
    console = console_mod.UiConsole(
        theme=theme_mod.detect_theme(force_color=False),
        file=buffer,
        force_plain=True,
    )
    self.assertFalse(console.is_rich)
    console.print("[bold red]danger[/bold red]")
    console.success("saved")
    console.error("kaboom")
    output = buffer.getvalue()
    self.assertIn("danger", output)
    self.assertNotIn("[bold red]", output)
    self.assertIn("saved", output)
    self.assertIn("kaboom", output)

  def test_plain_console_keeps_literal_brackets(self):
    buffer = io.StringIO()
    console = console_mod.UiConsole(
        theme=theme_mod.detect_theme(force_color=False),
        file=buffer,
        force_plain=True,
    )
    console.info("[DRY-RUN] simulating")
    self.assertIn("[DRY-RUN] simulating", buffer.getvalue())

  def test_panel_fallback_without_rich(self):
    buffer = io.StringIO()
    console = console_mod.UiConsole(
        theme=theme_mod.detect_theme(force_color=False),
        file=buffer,
        force_plain=True,
    )
    console.panel(["line one", "line two"], title="Review")
    output = buffer.getvalue()
    self.assertIn("Review", output)
    self.assertIn("line one", output)

  def test_width_has_sane_fallback(self):
    console = console_mod.UiConsole(file=io.StringIO(), force_plain=True)
    self.assertGreaterEqual(console.width, 40)
    self.assertGreaterEqual(console.height, 10)

  @requires_rich
  def test_rich_console_renders_markup(self):
    buffer = io.StringIO()
    console = console_mod.UiConsole(
        theme=theme_mod.detect_theme(force_color=False), file=buffer, width=60
    )
    self.assertTrue(console.is_rich)
    console.print("[bold]styled[/bold]")
    console.rule("section")
    output = buffer.getvalue()
    self.assertIn("styled", output)
    self.assertIn("section", output)
    self.assertEqual(console.width, 60)


def event(event_type, stage=None, message="", **payload):
  """Builds a campaign event for the dashboard tests."""
  return events_mod.CampaignEvent(
      type=event_type, stage=stage, message=message, payload=payload
  )


class DashboardModelTest(unittest.TestCase):
  """Folding the event stream into UI state."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov", sft_runs=3)
    self.state = make_state()
    self.model = dashboard_mod.DashboardModel(self.config, self.state)

  def views_by_stage(self):
    return {view.key: view for view in self.model.stage_views()}

  def test_stage_started_marks_running_and_tracks_current(self):
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_STARTED,
            stage="sft",
            message="starting sft",
            metric="eval/loss",
            max_runs=3,
        )
    )
    self.assertEqual(self.model.current_stage, "sft")
    self.assertEqual(self.views_by_stage()["sft"].status, "RUNNING")
    self.assertIsNotNone(self.model.active_parser())

  def test_stage_completed_clears_current_stage(self):
    self.model.on_event(
        event(events_mod.EventType.STAGE_STARTED, stage="sft", max_runs=3)
    )
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_COMPLETED,
            stage="sft",
            metric=0.31,
            model_repo_id="leobianco/npov_SFT",
        )
    )
    self.assertIsNone(self.model.current_stage)
    view = self.views_by_stage()["sft"]
    self.assertEqual(view.status, "COMPLETED")
    self.assertAlmostEqual(view.metric_value, 0.31)
    self.assertEqual(view.model_repo_id, "leobianco/npov_SFT")

  def test_stage_failed_records_error(self):
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_FAILED, stage="rm", message="boom: OOM"
        )
    )
    self.assertEqual(self.views_by_stage()["rm"].status, "FAILED")

  def test_skipped_stage_is_not_active(self):
    self.model.on_event(event(events_mod.EventType.STAGE_SKIPPED, stage="rm"))
    self.assertEqual(self.views_by_stage()["rm"].status, "SKIPPED")
    self.assertFalse(self.views_by_stage()["rm"].is_active)

  def test_trial_finished_updates_counter_and_metric(self):
    self.model.on_event(
        event(events_mod.EventType.STAGE_STARTED, stage="sft", max_runs=3)
    )
    self.model.on_event(
        event(
            events_mod.EventType.TRIAL_FINISHED,
            stage="sft",
            index=2,
            metric=0.5,
        )
    )
    view = self.views_by_stage()["sft"]
    self.assertEqual(view.trials_done, 2)
    self.assertAlmostEqual(view.metric_value, 0.5)

  def test_log_lines_drive_the_sweep_parser(self):
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_STARTED,
            stage="sft",
            metric="eval/loss",
            max_runs=3,
        )
    )
    for line in SweepProgressParserTest.AGENT_OUTPUT:
      self.model.on_event(
          event(events_mod.EventType.LOG, stage="sft", message=line)
      )
    view = self.views_by_stage()["sft"]
    self.assertGreaterEqual(view.trials_done, 1)
    self.assertIsNotNone(view.metric_value)

  def test_campaign_finished_sets_final_status(self):
    self.model.on_event(
        event(events_mod.EventType.CAMPAIGN_FINISHED, status="COMPLETED")
    )
    self.assertTrue(self.model.finished)
    self.assertEqual(self.model.final_status, "COMPLETED")

  def test_campaign_started_resets_the_clock(self):
    started = time.time() - 500
    self.model.on_event(
        events_mod.CampaignEvent(
            type=events_mod.EventType.CAMPAIGN_STARTED, timestamp=started
        )
    )
    self.assertGreater(self.model.elapsed, 400)

  def test_notices_are_bounded(self):
    for i in range(10):
      self.model.on_event(
          event(events_mod.EventType.NOTICE, message=f"notice {i}")
      )
    self.assertEqual(len(self.model.notices), 5)
    self.assertEqual(self.model.notices[-1], "notice 9")

  def test_pending_logs_are_drained_once(self):
    self.model.on_event(event(events_mod.EventType.LOG, message="hello"))
    self.assertEqual(len(self.model.drain_pending_logs()), 1)
    self.assertEqual(self.model.drain_pending_logs(), [])

  def test_drain_respects_limit(self):
    for i in range(5):
      self.model.on_event(event(events_mod.EventType.LOG, message=str(i)))
    self.assertEqual(len(self.model.drain_pending_logs(limit=2)), 2)
    self.assertEqual(len(self.model.drain_pending_logs()), 3)

  def test_milestone_mode_filters_chatter(self):
    self.model.log_level = dashboard_mod.LOG_MILESTONES
    self.model.on_event(event(events_mod.EventType.LOG, message="step 42/900"))
    self.model.on_event(
        event(events_mod.EventType.LOG, message="Model registered: x/y")
    )
    drained = [entry[2] for entry in self.model.drain_pending_logs()]
    self.assertEqual(drained, ["Model registered: x/y"])

  def test_log_off_still_surfaces_failures(self):
    self.model.log_level = dashboard_mod.LOG_OFF
    self.model.on_event(event(events_mod.EventType.LOG, message="noise"))
    self.model.on_event(
        event(events_mod.EventType.STAGE_FAILED, stage="sft", message="dead")
    )
    drained = [entry[2] for entry in self.model.drain_pending_logs()]
    self.assertEqual(drained, ["dead"])

  def test_history_is_kept_even_when_filtered(self):
    self.model.log_level = dashboard_mod.LOG_OFF
    self.model.on_event(event(events_mod.EventType.LOG, message="noise"))
    self.assertEqual(self.model.recent_logs()[-1][2], "noise")

  def test_log_buffer_is_capacity_bounded(self):
    model = dashboard_mod.DashboardModel(
        self.config, self.state, log_capacity=3
    )
    for i in range(10):
      model.on_event(event(events_mod.EventType.LOG, message=str(i)))
    self.assertEqual(
        [entry[2] for entry in model.recent_logs(99)], ["7", "8", "9"]
    )

  def test_cycle_log_level_wraps(self):
    self.assertEqual(self.model.cycle_log_level(), "milestones")
    self.assertEqual(self.model.cycle_log_level(), "off")
    self.assertEqual(self.model.cycle_log_level(), "all")
    self.assertEqual(self.model.log_level, dashboard_mod.LOG_ALL)

  def test_progress_summary_mentions_active_stage(self):
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_STARTED,
            stage="sft",
            metric="eval/loss",
            max_runs=3,
        )
    )
    self.model.on_event(
        event(
            events_mod.EventType.TRIAL_FINISHED,
            stage="sft",
            index=1,
            metric=0.4,
        )
    )
    summary = self.model.progress_summary()
    self.assertIn("stages 0/4", summary)
    self.assertIn("SFT Sweep", summary)
    self.assertIn("1/3", summary)
    self.assertIn("elapsed", summary)

  def test_progress_summary_counts_finished_stages(self):
    self.model.on_event(
        event(events_mod.EventType.STAGE_COMPLETED, stage="sft")
    )
    self.model.on_event(event(events_mod.EventType.STAGE_SKIPPED, stage="rm"))
    self.assertIn("stages 2/4", self.model.progress_summary())

  def test_concurrent_event_ingestion_is_consistent(self):
    def emit(start):
      for i in range(100):
        self.model.on_event(
            event(events_mod.EventType.LOG, message=f"{start}-{i}")
        )

    threads = [threading.Thread(target=emit, args=(n,)) for n in range(4)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join(timeout=5)
    self.assertEqual(len(self.model.drain_pending_logs(limit=1000)), 400)


class ResumedCounterTest(unittest.TestCase):
  """The launching pane must not undo the resume baseline.

  Reported symptom: a resumed sweep showed the recovered `06/10`, then as
  soon as the new agent finished a trial the counter "reset" and climbed
  from `01/10`, making it look like the sweep had restarted from scratch and
  was running extra trials. The dashboard's parser only sees the agent this
  process launched, and `live` overrides the persisted counter.
  """

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov", sft_runs=10)
    self.state = make_state()
    self.model = dashboard_mod.DashboardModel(self.config, self.state)

  def start(self, stage="sft", baseline=6, max_runs=10):
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_STARTED,
            stage=stage,
            metric="eval/loss",
            max_runs=max_runs,
            trials_baseline=baseline,
        )
    )

  def feed(self, line, stage="sft"):
    self.model.on_event(events_mod.CampaignEvent(
        type=events_mod.EventType.LOG, stage=stage, message=line, payload={}
    ))

  def trials_done(self, stage="sft"):
    return {v.key: v for v in self.model.stage_views()}[stage].trials_done

  def test_a_new_trial_adds_to_the_recovered_count(self):
    self.start(baseline=6)
    self.feed("wandb: Agent Starting Run: newrun01 with config:")
    self.feed(
        "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
        " finished run: newrun01"
    )
    self.assertEqual(self.trials_done(), 7)

  def test_the_counter_never_goes_below_the_baseline(self):
    self.start(baseline=6)
    seen = []
    for index in range(1, 5):
      self.feed(f"wandb: Agent Starting Run: run{index} with config:")
      seen.append(self.trials_done())
      self.feed(
          "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
          f" finished run: run{index}"
      )
      seen.append(self.trials_done())
    self.assertTrue(
        all(value >= 6 for value in seen), f"counter dipped below 6: {seen}"
    )
    # Six recovered plus the four this agent ran.
    self.assertEqual(self.trials_done(), 10)

  def test_a_fresh_stage_is_unaffected(self):
    self.start(baseline=0)
    self.feed("wandb: Agent Starting Run: run1 with config:")
    self.feed(
        "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
        " finished run: run1"
    )
    self.assertEqual(self.trials_done(), 1)

  def test_a_payload_without_a_baseline_still_works(self):
    # Older engines, and the plain renderer's synthetic events.
    self.model.on_event(
        event(events_mod.EventType.STAGE_STARTED, stage="sft", max_runs=10)
    )
    self.feed("wandb: Agent Starting Run: run1 with config:")
    self.feed(
        "2026-09-15 11:45:10,000 - wandb.wandb_agent - INFO - Cleaning up"
        " finished run: run1"
    )
    self.assertEqual(self.trials_done(), 1)


class HotkeyTest(unittest.TestCase):
  """Keyboard semantics, decoupled from any terminal."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov")
    self.controls = events_mod.ControlSignals()
    self.model = dashboard_mod.DashboardModel(
        self.config, make_state(), controls=self.controls
    )

  def press(self, key):
    return dashboard_mod.handle_key(key, self.model, self.controls)

  def test_p_toggles_pause_without_stopping(self):
    self.assertIn("Paused", self.press("p"))
    self.assertTrue(self.controls.is_paused)
    self.assertFalse(self.controls.stop_requested)
    self.assertIn("Resumed", self.press("p"))
    self.assertFalse(self.controls.is_paused)

  def test_a_requests_advance_only(self):
    self.press("a")
    self.assertTrue(self.controls.advance_requested)
    self.assertFalse(self.controls.stop_requested)

  def test_s_requests_graceful_stop(self):
    self.press("s")
    self.assertTrue(self.controls.stop_requested)
    self.assertFalse(self.controls.abort_requested)

  def test_x_aborts(self):
    self.press("x")
    self.assertTrue(self.controls.abort_requested)

  def test_l_cycles_verbosity(self):
    self.assertIn("milestones", self.press("l"))
    self.assertIn("off", self.press("l"))

  def test_question_mark_toggles_help_silently(self):
    self.assertIsNone(self.press("?"))
    self.assertTrue(self.model.show_help)
    self.press("?")
    self.assertFalse(self.model.show_help)

  def test_plus_and_minus_resize_the_log_window(self):
    self.press("+")
    self.press("+")
    self.assertEqual(self.model.log_window, 4)
    self.press("-")
    self.assertEqual(self.model.log_window, 2)

  def test_log_window_is_clamped(self):
    for _ in range(50):
      self.press("+")
    self.assertEqual(self.model.log_window, 20)
    for _ in range(50):
      self.press("-")
    self.assertEqual(self.model.log_window, 0)

  def test_q_detaches(self):
    self.assertEqual(self.press("q"), "detach")
    self.assertFalse(self.controls.stop_requested)

  def test_keys_are_case_insensitive(self):
    self.press("P")
    self.assertTrue(self.controls.is_paused)

  def test_unknown_and_empty_keys_are_ignored(self):
    self.assertIsNone(self.press("z"))
    self.assertIsNone(self.press(""))
    self.assertIsNone(self.press("\x1b"))
    self.assertEqual(
        self.controls.snapshot(),
        {"paused": False, "advance": False, "stop": False, "abort": False},
    )

  def test_escape_sequence_fragments_are_not_hotkeys(self):
    # Whole sequences, and the control bytes a terminal can emit, must never
    # reach a control signal - an Up arrow is not an "advance" request.
    for fragment in ("\x1b[A", "\x1bOP", "\x1b[<0;1;1M", "\x03", "\x7f"):
      self.assertIsNone(self.press(fragment), fragment)
    self.assertEqual(
        self.controls.snapshot(),
        {"paused": False, "advance": False, "stop": False, "abort": False},
    )


class KeyReaderTest(unittest.TestCase):
  """The key reader must be inert whenever stdin is not a terminal."""

  def test_disabled_for_non_tty(self):
    reader = dashboard_mod.KeyReader(lambda _: None, stream=io.StringIO())
    self.assertFalse(reader.enabled)
    self.assertFalse(reader.start())
    reader.stop()  # Must not raise.

  def test_disabled_when_isatty_raises(self):
    class Broken(io.StringIO):

      def isatty(self):
        raise OSError("closed")

    reader = dashboard_mod.KeyReader(lambda _: None, stream=Broken())
    self.assertFalse(reader.enabled)

  def test_context_manager_is_safe_without_tty(self):
    with dashboard_mod.KeyReader(lambda _: None, stream=io.StringIO()) as r:
      self.assertFalse(r.enabled)

  def test_start_fails_gracefully_on_fileno_error(self):
    class FakeTty(io.StringIO):

      def isatty(self):
        return True

      def fileno(self):
        raise OSError("no fd")

    reader = dashboard_mod.KeyReader(lambda _: None, stream=FakeTty())
    self.assertTrue(reader.enabled)
    self.assertFalse(reader.start())


class KeyReaderEscapeSequenceTest(unittest.TestCase):
  """Arrow keys must never be mistaken for hotkeys.

  ``ESC [ A`` (Up arrow) used to be delivered byte by byte, so its final
  ``A`` was lowercased into the ``[a]`` advance hotkey: a single arrow press
  sealed a running sweep and killed the campaign during materialization.
  A real pty is used here because the reader only arms itself on a TTY.
  """

  def setUp(self):
    self.master_fd, slave_fd = os.openpty()
    self.addCleanup(os.close, self.master_fd)
    self.stream = os.fdopen(slave_fd, "r", buffering=1)
    self.addCleanup(self.stream.close)
    self.seen = []
    self.reader = dashboard_mod.KeyReader(
        self.seen.append, stream=self.stream
    )
    self.assertTrue(self.reader.start())
    self.addCleanup(self.reader.stop)

  def _send(self, payload, expected=0):
    os.write(self.master_fd, payload.encode())
    deadline = time.time() + 2.0
    while time.time() < deadline and len(self.seen) < expected:
      time.sleep(0.01)
    # Give a rejected sequence the same grace period before asserting.
    if not expected:
      time.sleep(0.2)
    return list(self.seen)

  def test_up_arrow_is_not_the_advance_hotkey(self):
    self.assertEqual(self._send("\x1b[A"), [])

  def test_all_arrows_are_ignored(self):
    self.assertEqual(self._send("\x1b[A\x1b[B\x1b[C\x1b[D"), [])

  def test_function_and_home_keys_are_ignored(self):
    # F1 (SS3), Home/End/Delete (CSI with parameters).
    self.assertEqual(self._send("\x1bOP\x1b[1~\x1b[4~\x1b[3~"), [])

  def test_sgr_mouse_report_is_ignored(self):
    self.assertEqual(self._send("\x1b[<64;10;5M"), [])

  def test_control_bytes_are_ignored(self):
    self.assertEqual(self._send("\r\n\t\x03"), [])

  def test_typed_keys_still_reach_the_callback(self):
    self.assertEqual(self._send("a", expected=1), ["a"])

  def test_typed_key_after_an_arrow_still_works(self):
    self.assertEqual(self._send("\x1b[Aq", expected=1), ["q"])



class LiveDashboardTest(unittest.TestCase):
  """Log formatting and the plain (non-rich) render loop."""

  def setUp(self):
    self.config = CampaignConfig.create_default(task_name="npov", sft_runs=2)
    self.model = dashboard_mod.DashboardModel(self.config, make_state())
    self.buffer = io.StringIO()
    self.console = console_mod.UiConsole(
        theme=theme_mod.detect_theme(force_color=False),
        file=self.buffer,
        force_plain=True,
    )
    self.dashboard = dashboard_mod.LiveDashboard(
        self.model, console=self.console
    )

  def test_format_log_line_has_clock_and_stage_tag(self):
    line = self.dashboard.format_log_line(0.0, "sft", "training started")
    self.assertRegex(line, r"^\d{2}:\d{2}:\d{2} ")
    self.assertIn("SFT ", line)
    self.assertIn("training started", line)

  def test_format_log_line_tags_unknown_stage_as_run(self):
    self.assertIn("RUN ", self.dashboard.format_log_line(0.0, None, "hi"))

  def test_format_log_line_preserves_literal_brackets(self):
    line = self.dashboard.format_log_line(0.0, "sft", "[DRY-RUN] Trial 1/2")
    self.assertIn("[DRY-RUN] Trial 1/2", line)

  def test_flush_logs_returns_count_and_empties_buffer(self):
    self.model.on_event(event(events_mod.EventType.LOG, message="one"))
    self.model.on_event(event(events_mod.EventType.LOG, message="two"))
    self.assertEqual(self.dashboard.flush_logs(), 2)
    self.assertEqual(self.dashboard.flush_logs(), 0)
    self.assertIn("one", self.buffer.getvalue())

  def test_run_plain_terminates_and_emits_progress(self):
    self.model.on_event(event(events_mod.EventType.LOG, message="hello"))
    ticks = {"n": 0}

    def is_done():
      ticks["n"] += 1
      return ticks["n"] > 2

    self.dashboard.run_plain(is_done, poll_seconds=0.0)
    output = self.buffer.getvalue()
    self.assertIn("hello", output)
    self.assertIn("[progress]", output)

  def test_run_falls_back_to_plain_without_rich(self):
    with mock.patch.object(self.dashboard, "run_plain") as fake:
      self.dashboard.run(is_done=lambda: True)
    fake.assert_called_once()

  @requires_rich
  def test_render_includes_every_block(self):
    self.model.on_event(
        event(
            events_mod.EventType.STAGE_STARTED,
            stage="sft",
            metric="eval/loss",
            max_runs=2,
        )
    )
    for line in SweepProgressParserTest.AGENT_OUTPUT:
      self.model.on_event(
          event(events_mod.EventType.LOG, stage="sft", message=line)
      )
    self.model.show_help = True
    from rich.console import Console  # pylint: disable=g-import-not-at-top

    rich_console = console_mod.UiConsole(
        theme=theme_mod.detect_theme(force_color=False),
        file=io.StringIO(),
        width=120,
    )
    dash = dashboard_mod.LiveDashboard(self.model, console=rich_console)
    out = io.StringIO()
    Console(file=out, width=120, no_color=True).print(dash.render())
    text = out.getvalue()
    self.assertIn("SFT Sweep", text)
    self.assertIn("Live leaderboard", text)
    self.assertIn("[p]", text)

  @requires_rich
  def test_render_never_overflows_the_pane(self):
    from rich.console import Console  # pylint: disable=g-import-not-at-top

    for width in (60, 100, 200):
      rich_console = console_mod.UiConsole(
          theme=theme_mod.detect_theme(force_color=False),
          file=io.StringIO(),
          width=width,
      )
      dash = dashboard_mod.LiveDashboard(self.model, console=rich_console)
      out = io.StringIO()
      Console(file=out, width=width, no_color=True).print(dash.render())
      for line in out.getvalue().splitlines():
        self.assertLessEqual(len(line), width)


if __name__ == "__main__":
  unittest.main()

