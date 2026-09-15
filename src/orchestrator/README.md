# Auto-PERL: Automated Scientific Campaign Orchestrator

A modular, zero-lockin research orchestrator for Parameter-Efficient Reinforcement Learning (PE-RL) experiments on Gemma and other language models.

---

## 🚀 Key Features

* **End-to-End Automation**: Automatically chains **SFT Sweep** $\to$ **Reward Model Sweep** $\to$ **PE-RL (RLOO) Sweep** $\to$ **Final Evaluation** with Gemini 2.5 Flash autorater.
* **Automatic Sweep Bounds (`max_runs`)**: Enforces strict run budgets per stage (defaults: SFT=30, RM=30, PE-RL=10) and automatically seals sweeps on W&B (`STOPPED`), preventing runaway cloud jobs.
* **Dynamic Checkpoint Wiring**: Captures winning model checkpoints from SFT and RM sweeps and dynamically injects them into downstream PE-RL sweeps without manual copy-pasting.
* **Zero-Lockin & Non-Invasive**: Resides entirely in `src/orchestrator/`. Does not modify a single line of your core training scripts (`src/writer_sft.py`, `src/reward_model.py`, `src/perl.py`, `src/evaluator.py`). If removed, your existing codebase remains 100% intact.
* **Crash-Resilient & Resumable**: Atomic state persistence (`campaign_state.json`). If interrupted or preempted on a cloud VM, resume seamlessly with `resume`. The full configuration is persisted alongside the state, so a resumed campaign reproduces the original budgets and stage selection.
* **Pinned Mission Control dashboard**: A `rich` live block pinned to the bottom of the pane (header, DAG, live sweep leaderboard, hotkeys) with logs streaming *above* it. It deliberately does **not** use a full-screen alternate-buffer TUI, so `tmux` scrollback, copy-mode and `Ctrl-b [` keep working, re-attaching or resizing simply re-flows the layout, and piping to a file produces clean plain text.
* **Degrades gracefully**: Works with no color, ASCII-only glyphs, narrow panes, no TTY, and even before `pip install -r requirements.txt` has finished (`rich`/`questionary` are optional at import time).
* **Pre-flight `doctor`**: Checks packages, W&B/HF credentials, sweep YAMLs, writable dirs, disk space, GPU and tmux *before* you commit to a multi-hour run.
* **Automated Scientific Reports**: Generates publication-ready Markdown summaries with performance tables, metric deltas, and interactive W&B reports.

---

## 📦 Architecture

```
src/orchestrator/
├── __init__.py               # Package public API
├── config.py                 # Dataclasses & YAML serialization (CampaignConfig, StageConfig)
├── state.py                  # Atomic state persistence & crash resumption (CampaignState)
├── sweep_controller.py       # W&B Sweep registration, agent controller, and best-run query
├── model_manager.py          # Model materialization, local retention, and HF Hub publishing
├── reporter.py               # Markdown report generator & Rich console scorecards
├── engine.py                 # Linear DAG orchestrator (CampaignEngine), event/control aware
├── cli/
│   ├── __init__.py
│   ├── app.py                # argparse CLI: run / wizard / status / resume / report / list / doctor
│   ├── theme.py              # Terminal capability detection, glyphs, colors, formatting
│   ├── events.py             # Typed event bus, control signals, sweep log parser
│   ├── renderables.py        # DAG rows, leaderboard, status table, banners (rich + plain)
│   ├── console.py            # UiConsole: one markup dialect, rich or plain
│   ├── dashboard.py          # Pinned live dashboard, hotkeys, final scorecard
│   ├── wizard.py             # Interactive setup wizard (questionary, with a plain fallback)
│   └── tui.py                # Backwards-compatible shim over the dashboard
└── tests/
    ├── test_orchestrator.py  # Config/state/engine unit & dry-run integration tests
    ├── test_cli_ui.py        # Theme, events, renderables, console, dashboard
    ├── test_cli_wizard.py    # Wizard flows, validators, prompters
    └── test_cli_app.py       # Argument parsing, commands, diagnostics, exit codes
```

---

## ⚡ Quickstart

### 0. Pre-flight check (do this first)
```bash
python3 scripts/run_campaign.py doctor --task npov
```

### 1. Zero-Boilerplate Campaign (Runs full pipeline with defaults)
```bash
python3 scripts/run_campaign.py run --task npov
```
Runs SFT (30 trials) $\to$ RM (30 trials) $\to$ PE-RL (10 trials) $\to$ Evaluation $\to$ Report.

### 2. Guided Interactive Setup Wizard
```bash
python3 scripts/run_campaign.py wizard
```
Prompts for task, stages, budget preset and upstream checkpoints, then shows a
review panel with a runtime estimate and the equivalent non-interactive command.

### 3. Budget presets (or explicit overrides)
```bash
python3 scripts/run_campaign.py run --task npov --preset quick
python3 scripts/run_campaign.py run --task npov --sft-runs 10 --rm-runs 10 --perl-runs 5
```

| Preset | Trials (SFT/RM/PE-RL) | Eval samples | Intent |
| :--- | :--- | :--- | :--- |
| `smoke` | 1 / 1 / 1 | 20 | Wiring check (minutes) |
| `quick` | 5 / 5 / 3 | 200 | Fast signal (a few hours) |
| `standard` | 30 / 30 / 10 | 1000 | Project default |
| `thorough` | 60 / 60 / 20 | 2000 | Publication run |

### 4. Running Sub-Pipelines (Plug in Existing Checkpoints)
If SFT and RM were already trained, run only PE-RL and Evaluation by passing the checkpoints directly:
```bash
python3 scripts/run_campaign.py run --task npov \
  --stages perl,eval \
  --sft-model "leobianco/npov_SFT_gemma-4-E2B-it_..." \
  --reward-model "leobianco/npov_RM_gemma-3-1b-it_..."
```
Omitting a required checkpoint is caught before launch, not three hours in.

### 5. Pre-flight Dry-Run (Verify wiring without GPU allocation)
```bash
python3 scripts/run_campaign.py run --task npov --preset smoke --dry-run
```

### 6. Resuming vs. Restarting a Campaign

A campaign is identified by its **name**, and its name determines its **state
file**:

```
./checkpoints/<task>/<name>_state.json
```

If you do not set `name:` in your YAML, one is generated from the clock
(`bosch_campaign_2609151127`), so every launch is a brand-new campaign and
this section never concerns you. **If you *do* pin `name:` in your config,
every `run` points at the same state file** — and the orchestrator then has to
know whether you meant "continue that campaign" or "start it over".

It will not guess. The two answers are expensive in opposite directions:

| You meant | If the orchestrator guesses wrong |
|---|---|
| **Continue** | It re-runs finished stages and re-burns hours of GPU time. |
| **Start over** | It attaches to stale sweep ids and silently discards a campaign's results. |

So `run` behaves as follows:

| Situation | Behaviour |
|---|---|
| No state file exists | Starts a new campaign. Normal case, nothing to decide. |
| State file exists, `--resume` given | Continues it. Prints what it adopted. |
| State file exists, `--fresh` given | Archives the old state, starts over. |
| State file exists, no flag, **interactive TTY** | Prompts `[r]esume / [f]resh / [a]bort`. |
| State file exists, no flag, **no TTY** (piped, `nohup`, CI) | **Refuses**, exit code `2`, and tells you the three ways out. |
| No state file, `--resume` given | **Refuses**, exit code `3` (`Nothing to resume`). |

```bash
# Continue where you left off.
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --resume

# Throw the old attempt away and start clean.
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --fresh

# Equivalent to --resume, and the preferred spelling when you are only resuming.
python3 scripts/run_campaign.py resume --task bosch
```

`--fresh` and `--resume` are mutually exclusive; argparse rejects passing both.

> **`--yes` does not answer this question.** `--yes` only skips the *launch*
> confirmation. With an existing state file and no `--resume`/`--fresh`, the
> run is still refused — `--yes` must never be able to destroy a campaign.

#### `--fresh` archives, it never deletes

The old state file is **moved**, not removed:

```
./checkpoints/<task>/archive/<name>_state_<YYYYmmddHHMMSS>.json
```

It is the only record of which sweeps a campaign registered and which
checkpoints it pushed, so it is worth keeping. `status`, `list` and `resume`
glob one directory level deep, so archived campaigns disappear from those
listings but remain on disk for you to inspect or restore by moving the file
back.

#### Recipe: I deleted the W&B sweep from the web UI

You do not need `--fresh` for this. On resume, the orchestrator probes each
recorded sweep id before reusing it:

* **Sweep is gone (404)** → the stale pointer is cleared and a new sweep is
  registered automatically. Completed stages are still skipped.
* **W&B is unreachable** (503, network drop) → the sweep id is *kept*, because
  abandoning a live sweep over a transient error would throw away paid-for
  trials.

Use `--fresh` when you want to discard the campaign's *results*, not merely
its sweeps.

### 7. Monitoring from a second tmux pane
```bash
python3 scripts/run_campaign.py status --task npov --watch
python3 scripts/run_campaign.py status --task npov --json | jq .stages_completed
python3 scripts/run_campaign.py list
```

The watcher is a **separate process**. It cannot see the launching pane's
in-memory log parser, so everything it shows is read from the state file,
which the running campaign mirrors its progress into:

| Shown in the watch pane | Refreshed |
|---|---|
| Stage statuses, elapsed times, model repo ids | On every stage transition |
| `NN/total` trial counter | The moment a trial finishes |
| Best metric so far | At most every 5s while a trial runs |
| Final eval metrics | When the eval stage completes |

The per-trial leaderboard is **not** mirrored - it stays exclusive to the
launching pane's dashboard, because writing every trial's parameters to disk
on every refresh is not worth the I/O.

Three behaviours worth knowing:

* **You can start the watcher first.** With `--watch` it waits for a campaign
  to appear rather than exiting, so the natural `split-window` → `watch` →
  `run` order works.
* **It follows the newest campaign** for the task, re-resolved on every
  refresh. Pass `--state-file` to pin it to one campaign instead.
* **It never dies on a transient read.** A campaign archived by
  `run --fresh`, or a half-written file, produces a message and a retry on
  the next refresh - not a traceback in a pane you stopped looking at.

> [!NOTE]
> On resume the counter continues from where the previous attempt stopped
> rather than restarting at zero, because a new `wandb agent` counts only its
> own trials while the state file tracks the whole stage.

### 8. Headless Mode (for `nohup` or logging to file)
```bash
python3 scripts/run_campaign.py run --task npov --no-tui > campaign.log 2>&1 &
```

### Appearance flags
`--no-color` (also honors `NO_COLOR`), `--ascii` (also `PERL_ASCII=1`), `--plain`
(no live block at all), `--debug` (full tracebacks). They work on either side of
the subcommand.

---

## 🎛️ Interactive Hotkey Controls (Inside `tmux`)

While the live dashboard is attached, single keystrokes (no Enter needed) control the run.
Hotkeys are automatically disabled when stdin is not a TTY, and the terminal is
always restored on exit.

| Hotkey | Action | Description |
| :---: | :--- | :--- |
| **`[a]`** | **Advance with Best** | Seals the active sweep, selects the current top run, and promotes it to the next stage. |
| **`[p]`** | **Pause / Resume** | Parks the campaign before the next stage. The campaign stays alive — it does **not** stop. |
| **`[s]`** | **Stop** | Finishes the current stage, then ends the campaign gracefully and writes the report. |
| **`[x]`** | **Abort** | Same as stop, but skips report generation. |
| **`[l]`** | **Log verbosity** | Cycles all → milestones → off. |
| **`[+]` / `[-]`** | **Resize** | Grows/shrinks the live leaderboard. |
| **`[?]`** | **Help** | Toggles the in-place hotkey cheat-sheet. |
| **`[q]`** | **Detach** | Detaches the dashboard; the campaign keeps running headless. |

`Ctrl-C` once requests a graceful stop; twice aborts.

---

## 🖥️ Running in `tmux` on Cloud / GCP VMs

When executing long, multi-stage campaigns on a remote Google Cloud Compute Engine VM:

### 1. Launch Inside `tmux` with UTF-8 Support
To ensure proper rendering of box-drawing glyphs and borders:
```bash
tmux -u new -s auto-perl
```
Ensure your shell locale supports UTF-8:
```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

### 2. SIGHUP Immunity & Disconnect Protection
The orchestrator explicitly ignores `SIGHUP` (`_survive_terminal_hangup`), so your campaign will continue running even if an SSH connection drops. You can safely detach anytime (`Ctrl-B, d`) and reattach (`tmux attach -t auto-perl`).

### 3. Recommended 2-Pane Workflow
Split your tmux window (`Ctrl-B, %`):
* **Pane 1 (Runner)**:
  ```bash
  python3 scripts/run_campaign.py run --task npov --preset standard
  ```
* **Pane 2 (Monitor)**:
  ```bash
  python3 scripts/run_campaign.py status --task npov --watch
  ```
  `--watch` polls `state.json` every 5 seconds without touching GPU resources or interfering with the runner.

### 4. Low-Bandwidth & Terminal Fallbacks
* If your terminal font garbles box glyphs, force ASCII rendering:
  ```bash
  python3 scripts/run_campaign.py run --task npov --preset standard --ascii
  ```
* For slow SSH connections or plain streaming logs without full-screen redraws:
  ```bash
  python3 scripts/run_campaign.py run --task npov --preset standard --plain
  # or --no-tui
  ```

---

## 📝 Declarative Configuration Example (`configs/campaign.yaml`)

```yaml
campaign:
  # Pinning a name pins the state file to
  # ./checkpoints/npov/npov_production_v1_state.json, so re-running this
  # config requires --resume or --fresh. See "Resuming vs. Restarting"
  # above. Omit `name` to get a timestamped, always-new campaign.
  name: "npov_production_v1"
  task: "npov"
  seed: 130104
  user: "leobianco"
  project: "new_perl"
  base_model: "google/gemma-4-E2B-it"

stages: [sft, rm, perl, eval]

sft_stage:
  enabled: true
  sweep_config: "scripts/sweep_sft.yaml"
  max_runs: 30
  metric: "eval/loss"
  goal: "minimize"

rm_stage:
  enabled: true
  sweep_config: "scripts/sweep_rm.yaml"
  max_runs: 30
  metric: "eval/roc_auc"
  goal: "maximize"

perl_stage:
  enabled: true
  sweep_config: "scripts/sweep_perl.yaml"
  max_runs: 10
  metric: "rewards/reward_fn/mean"
  goal: "maximize"
  sft_model_path: "auto"
  reward_model_path: "auto"

eval_stage:
  enabled: true
  evaluator_model: "gemini-2.5-flash"
  max_eval_samples: 1000

reporting:
  generate_markdown: true
  publish_wandb_report: true
```

---

## 🗑️ Safe Removal (Zero-Lockin Guarantee)

If you ever decide to remove Auto-PERL and return to running manual bash scripts, simply delete the orchestrator files:
```bash
rm -rf src/orchestrator scripts/run_campaign.py
```
Your repository will be in its exact prior state, with all original models, pipelines, and scripts functioning without issue.
