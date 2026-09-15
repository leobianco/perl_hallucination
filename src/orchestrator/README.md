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

### 6. Resuming an Interrupted Run
```bash
python3 scripts/run_campaign.py resume --task npov
```
Prints a resume plan (what will be skipped, what will run) before doing anything.

### 7. Monitoring from a second tmux pane
```bash
python3 scripts/run_campaign.py status --task npov --watch
python3 scripts/run_campaign.py status --task npov --json | jq .stages_completed
python3 scripts/run_campaign.py list
```

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

## 📝 Declarative Configuration Example (`configs/campaign.yaml`)

```yaml
campaign:
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
