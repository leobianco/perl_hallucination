# Hallucination Reduction with PERL and Synthetic Data Generation

This project studies the reduction of hallucinations via RLAIF with *synthetic* data. More precisely, we use Parameter-Efficient Reinforcement Learning [PE-RL](https://arxiv.org/abs/2403.10704).

## Automated Campaign Orchestrator (Auto-PERL)

Rather than manually managing and chaining individual training scripts, you can run an end-to-end automated scientific campaign (SFT sweep $\to$ RM sweep $\to$ PE-RL sweep $\to$ Evaluation $\to$ Markdown Report) using the Auto-PERL orchestrator:

```bash
# 1. Pre-flight environment check (CUDA, W&B, HF tokens, disk space)
python3 scripts/run_campaign.py doctor --task npov

# 2. Guided interactive setup wizard
python3 scripts/run_campaign.py wizard

# 3. Launch with a budget preset (smoke, quick, standard, thorough)
python3 scripts/run_campaign.py run --task npov --preset standard

# 4. Monitor from a second tmux pane
python3 scripts/run_campaign.py status --task npov --watch

# 5. Seamlessly resume after VM preemption or disconnect
python3 scripts/run_campaign.py resume --task npov
```

### Resuming vs. restarting (read this if your config pins `name:`)

A campaign's `name` determines its state file
(`./checkpoints/<task>/<name>_state.json`). If your YAML pins `name:`, every
`run` targets the same campaign, and **`run` will not guess** whether you want
to continue it or start it over — guessing wrong either re-burns hours of GPU
time or throws away a finished campaign. Say which:

```bash
# Continue the existing campaign (same as `resume`).
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --resume

# Start over. The old state file is archived, never deleted, under
# ./checkpoints/bosch/archive/.
python3 scripts/run_campaign.py run --task bosch --config configs/campaign_bosch.yaml --fresh
```

Without a flag, an interactive terminal prompts `[r]esume / [f]resh / [a]bort`;
a non-interactive one (piped output, `nohup`, CI) refuses with exit code `2`.
`--yes` does *not* answer this question — it only skips the launch
confirmation. Omit `name:` from your config and every launch is a new,
timestamped campaign, so none of this applies.

For complete documentation on the orchestrator, interactive dashboard hotkeys, declarative YAML configs, and remote VM/tmux setups, see [**`src/orchestrator/README.md`**](src/orchestrator/README.md).

---

## Manual Script Execution

If running scripts individually, here is a brief description about the usage of each script, roughly in the order that they should be executed.
Substitute `TASK` by `npov`, `bosch`, `ragtruth-qa`, or `ragtruth-summarization` (or `ragtruth`, which defaults to `ragtruth-qa`) to select the correct dataset. 
Every `.sh` script in the `/scripts` folder runs the corresponding Python entrypoint located under `src/` (for example `src/writer_sft.py`).
**Run the scripts from the project folder (not from within the `scripts` folder) so paths resolve correctly.**

Please, set the Gemini API key in order to use the LLM-generated synthetic data, as well as the autorater:
```
export GEMINI_API_KEY="your-key-here"
```

From the project folder, run
* `./scripts/data_processing.sh (TASK)`: processes the raw dataset and saves resulting datasets to the HuggingFace Hub. This includes creating prompts to be passed to the LLMs later, putting data in the right format, creating synthetic hallucinations, creating data for evaluation.
* `writer_sft.sh (TASK)`: LoRA-SFT the model indicated as a writer.
* `reward_model.sh (TASK)`: trains the reward model. Set one of the `ORGANIC`, `SYN_HALL_LLM`, or `SYN_HALL_STRUCT` parameters to true to choose what type of hallucinations to train on.
* `perl.sh (TASK)`: runs the PERL loop with the indicated reward model and SFT model as reference. Since these jobs are longer, there is a `SHUTDOWN` parameter that allows the VM to be automatically shut down once the job finishes.
* `evaluator.sh (TASK) (MODE)`:
    * When `(MODE)` is set to `generate`, the model indicated by the `RUN_IDENTIFIER` parameter is used as a writer and generates completions for the prompts in `DATASET_PROMPTS`. This generation is done using vLLM.
    * When `(MODE)` is set to `score`, the `EVALUATOR_MODEL` scores the generations of the previous step, and a rate of hallucination is calculated using `THRESHOLD`. The evaluator uses `EVALUATOR_NUM_FEWSHOT` examples taken from `DATASET_LABELS` to help it classify the samples.
    * When `(MODE)` is set to `autoratereval`, the quality of the evaluator itself is evaluated. We ask for it to score the samples in `DATASET_LABELS`, and we return the best threshold along with the associated metrics.

Notice that you can chain commands, *e.g.* `./scripts/evaluator.sh npov generate ; ./scripts/evaluator.sh npov score`.

### Evaluating the Last Checkpoint vs. Best Checkpoint

By default, training scripts (`perl.sh`, `writer_sft.sh`, `reward_model.sh`, `scope_dpo.sh`, `ssfo_dpo.sh`) save checkpoints as follows:
1. **Hugging Face Hub (`<user>/<run_name>`)**: Stores only the **best model** (the checkpoint that achieved the highest validation reward / metric during training).
2. **Local Disk (`./checkpoints/<task>/<method>/<user>/<run_name>/`)**: Stores:
   - The **best model** at the root of the directory and in `checkpoint-<BEST_STEP>/`.
   - The **final step checkpoint** (adapters from the very last training step) in `checkpoint-<LAST_STEP>/`.
   - *(Note for PE-RL runs)*: Each checkpoint directory also contains a `ref/` subfolder holding TRL's frozen reference policy LoRA adapter used for KL computation; `src.evaluator` automatically ignores `ref/` and loads the trained policy adapter.

To evaluate the **best model** (default):
- Set `RUN_IDENTIFIER="<user>/<run_name>"` (loads from Hugging Face Hub) or `RUN_IDENTIFIER="./checkpoints/<task>/perl/<user>/<run_name>"` (loads from local root).

To evaluate the **last checkpoint** (`checkpoint-<LAST_STEP>`) instead of the best:
- Point `RUN_IDENTIFIER` to the specific local `checkpoint-<LAST_STEP>` directory.
- Explicitly set `DATASET_WITH_COMPLETIONS` to a distinct short Hugging Face dataset name (since auto-generated dataset names from long local filesystem paths are truncated at 96 characters on Hugging Face Hub and could collide with evaluations of other checkpoints from the same run):

```bash
RUN_IDENTIFIER="./checkpoints/npov/perl/leobianco/<run_name>/checkpoint-220" \
DATASET_WITH_COMPLETIONS="leobianco/npov_eval_last_ckpt_220" \
./scripts/evaluator.sh npov generate

RUN_IDENTIFIER="./checkpoints/npov/perl/leobianco/<run_name>/checkpoint-220" \
DATASET_WITH_COMPLETIONS="leobianco/npov_eval_last_ckpt_220" \
./scripts/evaluator.sh npov score
```

We perform our experiments in a multi-GPU setting. More precisely, we use 8 x L4 GPUs. For an efficient use of GPU memory, we employ pipeline parallelism, specifically ZeRO Phase-3 [(link to paper)](https://arxiv.org/abs/1910.02054). To do so, we use Hugging Face's Accelerate library integration of Microsoft's DeepSpeed. The configuration used for our experiments is stored in `scripts/deepspeed_config.yaml` (you should run `accelerate config` to set up your own environment, see [Accelerate's documentation](https://huggingface.co/docs/transformers/en/deepspeed) for more details).

## Other notes

The experiments were run using Python 3.11.

Please also install the necessary Python header files by installing the `python-dev` package.

Please also install the CUDA drivers following the [Google Cloud CUDA Driver Installation Guide](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu).

Install Python package requirements by first compiling `requirements.in` using `pip-tools`:
```
pip-compile requirements.in -o requirements.txt
```
then install them:
```
pip install -r requirements.txt
```

*Note:* `pip-compile` can be slow. For this reason we recommend using [uv](https://docs.astral.sh/uv/), a Python package manager written in Rust that is much faster. After installing it, run
```
uv pip compile requirements.in > requirements.txt
pip install -r requirements.txt
```