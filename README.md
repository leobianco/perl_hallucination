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

## Base models, GPUs and DeepSpeed

The original experiments ran on 8 x L4 GPUs. For an efficient use of GPU
memory we employ ZeRO
[(link to paper)](https://arxiv.org/abs/1910.02054) through Hugging Face's
Accelerate integration of Microsoft's DeepSpeed. The configurations live in
`scripts/` (run `accelerate config` to set up your own environment, see
[Accelerate's documentation](https://huggingface.co/docs/transformers/en/deepspeed)).
All three ship with `num_processes: 2`, matching the 2 x A100-80GB box the
campaigns currently run on.

| Profile | File | Used for |
|---|---|---|
| `zero2` | `scripts/deepspeed_config.yaml` | everything below 6B, and SFT/RM at any size below 30B |
| `zero3` | `scripts/deepspeed_config_zero3.yaml` | the PE-RL stage from 6B up |
| `zero3_offload` | `scripts/deepspeed_config_zero3_offload.yaml` | from 30B up, or forced by hand |

### Running a campaign on another base model

```bash
python3 scripts/run_campaign.py run --task npov \
    --base-model "mistralai/Mistral-7B-Instruct-v0.3"
```

`--base-model` reaches the SFT, RM and PE-RL sweeps, the winners' retraining
and the evaluation, overriding the `--model_repo_id` pinned in every
`scripts/sweep_*.yaml`. The wizard (`run_campaign.py wizard`) asks for it too
and shows the resulting launcher in its review block. Nothing else needs
editing; `src/model_compat.py` handles the family differences (chat template,
turn terminator, missing pad token).

**What a large model changes automatically.** `src/orchestrator/accel.py`
reads the parameter count off the checkpoint's name and, at or above 6B:

* the PE-RL stage moves to ZeRO Stage 3, because it is the only stage holding
  two models at once - the policy and the reward model, ~29 GB of bf16 weights
  at 7B - before activations and the rollout KV cache;
* every training stage gets `--gradient_checkpointing True`;
* the PE-RL micro-batch is halved and gradient accumulation doubled, so the
  effective batch, and therefore the optimisation problem the sweep searches,
  is unchanged.

Below 6B nothing changes at all, so the existing 4B results stay comparable.
Override the profile with `--deepspeed-profile {auto,zero2,zero3,zero3_offload}`
or a path; override the batch geometry with
`PERL_PER_DEVICE_TRAIN_BATCH_SIZE` / `PERL_GRADIENT_ACCUMULATION_STEPS` (an
explicit environment value always beats the automatic one). Hand-run scripts
do not consult the orchestrator, so tell them directly:

```bash
DEEPSPEED_CONFIG=scripts/deepspeed_config_zero3.yaml \
MODEL_REPO_ID=mistralai/Mistral-7B-Instruct-v0.3 ./scripts/perl.sh npov
```

> **Cost of Stage 3.** Parameters are all-gathered on every forward, and the
> reward model is gathered once per scored batch (TRL's
> `ds3_gather_for_generation` only covers the policy, so `reward_fn` in
> `src/pipelines.py` does it explicitly). Budget 20-30% more wall clock for
> the PE-RL stage than the estimate the wizard prints.


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