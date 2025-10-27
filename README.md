# Hallucination Reduction with PERL and Synthetic Data Generation

This project studies the reduction of hallucinations via RLAIF with *synthetic* data. More precisely, we use Parameter-Efficient Reinforcement Learning [PE-RL](https://arxiv.org/abs/2403.10704).

## Usage

Here is a brief description about the usage of each script, roughly in the order that they should be executed.
Substitute `TASK` by `npov`, `bosch`, or `ragtruth` to select the correct dataset. 
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

Notice that you can chain commands, *e.g.* `.scripts/evaluator.sh npov generate ; .scripts/evaluator.sh npov score`.

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