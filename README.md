# Hallucination Reduction with PERL and Synthetic Data Generation

*TO DO:* write a proper README.

## Usage

Each shell script runs the corresponding python script. See the script's docstring for more information.

We perform our experiments in a multi-GPU setting. More precisely, we use 8 x L4 GPUs. For an efficient use of GPU memory, we employ pipeline parallelism, specifically ZeRO Phase-3 [(link to paper)](https://arxiv.org/abs/1910.02054). To do so, we use Hugging Face's Accelerate library integration of Microsoft's DeepSpeed. The configuration used for our experiments is as follows (you should run `accelerate config` to set up your environment, see [Accelerate's documentation](https://huggingface.co/docs/transformers/en/deepspeed) for more details).

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero3_save_16bit_model: false
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Please install the necessary Python header files by installing the `python-dev` package.

Please install the CUDA drivers following the [Google Cloud CUDA Driver Installation Guide](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu).
