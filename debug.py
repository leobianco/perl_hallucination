"""This script serves to debug the model and tokenizer loading, as well as the generation process, interactively."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from data import npov_writer_prompt

# Load data
data = load_dataset("leobianco/npov_rm_processed", split="test")
data = data.map(npov_writer_prompt)

# # Load model + tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "google/gemma-2-2b-it",
#     padding_side="left",
# )

# model = AutoModelForCausalLM.from_pretrained(
#     "checkpoints/npov/writer_sft/leobianco/npov_SFT_seed_130104_epochs_60_lr_3e-4_lora_8_fewshot_1/checkpoint-480",
#     device_map="auto",
#     attn_implementation="eager",
#     torch_dtype=torch.bfloat16,
# )
# model.eval()

# tok_prompt = tokenizer(data[0]["prompt"], return_tensors="pt").to(model.device)

# # Config
# generation_config = GenerationConfig(
#     max_new_tokens=160,
#     temperature=(1e-1 + 1e-7),
#     top_k=0.0,
#     top_p=1.0,
#     do_sample=True,
# )

# output = model.generate(
#     **tok_prompt,
#     generation_config=generation_config,
# )
