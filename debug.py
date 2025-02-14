"""This script serves to debug the model and tokenizer loading, as well as the generation process, interactively."""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Load data
data = load_dataset("leobianco/npov_rm_processed", split="test")

#queries = data.select_columns(["input_ids", "attention_mask"])
#query = queries[:5]

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-2b-it",
    padding_side="left",
)

model = AutoModelForCausalLM.from_pretrained(
    "leobianco/PERL_eval",
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)
model.eval()

# Config
generation_config = GenerationConfig(
    max_new_tokens=160,
    temperature=(1e-1 + 1e-7),
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
)

#output = model.generate(
#    input_ids=torch.tensor(query["input_ids"]).to(model.device),
#    attention_mask=torch.tensor(query["attention_mask"]).to(model.device),
#    generation_config=generation_config,
#    return_dict_in_generate=True,
#    output_scores=True,
#)
