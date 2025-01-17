from datasets import load_dataset

# Load data 
data = load_dataset("leobianco/perl_halomi_processed", split="train")

# Load model
model = AutoModelForCausalLM.from_pretrained(
  "google/gemma-2-2b-it",
  device_map="auto",
  attn_implementation="eager",
)
model.eval()

