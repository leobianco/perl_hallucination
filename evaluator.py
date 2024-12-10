"""
python -m IPython -i evaluator.py \
        -- \
        --seed 12345 \
        --writer_model_base "google/gemma-2-2b-it" \
        --writer_model_lora "leobianco/halomi_writer_sft" \
        --max_tokens 256 \
        --evaluator_model "google/gemma-2-27b-it" \
        --num_fewshot_examples 4 \
        --evaluate_evaluator False \
        --threshold 0.268

Evaluation script.

Given a checkpoint of the writer and validation prompts, we use the writer to 
generate completions, and send the prompts + completions to the evaluator. We 
then get the score for the token "No", normalized to consider only "Yes" as 
an alternative.

Checkpoints that we will consider: writer SFT (baseline), writer checkpoints 
over PERL. All of these are fine-tuned Gemma 2 2b models.

There is a separate script to evaluate the quality of the evaluator 
itself, and to choose the classification threshold.

The writer checkpoint will be loaded as a vLLM LLM.
"""


import gc
from dataclasses import dataclass, field
from typing import Optional

from sklearn.metrics import (
  roc_curve, roc_auc_score, RocCurveDisplay, accuracy_score, precision_score
)
import numpy as np
import torch
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel

from huggingface_hub import snapshot_download
from datasets import load_dataset, concatenate_datasets
from transformers import (
  HfArgumentParser, AutoModelForCausalLM, AutoTokenizer
)
from trl import BaseJudge

from data import process_data_for_perl


@dataclass
class ScriptArguments:

  writer_model_base: str = field(metadata=
    {
      "help": "The base model for the writer (name or path)."
    }
  )

  writer_model_lora: str = field(metadata=
    {
      "help": "The path to the LoRA adapters of the writer model."
    }
  )

  evaluator_model: str = field(
    default="google/gemma-2-27b-it",
    metadata={
      "help": "The model name or path to the model to use as evaluator."
    }
  ) 

  seed: int = field(default=12345)

  max_tokens: int = field(default=128)

  temperature: float = field(default=0.8)

  top_p: float = field(default=0.9)

  num_fewshot_examples: Optional[int] = field(default=None, metadata={
    "help": "The number of fewshot examples to give to the evaluator."
    }
  )

  evaluate_evaluator: bool = field(default=False, metadata={
      "help": "Whether to run evaluation of the evaluator or not."
    }
  )

  threshold: float = field(default=0.5, metadata={
      "help": "The value of the threshold to turn scores into classif."
    }
  )


def get_fewshot_examples(data, n: int, seed: int):
  """Returns n balanced fewshot examples from data."""

  positive_examples = (
    data.filter(lambda entry: entry['class_hall']=='Yes')
        .shuffle(seed=seed)
        .select(range(n // 2))
  )

  negative_examples = (
    data.filter(lambda entry: entry['class_hall']=='No')
        .shuffle(seed=seed)
        .select(range(n // 2))
  )

  fewshot_examples = concatenate_datasets(
    [positive_examples, negative_examples]).shuffle(seed=seed)

  return fewshot_examples


def evaluator_prompt_halomi(entry, fewshot_examples=None, use_mt_text=False):
  """Function for transforming entries in the HalOmi dataset into prompts for
  the evaluator model. 

  TO DO: perhaps split this function into two functions instead of using the 
  use_mt_text parameter.
  """

  preamble = (
    "The following are examples of an expert translator and linguist noting "
    "when the Translation to an Original text contains additional information "
    "that is not part of the original text.\n\n"
  )

  template = (
    "Original text in {src_lang}:{src_text}"
    "\nTranslated text in {tgt_lang}:{mt_text}"
    "\nExpert translator and linguist review: The Translated text contains "
    "additional information with respect to the Original text (Yes/No):{ans}"
  )

  prompt = preamble

  if use_mt_text:
    translation = entry["mt_text"]  # for HalOmi and evaluation of evaluator
  else:
    translation = entry["completion"]  # for evaluation of a writer checkpoint

  formatted_prompt = template.format(
      src_lang=entry["src_lang"],
      tgt_lang=entry["tgt_lang"],
      src_text=entry["src_text"],
      mt_text=translation,
      ans="",
  )

  if fewshot_examples is not None:
    for fewshot_example in fewshot_examples:
      fewshot_prompt = template.format(
        src_lang=fewshot_example["src_lang"],
        tgt_lang=fewshot_example["tgt_lang"],
        src_text=fewshot_example["src_text"],
        mt_text=fewshot_example["mt_text"],
        ans=fewshot_example["class_hall"],
      )

      prompt += fewshot_prompt + "\n\n"

    prompt += formatted_prompt

  else:
    prompt += formatted_prompt

  return prompt


def evaluator_score(
    evaluator,
    tokenized_evaluator_prompts,
    yes_token_id,
    no_token_id
  ):
  """Evaluator score for translation."""

  scores = []

  for idx, tokenized_prompt in enumerate(tokenized_evaluator_prompts):
    with torch.no_grad():
      # Using cache was giving me errors, related to Gemma 2 or to the fact
      # that I need to use an older version of Transformers for RLOO to work...
      # See https://github.com/huggingface/transformers/issues/33147
      outputs = evaluator(**tokenized_prompt, use_cache=False)

    score_yes = torch.exp(outputs.logits[:, -1, yes_token_id])
    score_no = torch.exp(outputs.logits[:, -1, no_token_id])
    score = score_no / (score_yes + score_no)
    scores.append(score)

  return scores


if __name__=="__main__":

  #########
  # SETUP #
  #########

  # Parse the arguments
  parser = HfArgumentParser(ScriptArguments)
  script_args = parser.parse_args_into_dataclasses()[0]

  # Load validation dataset.
  val_data = load_dataset(
    "leobianco/perl_halomi_processed",
    split="test",
  )

  # Load HalOmi data, either for evaluating the evaluator or for getting 
  # fewshot examples.
  data_halomi = load_dataset(
    "leobianco/halomi_processed",
    split="train",
  )

  # Get fewshot examples to aid the evaluator.
  # Notice that the fewshot examples come from the HalOmi dataset,
  # and not from okezieowen's one, since the examples must be "balanced"
  # i.e., as many examples with hallucinations than examples without.
  fewshot_examples = get_fewshot_examples(
      data_halomi,
      script_args.num_fewshot_examples,
      seed=script_args.seed,
  )

  # Instantiate tokenizer.
  tokenizer = AutoTokenizer.from_pretrained(
    script_args.evaluator_model
  )
  
  yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
  no_token_id = tokenizer.convert_tokens_to_ids("No")
  

  if script_args.evaluate_evaluator:

    #########################
    # SCORE HALOMI DATASET  #
    #########################

    # Instantiate evaluator.
    evaluator = AutoModelForCausalLM.from_pretrained(
        script_args.evaluator_model,
        device_map="auto",
        attn_implementation="eager",
    )
    evaluator.eval()

    # Build the evaluator prompts.
    evaluator_prompts = [
      evaluator_prompt_halomi(
        entry,
        fewshot_examples,
        use_mt_text=True
      ) for entry in data_halomi
    ]
  
    # Tokenize them.
    tokenized_evaluator_prompts = [
      tokenizer(prompt, return_tensors="pt") for prompt in evaluator_prompts 
    ]
  
    # Send to device.
    tokenized_evaluator_prompts = [
      {k: v.to(evaluator.device) for k, v in tokenized_prompt.items()}
      for tokenized_prompt in tokenized_evaluator_prompts
    ]

    # Score and bring back to cpu, transform back to list
    scores = evaluator_score(
      evaluator,
      tokenized_evaluator_prompts,
      yes_token_id,
      no_token_id,
    )

    scores = [score.cpu().item() for score in scores]
      
    #####################
    # CALCULATE METRICS #
    #####################

    ground_truth = data_halomi["class_hall_num"]
    auc = roc_auc_score(ground_truth, scores)
    fpr, tpr, thresholds = roc_curve(ground_truth, scores)
    threshold_idx = np.argmax(tpr - fpr)
    threshold = thresholds[threshold_idx]
    classif_at_threshold = [
      0 if score < threshold else 1 
      for score in scores
    ]

    ################################
    # DISPLAY METRICS + SAVE PLOTS #
    ################################

    print("AUC:", auc)
    print("Threshold:", threshold)
    print("TPR (recall):", tpr[threshold_idx])
    print("FPR:", fpr[threshold_idx])
    print("Accuracy:", accuracy_score(ground_truth, classif_at_threshold))
    print("Precision:", precision_score(ground_truth, classif_at_threshold))

    # ROC-AUC plot
    RocCurveDisplay.from_predictions(ground_truth, scores)
    plt.scatter([fpr[threshold_idx]], [tpr[threshold_idx]], c='r')
    plt.savefig(
      f"eval_evaluator_auc_curve_{script_args.num_fewshot_examples}_shot"
    )

    # Histogram
    bins = np.arange(0, 1, 0.05)
    scores_no = [
      score 
      for idx, score in enumerate(scores)
      if ground_truth[idx]==1
    ]
    scores_yes = [
      score 
      for idx, score in enumerate(scores)
      if ground_truth[idx]==0
    ]
    plt.vlines(x=threshold, ymin=0, ymax=175, colors='r')
    plt.hist(scores_no, bins=bins, alpha=0.5, label="No")
    plt.hist(scores_yes, bins=bins, alpha=0.5, label="Yes")
    plt.legend()
    plt.savefig(
      f"eval_evaluator_histogram_{script_args.num_fewshot_examples}_shot"
    )

  else:

    ########################
    # GENERATE COMPLETIONS #
    ########################
  
    # To evaluate the base model (no LoRA), just re-use the base model path
    # on the LoRA adapters path.
    enable_lora = (
      False 
      if script_args.writer_model_base==script_args.writer_model_lora
      else True
    )
    
    if enable_lora:
      # Dowload the LoRA adapters and save locally.
      lora_path = snapshot_download(
        repo_id=script_args.writer_model_lora,
        allow_patterns=["*.json", "*.safetensors"],
      )

    # Instantiate evaluated checkpoint as a vLLM LLM.
    llm = LLM(model=script_args.writer_model_base, enable_lora=enable_lora)
  
    # Generate completions with writer.
    sampling_params = SamplingParams(
      seed=script_args.seed,
      temperature=script_args.temperature,
      top_p=script_args.top_p,
      max_tokens=script_args.max_tokens,
    )
  
    prompts = [val_data[i]['prompt'] for i in range(val_data.num_rows)]
  
    if enable_lora:
      outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=LoRARequest("writer_lora_adapter", 1, lora_path),
      )
    else:
      outputs = llm.generate(
        prompts,
        sampling_params
      )

    generations = [output.outputs[0].text for output in outputs]
  
    # Add generations to the val_data under a column named "completion".
    val_data = val_data.add_column("completion", generations)
  
    # Free vLLM memory to free up space for evaluator.
    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()
  
    ########################
    # EVALUATE COMPLETIONS #
    ########################

    # Instantiate evaluator.
    evaluator = AutoModelForCausalLM.from_pretrained(
        script_args.evaluator_model,
        device_map="auto",
        attn_implementation="eager",
    )
    evaluator.eval()
   
    # Build the evaluator prompts using fewshot examples + completions.
    evaluator_prompts = [
      evaluator_prompt_halomi(entry, fewshot_examples) for entry in val_data
    ]
  
    # Tokenize them.
    tokenized_evaluator_prompts = [
      tokenizer(prompt, return_tensors="pt") for prompt in evaluator_prompts 
    ]
  
    # Send to device.
    tokenized_evaluator_prompts = [
      {k: v.to(evaluator.device) for k, v in tokenized_prompt.items()}
      for tokenized_prompt in tokenized_evaluator_prompts
    ]
  
    # Evaluate and bring back scores to cpu, make them floats. 
    scores = evaluator_score(
      evaluator,
      tokenized_evaluator_prompts,
      yes_token_id,
      no_token_id
    )

    scores = [score.cpu().item() for score in scores]
  
    # Compute global rate of hallucinations.
    classifs = [0 if score < script_args.threshold else 1 for score in scores]
    rate_hallucination = 1 - sum(classifs)/len(classifs)
    print("Rate of hallucination:", rate_hallucination)
    
