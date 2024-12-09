""""
TODO: write proper docstring.
TODO: clean imports.
"""


import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import roc_auc_score

from datasets import load_dataset
from transformers import (
  AutoModelForSequenceClassification, HfArgumentParser, TrainingArguments, 
  set_seed, AutoTokenizer, Trainer, TrainerCallback, 
  DataCollatorForLanguageModeling,
)
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
import evaluate

from data import *
from utils import ScriptArguments


def main():

  #########
  # SETUP #
  #########
  parser = HfArgumentParser(
    (ScriptArguments, TrainingArguments, LoraConfig)
  )

  (
    script_args,
    training_args,
    peft_args,
  ) = parser.parse_args_into_dataclasses()

  # Set seed before instantiating the model, for reproducibility.
  set_seed(training_args.seed)

  tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_identifier,
    padding_side="right",
  )

  ########
  # DATA #
  ########
  rm_data_halomi = load_dataset(script_args.dataset_name)

  id2label = {
    0: 'Yes',
    1: 'No',
  }
  
  label2id = {
    'Yes': 0,
    'No': 1,
  }

  #########
  # MODEL #
  #########
  reward_model = AutoModelForSequenceClassification.from_pretrained(
      script_args.model_identifier,
      num_labels=2,
      id2label=id2label,
      label2id=label2id,
      attn_implementation="eager",
  )

  if training_args.resume_from_checkpoint is not None:
    reward_model = PeftModel.from_pretrained(
      reward_model, 
      training_args.resume_from_checkpoint
    )

  reward_model = get_peft_model(reward_model, peft_args)

  ####################
  # EVALUATION SETUP #
  ####################
  metric = evaluate.load("roc_auc")

  def compute_metrics(eval_preds):
    """Recall that whereas logits where torch tensors before, now they are numpy 
    arrays.

    logits here are the processed_logits from the process_logits_for_evaluation 
    function.

    metric is a 'global' function inside the scope of main.
    """

    # Compute scores
    logits = eval_preds.predictions
    yes_scores = np.exp(logits)[:, 0]
    no_scores = np.exp(logits)[:, 1]
    scores = no_scores / (yes_scores + no_scores)

    # Compute ground truth
    label_ids = eval_preds.label_ids

    metrics = metric.compute(
      references=label_ids,
      prediction_scores=scores
    )

    return metrics


  ###########
  # TRAINER #
  ###########
  trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=rm_data_halomi['train'],
    eval_dataset=rm_data_halomi['test'],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
  )

  ############
  # TRAINING #
  ############
  if training_args.do_train:
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()

    if training_args.push_to_hub:
      trainer.push_to_hub()

  ##############
  # EVALUATION #
  ##############
  if training_args.do_eval:
    trainer.evaluate()


if __name__=="__main__":
  main()
  
