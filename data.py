"""TO DO: 
Write docstring.
Add seed for reproducibility.

Data processing utils.
If this script is ran directly, it will load all the datasets involved, 
process them, and save their processed version to HF Hub.
"""


from argparse import ArgumentParser

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from copy import deepcopy


#############################
# GENERAL ENCODING FUNCTION #
#############################

def encode(batch, tokenizer=None, max_seq_length=512):
  return tokenizer(
    batch["prompt"],
    padding="max_length",
    truncation=True,
    max_length=max_seq_length,
    return_tensors="pt",  # though using map() => set_format("torch") later
  )


#############################
# HALOMI DATA PREPROCESSING #
#############################

def change_language_labels(entry):
  """Data processing utility function which replaces the language labels by the
  natural language correspondent version.
  """

  lang_to_natural_lang = {
      'eng_Latn': "English",
      'kas_Deva': "Kashmiri",
      'spa_Latn': "Spanish",
      'mni_Beng': "Manipuri",
      'yor_Latn': "Yoruba",
      'deu_Latn': "German",
      'zho_Hans': "Mandarin",
      'arb_Arab': "Modern Standard Arab",
      'rus_Cyrl': "Russian",
  }

  entry['src_lang'] = lang_to_natural_lang[entry['src_lang']]
  entry['tgt_lang'] = lang_to_natural_lang[entry['tgt_lang']]

  return entry


def change_hallucination_labels(entry):
  """Data processing utility function which replaces the hallucination labels by
  a simplified version.
  """

  class_hallucination_to_label = {
      "1_No_hallucination": "No",
      "2_Small_hallucination": "Yes",
      "3_Partial_hallucination": "Yes",
      "4_Full_hallucination": "Yes",
  }

  entry['class_hall'] = class_hallucination_to_label[entry['class_hall']]

  return entry


def change_omission_labels(entry):
  """Data processing utility function which replaces the omission labels by
  a simplified version.
  """

  class_omit_to_label = {
      "1_No_omission": "No",
      "2_Small_omission": "Yes",
      "3_Partial_omission": "Yes",
      "4_Full_omission": "Yes",
  }

  entry['class_omit'] = class_omit_to_label[entry['class_omit']]

  return entry


def hallucination_labels_to_numerical(entry):
  """Data processing utility function which replaces the hallucination labels by
  a numerical version."""

  entry['class_hall_num'] = 0 if entry['class_hall']=='Yes' else 1

  return entry


def omission_labels_to_numerical(entry):
  """Data processing utility function which replaces the omission labels by
  a numerical version."""

  entry['class_omit_num'] = 0 if entry['class_omit']=='Yes' else 1

  return entry


def process_halomi_data(data_halomi, seed: int = 12345):
  """Preprocesses the HalOmi dataset."""

  # Select relevant subset of columns
  data_halomi = data_halomi.select_columns(
    ['src_lang', 'tgt_lang', 'src_text', 'mt_text', 'class_hall', 'class_omit']
  )

  # Restrict ourselves only to English <-> Spanish examples
  included_langs = ["eng_Latn", "spa_Latn"]

  data_halomi = data_halomi.filter(
    lambda example: (example['src_lang'] in included_langs
                     and example['tgt_lang'] in included_langs)
  )

  # Change language labels to natural language
  data_halomi = data_halomi.map(change_language_labels)

  # Change hallucination labels to 'Yes' or 'No'
  data_halomi = data_halomi.map(change_hallucination_labels)

  # Add numerical version of hallucination
  data_halomi = data_halomi.map(hallucination_labels_to_numerical)

  # Change omission labels to 'Yes' or 'No'
  data_halomi = data_halomi.map(change_omission_labels)

  # Add numerical version of omission
  data_halomi = data_halomi.map(omission_labels_to_numerical)

  # Shuffle rows to mix languages and examples
  data_halomi = data_halomi.shuffle(seed=seed)

  return data_halomi


def load_halomi_data(seed: int = 12345):
  """Loads HalOmi data that is saved in my HF Hub."""

  data_halomi = load_dataset(
    "leobianco/halomi",
    data_files={"train": "halomi_full.tsv"},
    sep='\t',
    split='train'
  )

  data_halomi = process_halomi_data(data_halomi, seed)

  return data_halomi


#####################
# REWARD MODEL DATA #
#####################

def rm_prompt_halomi(entry, validation_test=False):
  """Function for transforming entries in the HalOmi dataset into training
  prompts for the reward model.
  """

  template = (
    "Original text in {src_lang}: {src_text}"
    "\nTranslated text in {tgt_lang}: {mt_text}"
    "\nQuestion: does the translated text contain more information than the "
    "original text?"
    "\nAnswer:"  # It is important that there is no space here
  )

  formatted_prompt = template.format(
      src_lang=entry['src_lang'],
      tgt_lang=entry['tgt_lang'],
      src_text=entry['src_text'],
      mt_text=entry['mt_text'],
  )

  entry['prompt'] = formatted_prompt

  return entry


def process_halomi_data_for_rm(
    data_halomi,
    tokenizer,
    max_seq_length=512,
    validation_size=0.25,
    seed=12345,
):
  # Copy original Halomi data
  rm_data_halomi = deepcopy(data_halomi)

  # Create reward model HalOmi prompts
  rm_data_halomi = rm_data_halomi.map(rm_prompt_halomi)

  # Tokenize reward model prompts
  rm_data_halomi = rm_data_halomi.map(
    encode,
    batched=True,
    fn_kwargs={
      "tokenizer": tokenizer,
      "max_seq_length": max_seq_length,
    }
  )
  rm_data_halomi.set_format("torch")  # due to using map()

  # Rename hallucination class to label
  rm_data_halomi = rm_data_halomi.rename_column("class_hall_num", "label")

  # Split the dataset 
  rm_data_halomi = rm_data_halomi.train_test_split(
    seed=seed,
    test_size=validation_size
  )

  return rm_data_halomi


#################################
# WRITER SFT DATA PREPROCESSING #
#################################

def writer_prompt_halomi(entry, SFT=False):
  """Function for transforming entries in the HalOmi dataset into prompts for
  the writer to translate.
  """

  template = (
    "Translate a text originally written in {src_lang} into {tgt_lang}. "
    "Generate only the translated text, and nothing else."
    "\nOriginal text: {src_text}"
    "\nTranslated text: {ans}"
  )

  ans = entry["mt_text"] if SFT else ""

  formatted_prompt = template.format(
      src_lang=entry["src_lang"],
      tgt_lang=entry["tgt_lang"],
      src_text=entry["src_text"],
      ans=ans,
  )

  entry["prompt"] = formatted_prompt

  return entry


def process_halomi_data_for_writer_sft(halomi_data):
  """Check out 
  https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
  to see the data format for the SFTTrainer. It just requires a 'prompt' and a 
  'completion' column in the dataset, no need for tokenizing directly.
  """

  # Filter for examples without hallucinations and without coverage errors 
  writer_sft_data = halomi_data.filter(
    lambda example:
    (example['class_hall']=='No' and example['class_omit']=='No')
  )
  
  # Create 'prompt' column
  writer_sft_data = writer_sft_data.map(writer_prompt_halomi)
  
  # Create 'completion' column
  writer_sft_data = writer_sft_data.rename_column('mt_text', 'completion')
  
  # Keep only 'prompt' and 'completion' columns --> breaks train on completion
  # writer_sft_data = writer_sft_data.select_columns(['prompt', 'completion'])

  return writer_sft_data


def formatting_prompts_func(entry):
  """Weird function required by the SFTTrainer."""

  template = (
    "Translate a text originally written in {src_lang} into {tgt_lang}. "
    "Generate only the translated text, and nothing else."
    "\nOriginal text: {src_text}"
  )

  output_texts = []

  for i in range(len(entry['src_text'])):

    formatted_prompt = template.format(
        src_lang=entry["src_lang"][i],
        tgt_lang=entry["tgt_lang"][i],
        src_text=entry["src_text"][i],
    )

    text = (
        f"{formatted_prompt}\nTranslated text: {entry['completion'][i]}"
    )

    output_texts.append(text)

  return output_texts


###########################
# PERL DATA PREPROCESSING #
###########################

def format_perl_translation_data(entry, src_lang: str, tgt_lang: str):
  """Helper function for formatting PERL data, to be mapped over dataset."""

  entry["src_lang"] = src_lang
  entry["tgt_lang"] = tgt_lang
  entry["src_text"] = entry[src_lang]
  entry["mt_text"] = entry[tgt_lang]

  return entry


def process_data_for_perl(
    perl_data_halomi,
    tokenizer,
    max_seq_length=512,
    seed=12345,
    perl_train_size=1000,
    perl_validation_size=500,
):
  # Half of it will be English -> Spanish, the other half Spanish -> English.
  n = perl_data_halomi.num_rows
  first_half = perl_data_halomi.select(range(n // 2))
  second_half = perl_data_halomi.select(range(n // 2, n))

  first_half = first_half.map(
      format_perl_translation_data,
      fn_kwargs=dict(
          src_lang="English",
          tgt_lang="Spanish"
      )
  )

  second_half = second_half.map(
      format_perl_translation_data,
      fn_kwargs=dict(
          src_lang="Spanish",
          tgt_lang="English"
      )
  )

  perl_data_halomi = concatenate_datasets([first_half, second_half])
  perl_data_halomi = perl_data_halomi.shuffle(seed=seed)

  # Create and tokenize prompts for writer.
  perl_data_halomi = perl_data_halomi.map(writer_prompt_halomi)

  perl_data_halomi = perl_data_halomi.map(
    encode,
    batched=True,
    fn_kwargs={
      "tokenizer": tokenizer,
      "max_seq_length": max_seq_length,
    }
  )
  perl_data_halomi.set_format("torch")  # due to using map()

  # Split into training, validation, and test splits.
  perl_data_halomi = perl_data_halomi.train_test_split(
    seed=seed,
    train_size=perl_train_size,
    test_size=perl_validation_size,
  )

  return perl_data_halomi


if __name__=="__main__":

  parser = ArgumentParser()

  # General
  parser.add_argument("--seed", type=int, default=12345)

  # Tokenizer
  parser.add_argument(
    "--tokenizer_model",
    type=str,
    default="google/gemma-2-2b-it"
  )
  parser.add_argument("--max_seq_length", type=int, default=512)

  # Halomi
  parser.add_argument(
    "--halomi_repo_id",
    type=str,
    default="leobianco/halomi"
  )
  parser.add_argument(
    "--halomi_processed_repo_id",
    type=str,
    default="leobianco/halomi_processed"
  )

  # Reward model
  parser.add_argument(
    "--rm_halomi_processed_repo_id",
    type=str,
    default="leobianco/rm_halomi_processed"
  )
  parser.add_argument("--rm_validation_size", type=float, default=0.2)

  # Writer SFT
  parser.add_argument(
    "--writer_sft_halomi_processed_repo_id",
    type=str,
    default="leobianco/writer_sft_halomi_processed"
  )

  # PERL
  parser.add_argument(
    "--perl_data_repo_id",
    type=str,
    default="okezieowen/english_to_spanish",
  )
  parser.add_argument(
    "--perl_data_processed_repo_id",
    type=str,
    default="leobianco/perl_halomi_processed"
  )
  parser.add_argument("--perl_train_size", type=int, default=1000)
  parser.add_argument("--perl_validation_size", type=int, default=500)

  script_args = parser.parse_args()

  # Tokenizer
  tokenizer = AutoTokenizer.from_pretrained(
    script_args.tokenizer_model,
    padding_side="right",
  )

  # Halomi data.
  data_halomi = load_dataset(
    script_args.halomi_repo_id,
    data_files={"train": "halomi_full.tsv"},
    sep='\t',
    split='train'
  )

  data_halomi_processed = process_halomi_data(data_halomi, script_args.seed)
  
  data_halomi_processed.push_to_hub(
    repo_id=script_args.halomi_processed_repo_id,
  )


  # Reward model data.
  rm_data_halomi_processed = process_halomi_data_for_rm(
    data_halomi_processed,
    tokenizer,
    max_seq_length=script_args.max_seq_length,
    validation_size=script_args.rm_validation_size,
    seed=script_args.seed,
  )
  
  rm_data_halomi_processed["train"].push_to_hub(
    repo_id=script_args.rm_halomi_processed_repo_id,
    split="train",
  )

  rm_data_halomi_processed["test"].push_to_hub(
    repo_id=script_args.rm_halomi_processed_repo_id,
    split="test",
  )

  # Writer SFT data. 
  writer_sft_processed = process_halomi_data_for_writer_sft(
    data_halomi_processed
  )

  writer_sft_processed.push_to_hub(
    repo_id=script_args.writer_sft_halomi_processed_repo_id,
  )

  # PERL data.
  perl_data_halomi = load_dataset(
    script_args.perl_data_repo_id,
    split="train",
  )

  perl_data_halomi_processed = process_data_for_perl(
    perl_data_halomi,
    tokenizer,
    max_seq_length=script_args.max_seq_length,
    seed=script_args.seed,
    perl_train_size=script_args.perl_train_size,
    perl_validation_size=script_args.perl_validation_size,
  )

  perl_data_halomi_processed["train"].push_to_hub(
    repo_id=script_args.perl_data_processed_repo_id,
    split="train",
  )

  perl_data_halomi_processed["test"].push_to_hub(
    repo_id=script_args.perl_data_processed_repo_id,
    split="test",
  )

