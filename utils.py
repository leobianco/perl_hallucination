"""TODO: write docstring.
"""


from dataclasses import dataclass, field
from typing import Optional

from datasets import concatenate_datasets


@dataclass
class ScriptArguments:
  """"Arguments common to all scripts."""

  dataset_name: str
  model_identifier: str


# TODO: DEPRECATED, ERASE THIS CLASS.
@dataclass
class ModelArguments:
  """
  Arguments specifying the model to train, as well 
  as tokenizer configuration.
  """

  model_identifier: Optional[str] = field(
    default=None, metadata={"help": "the name of the model to train."}
  )

  max_seq_length: Optional[int] = field(
    default=512, metadata={"help": "maximum number of tokens in generations."}
  )


# TODO: DEPRECATED, ERASE THIS CLASS.
@dataclass
class DataArguments:
  """Arguments specifying the data used to train the model."""

  validation_size: Optional[float] = field(
    default=0.2, metadata={"help": "fraction of data to use for validation."}
  )
 

@dataclass
class EvaluatorArguments:
  """TODO: write docstring.
  """

  model_identifier_evaluated: Optional[str] = field(
    default=None, metadata={"help": "identifier of evaluated model."}
  )

  model_identifier_evaluator: Optional[str] = field(
    default=None, metadata={"help": "identifier of evaluator model."}
  )

  n_fewshot: Optional[int] = field(
    default=0, metadata={"help": "the number of fewshot examples."}
  )

