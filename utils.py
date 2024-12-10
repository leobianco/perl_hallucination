"""TODO: write docstring.
"""


from dataclasses import dataclass, field
from typing import Optional

from datasets import concatenate_datasets


@dataclass
class ScriptArguments:
  """"Arguments common to all scripts (reward model, SFT, PERL)."""

  dataset_name: str
  model_identifier: str

