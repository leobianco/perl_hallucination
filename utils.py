"""TODO: write docstring."""

from dataclasses import dataclass, field

from peft import LoraConfig


@dataclass
class ScriptArguments:
    """ "Arguments common to all scripts (reward model, SFT, PERL)."""

    dataset_name: str
    model_identifier: str


@dataclass
class CustomLoraConfig(LoraConfig):
    """Work around HfArgumentParser bug..."""

    init_lora_weights: bool = field(default=True)
    layers_to_transform: int = field(default=None)
    loftq_config: dict = field(default_factory=dict)
