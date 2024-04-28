import torch

from dataclasses import dataclass
from transformers.modeling_outputs import MultipleChoiceModelOutput


@dataclass
class TextClozeModelOutput(MultipleChoiceModelOutput):
    prediction: torch.Tensor = None

    def __post_init__(self):
        super().__post_init__()
        self.prediction = torch.argmax(self.logits, dim=1)