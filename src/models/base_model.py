import torch

from abc import abstractmethod
from typing import Any, OrderedDict
from transformers.modeling_outputs import BaseModelOutput


class BaseModel(torch.nn.Module):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(BaseModel, self).__init__()
        self.config = config
        self.device = device

    def load_checkpoint(self, state_dict: OrderedDict) -> None:
        # Change Multi GPU to single GPU
        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module."):
                new_key = key[len("module."):]
                state_dict[new_key] = state_dict.pop(key)

        self.load_state_dict(state_dict, strict=False)

    @abstractmethod
    def run(self) -> BaseModelOutput:
        """
        Runs one iteration of the model for training or validation.

        Returns:
            The output of the model.
        """
        raise NotImplementedError