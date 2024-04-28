import torch
from typing import Any
from torch import nn


class MeanPooler(nn.Module):
    """
    Based on https://www.kaggle.com/debarshichanda/explore-t5
    """

    def __init__(self, config: Any, activation=nn.Tanh()):
        super().__init__()
        self.dense = nn.Linear(config.encoder_size, config.pooler_size)
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_tensor = torch.mean(hidden_states, dim=1)
        # first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class FirstTokenPooler(nn.Module):
    def __init__(self, config: Any, activation=nn.Tanh()):
        super().__init__()
        self.projection = nn.Linear(config.encoder_size, config.pooler_size)
        self.activation = activation

    def forward(self, tokens_outs: torch.Tensor) -> torch.Tensor:
        first_token = tokens_outs[:, 0]
        first_token = self.projection(first_token)
        first_token = self.activation(first_token)
        return first_token