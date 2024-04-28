import torch
import math
from typing import Any
from torch import nn
from transformers import RobertaModel, T5EncoderModel


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float(
        ) * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class T5Embedding(nn.Module):

    def __init__(self, config: Any) -> None:
        super(T5Embedding, self).__init__()
        self.config = config
        self.embedding = T5EncoderModel.from_pretrained("t5-small")

        # for param in self.embedding.parameters():
        #     param.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Linear(512, config.embedding_size),
            nn.ReLU()
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size = sequences.size(0)
        sequence_length = sequences.size(2)
        sequences = sequences.view(-1, sequence_length)
        embedding = self.embedding(sequences).last_hidden_state
        embedding = embedding.view(batch_size, -1, 512)
        embedding = self.fc1(embedding)
        return embedding


class RobertaEmbedding(nn.Module):

    def __init__(self, config: Any) -> None:
        super(RobertaEmbedding, self).__init__()
        self.config = config
        self.embedding = RobertaModel.from_pretrained("roberta-base")

        # for param in self.embedding.parameters():
        #     param.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Linear(768, config.embedding_size),
            nn.ReLU()
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size = sequences.size(0)
        sequence_length = sequences.size(2)
        sequences = sequences.view(-1, sequence_length)
        embedding = self.embedding(sequences).last_hidden_state
        embedding = embedding.view(batch_size, -1, 768)
        embedding = self.fc1(embedding)
        return embedding
