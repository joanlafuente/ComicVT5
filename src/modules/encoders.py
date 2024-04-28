import torch
import math
from typing import Any
from torch import nn
from transformers import AutoModel, RobertaModel, T5EncoderModel

from src.modules.poolers import MeanPooler, FirstTokenPooler


class BaseT5EncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseT5EncoderModule, self).__init__()
        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(config.architecture)
        self.embedding = self.encoder.shared
        self.pooler = MeanPooler(config)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(sequences)
        encoder_outputs = self.encoder(inputs_embeds=embedding)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs
        

class BaseT5EncoderModuleCLS_Token(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseT5EncoderModuleCLS_Token, self).__init__()
        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(config.architecture)
        self.embedding = self.encoder.shared
        self.pooler = FirstTokenPooler(config)

    def forward(self, context: torch.Tensor, answers: torch.Tensor, cls_token: torch.Tensor, sep_token: torch.Tensor) -> torch.Tensor:
        context_embed = self.embedding(context)
        answers_embed = self.embedding(answers)
        # print("context:", context_embed.size(), ",answers:", answers_embed.size())
        # print("cls:", cls_token.size(), ",sep:", sep_token.size())
        sequence_embeded = torch.cat((cls_token, context_embed, sep_token, answers_embed, sep_token), dim=1)
        encoder_outputs = self.encoder(inputs_embeds=sequence_embeded)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs


class ImageTextT5EncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(ImageTextT5EncoderModule, self).__init__()
        self.config = config
        self.encoder = T5EncoderModel.from_pretrained(config.architecture)
        self.text_embedding = self.encoder.shared
        self.image_embedding = nn.Linear(config.image_embedding_size, config.embedding_size)
        self.pooler = MeanPooler(config)

    def forward(self, dialogues: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        text_embedding = self.text_embedding(dialogues)
        image_embedding = self.image_embedding(images)
        # print("text:", text_embedding.size(), ",imgs:", image_embedding.size())
        embedding = torch.cat((text_embedding, image_embedding), dim=1)
        encoder_outputs = self.encoder(inputs_embeds=embedding)
        # print("encoder:", encoder_outputs.last_hidden_state.size())
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        # print("pooler:", pooler_outputs.size())
        return pooler_outputs

class T5HierarchyEncoderModule(BaseT5EncoderModule):

    def __init__(self, config: Any) -> None:
        super(T5HierarchyEncoderModule, self).__init__(config)
        self.embedding = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.fc_embedding = nn.Sequential(
            nn.Linear(384, self.config.embedding_size),
            nn.ReLU()
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        batch_size = sequences.size(0)
        sequences = sequences.view(-1, sequences.size(2))
        embedding = self.embedding(sequences)
        embedding = embedding.pooler_output
        embedding = embedding.view(batch_size, -1, 384)
        embedding = self.fc_embedding(embedding)
        embedding = embedding * math.sqrt(self.config.embedding_size)
        embedding = self.positional_encoder(embedding)
        encoder_outputs = self.encoder(inputs_embeds=embedding)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs


class BaseRobertaEncoderModule(nn.Module):

    def __init__(self, config: Any) -> None:
        super(BaseRobertaEncoderModule, self).__init__()
        self.config = config
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.pooler = MeanPooler(config)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(sequences)
        pooler_outputs = self.pooler(encoder_outputs.last_hidden_state)
        return pooler_outputs
