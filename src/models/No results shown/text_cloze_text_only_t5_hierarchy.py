import torch
from typing import Any
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput
from src.common.model_outputs import TextClozeModelOutput

from src.models.base_model import BaseModel
from src.modules.encoders import T5HierarchyEncoderModule


class TextClozeTextOnlyT5HierarchyModel(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(TextClozeTextOnlyT5HierarchyModel, self).__init__(config, device)
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.encoder = T5HierarchyEncoderModule(config)
        self.embedding = self.encoder.embedding
        self.fc2 = nn.Sequential(
            nn.Linear(
                config.answer_candidates * config.answer_embed_size,
                config.answer_embed_size),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(
            config.pooler_size + config.answer_embed_size,
            config.answer_candidates
        )

    def forward(self,
                context: torch.Tensor,
                answers: torch.Tensor,
                target: torch.Tensor) -> TextClozeModelOutput:
        batch_size = context.size(0)
        context_encoding_outputs = self.encoder(context)
        context_encoding_outputs = self.dropout(context_encoding_outputs)
        
        answers = answers.view(-1, answers.size(2))
        answer_embedding_outputs = self.embedding(answers).pooler_output
        answer_embedding_outputs = answer_embedding_outputs.view(batch_size, -1)
        answer_embedding_outputs = self.fc2(answer_embedding_outputs)
        answer_embedding_outputs = self.dropout(answer_embedding_outputs)
        # .view(outputs.size(0), -1)

        outputs = torch.cat(
            (context_encoding_outputs, answer_embedding_outputs), 1)
        # outputs = context_encoding_outputs + answer_embedding_outputs
        logits = self.scores_fc(outputs)
        loss = None

        if target is not None:
            loss = self.loss_function(logits, target)

        return TextClozeModelOutput(
            loss=loss,
            logits=logits,
        )

    def run(self,
            context: torch.Tensor,
            answers: torch.Tensor,
            target: torch.Tensor) -> TextClozeModelOutput:
        return self.forward(context, answers, target)
