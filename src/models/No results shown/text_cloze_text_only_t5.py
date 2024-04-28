import torch
from typing import Any
from torch import nn
from transformers.modeling_outputs import MultipleChoiceModelOutput
from src.common.model_outputs import TextClozeModelOutput

from src.models.base_model import BaseModel
from src.modules.encoders import BaseT5EncoderModule


class TextClozeTextOnlyT5Model(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(TextClozeTextOnlyT5Model, self).__init__(config, device)
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.encoder = BaseT5EncoderModule(config)
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(
            config.pooler_size,
            config.answer_candidates
        )

    def forward(self,
                context: torch.Tensor,
                answers: torch.Tensor,
                target: torch.Tensor) -> TextClozeModelOutput:
        batch_size = context.size(0)
        joint = torch.cat((context.view(batch_size, -1), answers.view(batch_size, -1)), 1)
        context_encoding_outputs = self.encoder(joint)
        context_encoding_outputs = self.dropout(context_encoding_outputs)
        logits = self.scores_fc(context_encoding_outputs)
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
