import torch
from typing import Any
from torch import nn

from src.common.model_outputs import TextClozeModelOutput
from src.modules.encoders import BaseT5EncoderModuleCLS_Token
from src.models.base_model import BaseModel


class TextClozeTextOnlyT5Model(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(TextClozeTextOnlyT5Model, self).__init__(config, device)
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.encoder = BaseT5EncoderModuleCLS_Token(config)
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(config.pooler_size, config.answer_candidates)
        self.CLS_token = torch.nn.Parameter(torch.randn(config.encoder_size))
        self.SEP_token = torch.nn.Parameter(torch.randn(config.encoder_size))

    def forward(self,
        context_dialogues: torch.Tensor,
        answer_dialogues: torch.Tensor,
        target: torch.Tensor
    ) -> TextClozeModelOutput:

        batch_size = context_dialogues.size(0)

        cls_token = self.CLS_token.repeat(batch_size, 1).unsqueeze(1)
        sep_token = self.SEP_token.repeat(batch_size, 1).unsqueeze(1)

        context_dialogues = context_dialogues.view(batch_size, -1)
        answer_dialogues = answer_dialogues.view(batch_size, -1)

        out_encoder = self.encoder(context=context_dialogues, 
                                answers=answer_dialogues, 
                                cls_token=cls_token, 
                                sep_token=sep_token
                                )
        out_encoder = self.dropout(out_encoder)
        logits = self.scores_fc(out_encoder)

        loss = None

        if target is not None:
                loss = self.loss_function(logits, target)

        return TextClozeModelOutput(
            loss=loss,
            logits=logits,
        )


    def run(
        self,
        context: torch.Tensor,
        answers: torch.Tensor,
        target: torch.Tensor
    ) -> TextClozeModelOutput:
        """
        Args:
            dialogues: [batch_size, max_dialogues, max_dialogue_length]
            images: [batch_size, max_panels, 1, 768]
            answers: [batch_size, max_dialogues, max_dialogue_length]
            target: [batch_size]

        Returns:
            loss: [batch_size]
            logits: [batch_size, num_labels]
        """
        return self.forward(
            context,
            answers,
            target
        )
        
