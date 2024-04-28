import torch
from typing import Any
from torch import nn

from src.common.model_outputs import TextClozeModelOutput
from src.modules.encoders import ImageTextT5EncoderModule, BaseT5EncoderModule
from src.modules.poolers import MeanPooler
from src.models.base_model import BaseModel


class TextClozeImageTextT5Model(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(TextClozeImageTextT5Model, self).__init__(config, device)
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.images_pooler = MeanPooler(config)
        self.encoder = ImageTextT5EncoderModule(config)
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(config.pooler_size, config.answer_candidates)

    def forward(self,
        context_dialogues: torch.Tensor,
        images: torch.Tensor,
        answer_dialogues: torch.Tensor,
        target: torch.Tensor
    ) -> TextClozeModelOutput:
        batch_size = context_dialogues.size(0)
        
        # Images are not to be embedded
        # images = self.images_pooler(images)

        dialogues_joint = torch.cat((
            context_dialogues.view(batch_size, -1),
            answer_dialogues.view(batch_size, -1)
            ), dim=1)
        context_encoding_outputs = self.encoder(dialogues_joint, images)
        context_encoding_outputs = self.dropout(context_encoding_outputs)
        logits = self.scores_fc(context_encoding_outputs)

        loss = None

        if target is not None:
                loss = self.loss_function(logits, target)

        return TextClozeModelOutput(
            loss=loss,
            logits=logits,
        )


    def run(
        self,
        context_dialogues: torch.Tensor,
        images: torch.Tensor,
        answer_dialogues: torch.Tensor,
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
            context_dialogues,
            images,
            answer_dialogues,
            target
        )
        
