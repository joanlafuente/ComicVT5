import torch
from typing import Any
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from thop import profile, clever_format

from src.common.model_outputs import TextClozeModelOutput
from src.modules.encoders import ImageTextT5EncoderModule
from src.modules.poolers import MeanPooler
from src.models.base_model import BaseModel



# Based on: https://github.com/leftthomas/SimCLR
class Model_SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_SimCLR, self).__init__()

        list_modules = []
        for name, module in resnet50().named_children():
            if not isinstance(module, nn.Linear):
                list_modules.append(module)
        # encoder
        self.encoder = nn.Sequential(*list_modules)

        # projection head
        self.projection = nn.Sequential(
                                nn.Linear(2048, 512, bias=False), 
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True), 
                                nn.Linear(512, feature_dim, bias=True))
    
    def encode(self, x):
        features = self.encoder(x)
        return torch.flatten(features, start_dim=1)

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.projection(feature)
        return feature, out


class TextClozeImageTextT5Model(BaseModel):

    def __init__(self, config: Any, device: torch.device) -> None:
        super(TextClozeImageTextT5Model, self).__init__(config, device)
        self.num_labels = config.answer_candidates
        self.loss_function = nn.CrossEntropyLoss()
        self.images_embed = Model_SimCLR().to(device)
        profile(self.images_embed, inputs=(torch.randn(1, 3, 112, 112).cuda(),))
        self.images_embed.load_state_dict(torch.load(config.path_SIM_CLR_model))
        self.encoder = ImageTextT5EncoderModule(config)
        self.dropout = nn.Dropout(config.dropout)
        self.scores_fc = nn.Linear(
            config.pooler_size,
            config.answer_candidates
        )

    def forward(self,
        context_dialogues: torch.Tensor,
        images: torch.Tensor,
        answer_dialogues: torch.Tensor,
        target: torch.Tensor
    ) -> TextClozeModelOutput:
        batch_size = context_dialogues.size(0)
        

        dialogues_joint = torch.cat((
            context_dialogues.view(batch_size, -1),
            answer_dialogues.view(batch_size, -1)
            ), dim=1)

        imgs_embedings = self.images_embed(images)

        context_encoding_outputs = self.encoder(dialogues_joint, imgs_embedings)
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
            images: [batch_size, max_panels, 197, 768]
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
        
