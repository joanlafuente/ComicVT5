import torch
from torch import nn
from typing import Any
from src.models.modeling_vlt5_head import VLT5
from pyats.datastructures import AttrDict

"""
In this file we define the model based on the modified VLT5
which has a classification head with LeakyReLU activation.
"""
class TextClozeImageTextVLT5Model(VLT5):
    def __init__(self, config: Any, device: torch.device):
        model_config = VLT5.create_model_config(config)
        super().__init__(model_config)
        self.m_device = device
        pretrained_w = torch.load(
            config.pretrained_weights,
            map_location=device
        )
        self.load_checkpoint(pretrained_w)
        self.loss_function = nn.CrossEntropyLoss()
        # To use feature extractors with other number of dimensions
        if config.feature_dim_image != 2048:
            self.linear_projection = torch.nn.Linear(768, 2048)
            self.project = True
        else:
            self.project = False
        

    def run(self, *args, **kwargs):
        device = self.m_device
        # print(kwargs.keys())
        input_ids = kwargs['input_ids'].to(device)
        B = len(input_ids)
        V_L = kwargs['vis_feats'].size(2)
        # In case the input features are not 2048, the dimensions in wich the model expects them
        # Change the dtype of vis_feats to float32 tensor
        kwargs['vis_feats'] = kwargs['vis_feats'].to(dtype=torch.float32)
        if self.project:
            vis_feats = self.linear_projection(kwargs['vis_feats'].to(device))
            vis_feats = vis_feats.view(B, 4*V_L, 2048).to(device)
        else:
            vis_feats = kwargs['vis_feats'].view(B, 4*V_L, 2048).to(device)
        vis_pos = kwargs['boxes'].to(device).view(B, 4*V_L, 4)

        

        img_order_ids = [0] * V_L + [1] * V_L + [2] * V_L + [3] * V_L
        img_order_ids = torch.tensor(
            img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 4*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(
            B, 4, -1).contiguous().view(B, 4*V_L)

    
        logits = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
        )

        output = AttrDict()
        if "labels" in kwargs:
            loss = self.loss_function(logits, kwargs["labels"].to(device))
            output.loss = loss

        
        output.logits = logits.detach()
        output.prediction = logits.max(1)[1]

        return output


