import torch

from typing import Any
from src.models.modeling_vlt5 import VLT5
from attrdict import AttrDict
from math import floor

"""
A model only to be used for inference, it generates a dialogue given 
the context, the panel images and the start of the correct dialogue.
"""

class DialogueGenerationVLT5Model(VLT5):
    def __init__(self, config: Any, device: torch.device):
        model_config = VLT5.create_model_config(config)
        super().__init__(model_config)
        self.m_device = device
        pretrained_w = torch.load(
            config.pretrained_weights,
            map_location=device
        )
        self.load_checkpoint(pretrained_w)
        self.percentage_answer_given_dec = config.percentage_answer_given_dec
        if config.feature_dim_image != 2048:
            self.linear_projection = torch.nn.Linear(768, 2048)
            self.project = True
        else:
            self.project = False


    def run(self, *args, **kwargs):
        device = self.m_device
        input_ids = kwargs['input_ids'].to(device)
        B = len(input_ids)
        assert B == 1, "Batch size must be 1."
        V_L = kwargs['vis_feats'].size(2)
        kwargs['vis_feats'] = kwargs['vis_feats'].to(dtype=torch.float32)
        if self.project:
            vis_feats = self.linear_projection(kwargs['vis_feats'].to(device))
            vis_feats = vis_feats.view(B, 4*V_L, 2048).to(device)
        else:
            vis_feats = kwargs['vis_feats'].view(B, 4*V_L, 2048).to(device)
        vis_pos = kwargs['boxes'].to(device).view(B, 4*V_L, 4)

        lm_labels = kwargs["target"].to(device)

        img_order_ids = [0] * V_L + [1] * V_L + [2] * V_L + [3] * V_L
        img_order_ids = torch.tensor(
            img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 4*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(
            B, 4, -1).contiguous().view(B, 4*V_L)

        output = AttrDict({})
        # print(floor(lm_labels.size(1)*self.percentage_answer_given_dec))

        output["prediction"] = self.generate(
            input_ids=input_ids,
            decoder_input_ids=lm_labels[0, :floor(lm_labels.size(1)*self.percentage_answer_given_dec)].to(device).unsqueeze(0),
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            do_sample=True,
            temperature=0.6,
            max_length=35,
            top_p=0.9
        )
        # print(output["prediction"].size())
        # print(lm_labels.size())
        # print(output["prediction"])
        # print(lm_labels)
        output["loss"] = torch.tensor([0.0]*B)
        output["target_tokens_croped"] = lm_labels[0, floor(lm_labels.size(1)*self.percentage_answer_given_dec)-1:]
        output["out_tokens_croped"] = output["prediction"][0, floor(lm_labels.size(1)*self.percentage_answer_given_dec)-1:]

        return output
