import torch
import numpy as np

from typing import Any
from src.models.modeling_vlt5 import VLT5
from src.tokenizers.vlt5_tokenizers import VLT5TokenizerFast
from attrdict import AttrDict
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

"""
A model based on VLT5 that we use to generate a posible dialogue
given the context, the panel images and optionally a list of possible 
dilogues in the transformer encoder.
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
        if config.feature_dim_image != 2048:
            self.linear_projection = torch.nn.Linear(768, 2048)
            self.project = True
        else:
            self.project = False

        self.tokenizer = VLT5TokenizerFast.from_pretrained(
                config.backbone,
                max_length=config.max_text_length,
                do_lower_case=config.do_lower_case,
            )
        self.sentence_embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v1', device=device)


    def run(self, *args, **kwargs):
        device = self.m_device
        input_ids = kwargs['input_ids'].to(device)
        B = len(input_ids)
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
        text_token_pred = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            do_sample=False,
            num_beams=2,
            max_length=35
        )

        # Decode the predicted token to text
        text_predicted = []
        for i in range(text_token_pred.size(0)):
            text_predicted.append(self.tokenizer.decode(text_token_pred[i, :], skip_special_tokens=True))
        # print(text_predicted)
        output['text_predicted'] = text_predicted
        embeddings_predicted = self.sentence_embedder.encode(text_predicted, show_progress_bar=False)
        # print(text_predicted)
        # print(kwargs["target_text"])
        all_preds = []
        for i in range(len(kwargs["list_answers"])):
            similarities = []
            for j in range(len(kwargs["list_answers"][i])):
                # Compute the BLEU score between the predicted text and the posible answer
                similarities.append(cos_sim(self.sentence_embedder.encode(kwargs["list_answers"][i][j], show_progress_bar=False), embeddings_predicted[i]))
            all_preds.append(np.argmax(similarities))
        # print(all_preds)
        # print(kwargs["label"])
        output["prediction"] = torch.tensor(all_preds)
        output["target"] = kwargs["label"]
        output["loss"] = torch.tensor([0.0]*B)

        return output
