import os
import torch
import numpy as np
import pandas as pd
import h5py as h5

from torch.utils.data import DataLoader, Dataset
from typing import Any, Tuple
from transformers import PreTrainedTokenizer

"""
Dataset class thought for the task of dialogue generation 
task. This class is specific to the use features of multiple
objects detected for each panel.

As input text we provide the context text, and as target the correct 
dialogue.
"""

class ComicsDialogueGenerationDataset(Dataset[Any]):
    def __init__(self,
                 data: pd.DataFrame,
                 features_h5: h5.File,
                 tokenizer: PreTrainedTokenizer,
                 device: torch.device,
                 config: Any
                 ):
        self.device = device
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.features_h5 = features_h5

    def __len__(self):
        return len(self.data)

    def get_normalized_boxes(self, image_ds) -> torch.Tensor:
        """
        Normalize the bounding boxes to be between 0 and 1.

        Args:
            image_ds: A dataset containing the image features.
        """
        img_h = image_ds['img_h'][()]
        img_w = image_ds['img_w'][()]
        boxes = image_ds['boxes'][()]  # (x1, y1, x2, y2)
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)
        return boxes

    def __getitem__(self, idx: int) -> dict:
        out_dict = {}
        out_dict["sample_id"] = str(idx)
        out_dict['args'] = self.config

        sample = self.data.iloc[idx]
        book_id = sample["book_id"]
        page_id = sample["page_id"]

        ###### Image ######
        f = self.features_h5
        per_image_feats = []
        per_image_boxes = []

        for i in range(4):
            if i == 3:
                panel_id = sample["answer_panel_id"]
            else:
                panel_id = sample[f'context_panel_{i}_id']

            img_id = f"{book_id}_{page_id}_{panel_id}"
            image_ds = f[img_id]

            boxes = self.get_normalized_boxes(image_ds)
            per_image_boxes.append(boxes)

            feats = np.zeros(
                shape=(self.config.n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/features'].read_direct(feats)
            per_image_feats.append(feats)

        boxes = np.stack(per_image_boxes)  # [4, n_boxes, 4]
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        out_dict['boxes'] = boxes

        feats = np.stack(per_image_feats)  # [4, n_boxes, feat_dim]
        feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats

        ###### Text #####
        prefix = "dialogue generation:"
        input_tokens = [prefix]
        source_text = [
            "context00",
            sample["context_text_0_0"],
            "context01",
            sample["context_text_0_1"],
            "context02",
            sample["context_text_0_2"],
            "context10",
            sample["context_text_1_0"],
            "context11",
            sample["context_text_1_1"],
            "context12",
            sample["context_text_1_2"],
            "context20",
            sample["context_text_2_0"],
            "context21",
            sample["context_text_2_1"],
            "context22",
            sample["context_text_2_2"],
        ]
        input_tokens += source_text
        input_text = ' '.join(input_tokens)
        input_ids = self.tokenizer(
            input_tokens,
            max_length=self.config.context_max_speech_size,
            truncation=True,
        ).input_ids
        input_ids = [t for sent in input_ids for t in sent[:-1]] + \
            [self.tokenizer.eos_token_id]
        input_ids += [self.tokenizer.pad_token_id] * \
            (self.config.max_text_length - len(input_ids))

        out_dict['input_text'] = input_text
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        target_text = sample[f"answer_candidate_{sample['correct_answer']}_text"]
        target_ids = self.tokenizer.encode(
            str(target_text), max_length=self.config.gen_max_length, truncation=True)

        assert len(target_ids) <= self.config.gen_max_length, len(target_ids)
        out_dict['target_text'] = target_text
        out_dict['target'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}
        batch_entry['sample_id'] = [entry['sample_id'] for entry in batch]

        B = len(batch)
        V_L = batch[0]['boxes'].size(1)
        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        feat_dim = batch[0]['vis_feats'].size(-1)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * \
            self.tokenizer.pad_token_id

        boxes = torch.zeros(B, 4, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, 4, V_L, feat_dim, dtype=torch.float)
        target = torch.ones(
            B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_text = []

        input_texts = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            boxes[i] += entry['boxes']
            vis_feats[i] += entry['vis_feats']

            target[i, :entry['target_length']] = entry['target']

            input_texts.append(entry['input_text'])

            target_text.append(entry['target_text'])

        batch_entry['input_ids'] = input_ids
        word_mask = target != self.tokenizer.pad_token_id
        target[~word_mask] = -100
        batch_entry['target'] = target

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats
        batch_entry['input_text'] = input_texts
        batch_entry['target_text'] = target_text
        batch_entry['task'] = 'text cloze'

        return batch_entry


def create_dataloader(
    batch_size: int,
    dataset_path: str,
    device: torch.device,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    assert not inference, "This dataset cannot be used for inference."

    train_df = pd.read_csv(
        f"{dataset_path}/text_cloze_train_{config.mode}.csv", delimiter=',')
    train_df = train_df.fillna("")
    dev_df = pd.read_csv(
        f"{dataset_path}/text_cloze_dev_{config.mode}.csv", delimiter=',')
    dev_df = dev_df.fillna("")
    test_df = pd.read_csv(
        f"{dataset_path}/text_cloze_test_{config.mode}.csv", delimiter=',')
    test_df = test_df.fillna("")

    feats_h5_path = os.path.join(dataset_path, config.panel_features_path)
    feats_h5 = h5.File(feats_h5_path, 'r')

    train_dataset = ComicsDialogueGenerationDataset(
        train_df, feats_h5, dataset_kwargs["tokenizer"], device, config)
    val_dataset = ComicsDialogueGenerationDataset(
        dev_df, feats_h5, dataset_kwargs["tokenizer"], device, config)
    test_dataset = ComicsDialogueGenerationDataset(
        test_df, feats_h5, dataset_kwargs["tokenizer"], device, config)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True, 
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True, 
        collate_fn=val_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True, 
        collate_fn=test_dataset.collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader
