import os
import torch
import numpy as np
import pandas as pd
import h5py as h5

from torch.utils.data import DataLoader, Dataset
from typing import Any, Tuple
from transformers import PreTrainedTokenizer

"""
Dataset class thought for the text-cloze task. This class 
is specific to the use features of multiple objects detected 
for each panel.

As input text we provide the context text and the three posible
options, and as target the index of the correct answer.
"""

class TextClozeImageTextVLT5Dataset(Dataset[Any]):
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
        self.keys = self.features_h5.keys()
        panel_cordinates = pd.read_csv("/data/data/datasets/COMICS/panels_cordinates.csv", delimiter=',')
        panel_cordinates['id'] = panel_cordinates[panel_cordinates.columns[:3]].apply(
            lambda x: '_'.join(x.dropna().astype(str)),
            axis=1
        )
        panel_cordinates.drop(panel_cordinates.columns[:3], axis=1, inplace=True)
        print(panel_cordinates.head())
        self.panel_cordinates = panel_cordinates.set_index('id')

    def __len__(self):
        return len(self.data)

    def get_normalized_boxes(self, image_ds, panel_cordinates=None) -> torch.Tensor:
        """
        Normalize the bounding boxes to be between 0 and 1.

        Args:
            image_ds: A dataset containing the image features.
        """
        boxes = image_ds['boxes'][()]  # (x1, y1, x2, y2)
        x1, y1, x2, y2 = panel_cordinates.values
        boxes[:, 0] -= x1
        boxes[:, 1] -= y1

        img_w = x2 - x1
        img_h = y2 - y1

        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Clip the boxes to have values between 0 and 1,
        # when normalizing there are values outside this range
        # because the bbox can be outside the panel
        boxes = np.clip(boxes, 0.0, 1.0)

        # print(boxes)
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
        per_image_mask = []

        for i in range(4):
            if i == 3:
                panel_id = sample["answer_panel_id"]
            else:
                panel_id = sample[f'context_panel_{i}_id']

            img_id = f"{book_id}_{page_id}_{panel_id}"
            if img_id in self.keys:
                image_ds = f[img_id]

                if self.panel_cordinates is not None:
                    boxes = self.get_normalized_boxes(image_ds, self.panel_cordinates.loc[img_id])
                else:
                    boxes = self.get_normalized_boxes(image_ds)

                feats = image_ds['features'][()]
                n_boxs = boxes.shape[0] # Number of boxes in the panel
                
                # Matrix of 1s for each box
                visual_mask = np.ones(shape=(n_boxs,),
                                        dtype=np.int32)
                
            else:
                n_boxs = self.config.n_boxes
                boxes = np.zeros(shape=(self.config.n_boxes, 4),
                                dtype=np.float32)
                feats = np.zeros(shape=(self.config.n_boxes, 2048),
                                dtype=np.float32)
                visual_mask = np.zeros(shape=(self.config.n_boxes,),
                                dtype=np.int32)
                
            # print(feats.shape, boxes.shape, visual_mask.shape)
                
        
            # Pad features, boxes and visual mask, with 0
            if n_boxs < self.config.n_boxes:
                boxes = np.pad(boxes, ((0, self.config.n_boxes - n_boxs), (0, 0)))
                feats = np.pad(feats, ((0, self.config.n_boxes - feats.shape[0]), (0, 0)))
                visual_mask = np.pad(visual_mask, ((0, self.config.n_boxes - n_boxs)))

            per_image_boxes.append(boxes)
            per_image_feats.append(feats)
            per_image_mask.append(visual_mask)

        boxes = np.stack(per_image_boxes)  # [4, n_boxes, 4]
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)
        out_dict['boxes'] = boxes

        feats = np.stack(per_image_feats)  # [4, n_boxes, feat_dim]
        feats = torch.from_numpy(feats)
        out_dict['vis_feats'] = feats

        masks = np.stack(per_image_mask) # [4, n_boxes]
        masks = torch.from_numpy(masks)
        out_dict['vis_mask'] = masks

        ###### Text #####
        prefix = "text cloze:"
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
            "answer0",
            sample["answer_candidate_0_text"],
            "answer1",
            sample["answer_candidate_1_text"],
            "answer2",
            sample["answer_candidate_2_text"],
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
        # input_ids = self.tokenizer.encode(
        #     input_text,
        #     max_length=self.config.max_text_length, truncation=True)

        out_dict['input_text'] = input_text
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)

        label = sample["correct_answer"]
        target_ids = self.tokenizer.encode(
            str(label), max_length=self.config.gen_max_length, truncation=True)

        assert len(target_ids) <= self.config.gen_max_length, len(target_ids)
        out_dict['label'] = label
        out_dict['target'] = target_ids[0]
        out_dict['target_ids'] = torch.LongTensor(target_ids)
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
        masks = torch.zeros(B, 4, V_L, dtype=torch.int)
        vis_feats = torch.zeros(B, 4, V_L, feat_dim, dtype=torch.float)
        target = torch.ones(B, dtype=torch.long) * \
            self.tokenizer.pad_token_id
        target_ids = torch.ones(
            B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        labels = torch.zeros(B, dtype=torch.long)

        input_texts = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            boxes[i] += entry['boxes']
            vis_feats[i] += entry['vis_feats']
            masks[i] += entry["vis_mask"]

            target[i] = entry['target']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            input_texts.append(entry['input_text'])

            labels[i] += entry['label']

        batch_entry['input_ids'] = input_ids
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['target'] = target
        batch_entry['target_ids'] = target_ids

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats
        batch_entry["vis_mask"] = masks
        batch_entry['input_text'] = input_texts
        batch_entry['labels'] = labels
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

    train_dataset = TextClozeImageTextVLT5Dataset(
        train_df, feats_h5, dataset_kwargs["tokenizer"], device, config)
    val_dataset = TextClozeImageTextVLT5Dataset(
        dev_df, feats_h5, dataset_kwargs["tokenizer"], device, config)
    test_dataset = TextClozeImageTextVLT5Dataset(
        test_df, feats_h5, dataset_kwargs["tokenizer"], device, config)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
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
