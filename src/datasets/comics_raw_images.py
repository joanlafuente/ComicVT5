import torch
import glob
import os
import pandas as pd
import numpy as np

from PIL import Image
from typing import Any, Tuple, List, Optional
from torch.utils.data import DataLoader
from transformers import BeitFeatureExtractor

from src.common.sample import Sample
from src.datasets.base_dataset import BaseDataset

"""
Dataset class in order to load the raw images of the comics
with the text masked using the OCR bounding boxes.
"""
class ComicsRawImages(BaseDataset):

    def __init__(self,
                 image_paths: List[str],
                 ocr_file: pd.DataFrame,
                 device: torch.device,
                 config: Any,
                 transform: Optional[Any] = None,
                 feature_extractor: Optional[BeitFeatureExtractor] = None
                 ):
        super().__init__(device, config)
        self.image_paths = image_paths
        self.ocr_file = ocr_file
        self.transform = transform
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def obfuscate_bounding_boxes(self,
                                 image: Image.Image,
                                 bounding_boxes: List[
                                     Tuple[int, int, int, int]]
                                 ) -> None:
        """
        Obfuscates the bounding boxes in the image.

        Args:
            image: Image to obfuscate.
            bounding_boxes: List of bounding boxes to obfuscate.
        """
        for x1, y1, x2, y2 in bounding_boxes:
            image.paste(0, (x1, y1, x2, y2))

    def getitem(self, idx: int) -> Sample:
        image_path = self.image_paths[idx]

        # Generates the id of the sample from the name of the image
        # and the parent directory.
        comic_no = os.path.basename(os.path.dirname(image_path))
        page_no, panel_no = os.path.basename(
            image_path).split(".")[0].split("_")
        sample_id = f"{comic_no}_{page_no}_{panel_no}"

        image = Image.open(image_path)
        image = image.convert("RGB")

        # Get all the bounding boxes for the current sample.
        # Filter by the comic number, page number and panel number.
        filtered_rows = self.ocr_file.loc[
            (self.ocr_file["comic_no"] == int(comic_no)) &
            (self.ocr_file["page_no"] == int(page_no)) &
            (self.ocr_file["panel_no"] == int(panel_no))
        ]
        bounding_boxes = filtered_rows[[
            "x1", "y1", "x2", "y2"]].dropna().values.astype("int").tolist()

        # Obfuscate the bounding boxes.
        self.obfuscate_bounding_boxes(image, bounding_boxes)

        if self.transform:
            image = self.transform(image)
        elif self.feature_extractor:
            image = self.feature_extractor(image, return_tensors="pt")
            image["pixel_values"] = image["pixel_values"].squeeze(0)
        else:
            raise ValueError(
                "Either transform or feature_extractor must be provided")

        return Sample(sample_id, {
            "image": image,
        })


def create_dataloader(
    batch_size: int,
    dataset_path: str,
    device: torch.device,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
) -> Tuple[DataLoader[Any], Optional[DataLoader[Any]], Optional[DataLoader[Any]]]:
    image_paths = glob.glob(os.path.join(
        dataset_path, "**/*.jpg"), recursive=True)
    ocr_file = pd.read_csv(os.path.join(
        dataset_path, "COMICS_ocr_file.csv"))

    dataset = ComicsRawImages(image_paths, ocr_file,
                              device, config, **dataset_kwargs)

    if not inference:
        # Split the dataset into train, validation, and test
        train_size = int(0.8 * len(image_paths))
        val_size = int(0.1 * len(image_paths))
        test_size = len(image_paths) - train_size - val_size
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        val_dataloader = test_dataloader = None
        train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, val_dataloader, test_dataloader
