# coding=utf-8

"""
Base code from Unifying Vision-and-Language Tasks via Text Generation
(https://github.com/j-min/VL-T5)
"""

import argparse
import pandas as pd
import cv2
import os
import glob

from typing import Any, List, Tuple
from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ComicsRawImagesDataset(Dataset):

    def __init__(self,
                 image_paths: List[str],
                 ocr_file: pd.DataFrame,
                 ):
        self.image_paths = image_paths
        self.ocr_file = ocr_file

    def __len__(self):
        return len(self.image_paths)

    def obfuscate_bounding_boxes(self,
                                 image: Any,
                                 bounding_boxes: List[
                                     Tuple[int, int, int, int]]
                                 ) -> Any:
        """
        Obfuscates the bounding boxes in the image.

        Args:
            image: Image to obfuscate.
            bounding_boxes: List of bounding boxes to obfuscate.
        """
        for x1, y1, x2, y2 in bounding_boxes:
            start_point = (x1, y1)
            end_point = (x2, y2)
            image = cv2.rectangle(image, start_point, end_point, (0, 0, 0), -1)

        return image

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]

        # Generates the id of the sample from the name of the image
        # and the parent directory.
        comic_no = os.path.basename(os.path.dirname(image_path))
        page_no, panel_no = os.path.basename(
            image_path).split(".")[0].split("_")
        sample_id = f"{comic_no}_{page_no}_{panel_no}"

        image = cv2.imread(image_path)

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
        image = self.obfuscate_bounding_boxes(image, bounding_boxes)

        return {
            "img": image,
            "img_id": sample_id,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--dataset_path', type=str,
                        default='/data/data/datasets/COMICS',)
    parser.add_argument('--out_dir', type=str,
                        default='/data/data/datasets/COMICS/frcnn_features',)

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        out_dir.mkdir()

    dataset_path = args.dataset_path
    image_paths = glob.glob(os.path.join(
        dataset_path, "panels", "**/*.jpg"), recursive=True)
    ocr_file = pd.read_csv(os.path.join(
        dataset_path, "COMICS_ocr_file.csv"))

    print('Load images from', dataset_path)

    dataset = ComicsRawImagesDataset(image_paths, ocr_file)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)
    desc = f'COMICS_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)