import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model

device = "cuda" if torch.cuda.is_available() else "cpu"


import argparse
import pandas as pd
import cv2
import os
import glob
import torch
from thop import profile
from torch import nn
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
from typing import Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



class ComicsRawImagesDataset(Dataset):
    """
    Dataset class for loading raw images, with the text masked from the COMICS dataset.
    """
    def __init__(self,
                 image_paths: List[str],
                 ocr_file: pd.DataFrame,
                 processor: Blip2Processor,
                 ):
        self.image_paths = image_paths
        self.ocr_file = ocr_file
        self.processor = processor

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
        image = Image.fromarray(image)
        input_blip = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return {
            "input": input_blip,
            "img_id": sample_id,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=5, type=int, help='batch_size')
    parser.add_argument('--dataset_path', type=str,
                        default='/data/data/datasets/COMICS/',)
    parser.add_argument('--out_dir', type=str,
                        default='/data/data/datasets/COMICS/Sim_CLR_panels_features',)

    args = parser.parse_args()

    # Create the output directory if it does not exist
    out_dir = Path(args.out_dir).resolve()
    if not out_dir.exists():
        out_dir.mkdir()

    # Get the list of paths to the images
    dataset_path = args.dataset_path
    image_and_book_names = []
    books_paths = os.listdir(os.path.join(dataset_path, "panels"))
    for book_path in books_paths:
        image_and_book_names.extend([os.path.join(book_path, name) for name in os.listdir(os.path.join(dataset_path, "panels", book_path))])
    
    image_paths = [os.path.join(dataset_path, "panels", image_name) for image_name in image_and_book_names]

    ocr_file = pd.read_csv(os.path.join(dataset_path, "COMICS_ocr_file.csv"))

    print('Load images from', dataset_path)

    from torch import nn
    import torch

    # Load the Blip2 model
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    feature_extractor = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    feature_extractor.to(device)


    dataset = ComicsRawImagesDataset(image_paths, ocr_file, processor)
    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, num_workers=4)
    
    

    list_features = []
    list_ids = []
    print('Extracting features...')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_blip = {}
            input_blip["pixel_values"] = batch['input'].to(device, torch.float16)
            imgs_ids = batch['img_id']
            list_ids.extend(imgs_ids)
            # Extract the qformer features of blip2 for the input images
            features = feature_extractor.get_qformer_features(**input_blip)
            features = features["pooler_output"] # A tensor of shape (batch_size, 768)
            features.resize_(args.batchsize, 768)
            features = features.to("cpu").detach().numpy()
            list_features.extend(features)

    dict_features = dict(zip(list_ids, list_features))


    print('Saving features...')
    # Saving the dictionary into a pickle
    import pickle
    with open(os.path.join(out_dir, 'blip2_panels_features.pkl'), 'wb') as f:
        pickle.dump(dict_features, f)
