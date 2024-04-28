import argparse
import pandas as pd
import cv2
import os
import glob
import torch
from thop import profile
from torchvision.models.resnet import resnet50
from torch import nn
from torchvision import transforms
from PIL import Image

from tqdm import tqdm
from typing import Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ComicsRawImagesDataset(Dataset):
    """
    Dataset class for loading raw images.
    """
    def __init__(self,
                 image_paths: List[str],
                 ocr_file: pd.DataFrame,
                 ):
        self.image_paths = image_paths
        self.ocr_file = ocr_file
        self.transform = transforms.Compose([transforms.Resize((112, 112)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]

        # Generates the id of the sample from the name of the image
        # and the parent directory.
        comic_no = os.path.basename(os.path.dirname(image_path))
        page_no, panel_no = os.path.basename(
            image_path).split(".")[0].split("_")
        sample_id = f"{comic_no}_{page_no}_{panel_no}"

        image = cv2.imread(image_path)


        # Obfuscate the bounding boxes.
        image = Image.fromarray(image)
        image = self.transform(image)

        return {
            "img": image,
            "img_id": sample_id,
        }
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=50, type=int, help='batch_size')
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

    dataset = ComicsRawImagesDataset(image_paths, ocr_file)
    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, num_workers=4)
    
    # Load the resnet50 model trained with SimCLR
    feature_extractor = Model_SimCLR().to("cuda:1")
    profile(feature_extractor, inputs=(torch.randn(1, 3, 112, 112).to("cuda:1"),))
    feature_extractor.load_state_dict(torch.load("/home/jlafuente/ComicsQA/Models/test_model_25_epochs_730k_img.pt"))
    feature_extractor.eval()

    list_features = []
    list_ids = []
    print('Extracting features...')
    for batch in tqdm(dataloader):
        imgs = batch['img'].to("cuda:1")
        imgs_ids = batch['img_id']
        list_ids.extend(imgs_ids)
        features = feature_extractor.encode(imgs)
        features = features.to("cpu").detach().numpy()
        list_features.extend(features)

    dict_features = dict(zip(list_ids, list_features))


    print('Saving features...')
    # Saving the dictionary into a pickle
    import pickle
    with open(os.path.join(out_dir, 'Sim_CLR_panels_features_not_masked.pkl'), 'wb') as f:
        pickle.dump(dict_features, f)