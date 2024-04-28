import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw
from torchvision import transforms as T
import h5py
from torchvision.models.resnet import resnet50
from torch import nn
import torch
from thop import profile
import ctypes
# To free memory not used anymore
# If not used, it keeps increasing the memory used until it crashes
libc = ctypes.CDLL("libc.so.6")

path_objects_bbox = '/data/data/datasets/COMICS/bboxes/'
path_pages = '/data/data/datasets/COMICS/books'
out_path = '/data/data/datasets/COMICS'
name_h5f5 = 'test_bb_SAM_ofuscate'
panels_cordinates_path = '/data/data/datasets/COMICS/panels_cordinates.csv'
path_model = '/home/jlafuente/ComicsQA/Models/test_model_25_epochs_730k_img.pt' # Path to the trained model using SimCLR
ocr_path = '/data/data/datasets/COMICS/COMICS_ocr_file_updated.csv'

max_intersection_other_panels = 0.05
min_intersection_panel = 0.8
min_area_object = 150
max_number_objects = 36

# Transform to apply to the images
transform = T.Compose([T.Resize((112, 112)),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])

def overlaping_area_norm(r1, r2):
    # The overlaping area is normalized by the area of the first rectangle
    x1, y1, x2, y2 = r1
    x3, y3, x4, y4 = r2
    dx = min(x2, x4) - max(x1, x3)
    dy = min(y2, y4) - max(y1, y3)
    if (dx>=0) and (dy>=0):
        return (dx*dy)/(abs(x2-x1)*abs(y2-y1))
    else:
        return 0

def return_intersected_panels(bbox, panel_cordinates_page):
    # Returns the panels that intersect with the bbox more than a 5%

    # To change from width and height to x2 and y2
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]
    dict_panels = {}
    for row in panel_cordinates_page.itertuples():
        bbox_panel = [int(i) for i in [row[1], row[2], row[3], row[4]]]
        overlap = overlaping_area_norm(bbox, bbox_panel)
        if overlap > 0.05:
            dict_panels[row[0]] = overlap
    
    return dict_panels

class Model_SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model_SimCLR, self).__init__()

        list_modules = []
        for name, module in resnet50().named_children():
            list_modules.append(module)
        # encoder
        self.encoder = nn.Sequential(*list_modules[:-1])

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
    # Load the SimCLR trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = Model_SimCLR().to(device)
    profile(feature_extractor, inputs=(torch.randn(1, 3, 200, 200).to(device),))
    feature_extractor.load_state_dict(torch.load(path_model))
    feature_extractor.eval()

    # Load the dataset with panel cordinates
    panels_cordinates = pd.read_csv(panels_cordinates_path, sep=',')
    panels_cordinates.set_index(["comic_no","page_no","panel_no"], inplace=True)

    # Load the dataset with the OCR information
    ocr = pd.read_csv(ocr_path, sep=',')
    ocr.set_index(["comic_no","page_no","panel_no", "textbox_no"], inplace=True)
    
    # Get the list of books
    books = os.listdir(path_objects_bbox)
    books = sorted([int(i) for i in books if i != "test_bb_SAM.hdf5"])
    # Create the h5 file
    with h5py.File(f'{out_path}/{name_h5f5}.hdf5', 'w') as f:
        keys = list(f.keys())
        for book_id in tqdm(books):
            # Get the list of pages
            pages = os.listdir(path_objects_bbox + str(book_id))
            pages = sorted([int(i.split("_")[0]) for i in pages])
            for page_id in pages:
                try:
                    # Load the csv with the objects on the page
                    csv_page = pd.read_csv(f"{path_objects_bbox}/{book_id}/{page_id}_anns.csv", sep=',')
                    # Load the image of the page
                    img = Image.open(f"{path_pages}/{book_id}/{page_id}.jpg").convert('RGB')
                    width_img, height_img = img.size
                    
                    # To remove the text from the image
                    ocr_page = ocr.loc[book_id, page_id, :, :]
                    draw = ImageDraw.Draw(img)
                    for text_info in ocr_page.itertuples(index=False):
                        x1, x2, y1, y2 = text_info.x1, text_info.x2, text_info.y1, text_info.y2
                        draw.rectangle(((x1, y1), (x2, y2)), fill="black")
                    
                    panel_cordinates_page = panels_cordinates.loc[book_id, page_id, :]
                    # Iterate over the objects in the page
                    panel_objects = {}
                    for row in csv_page.itertuples():
                        stability_score = row[5]
                        bbox = row[2]
                        bbox = [float(i) for i in bbox[1:-1].split(',')]
                        width = bbox[2]
                        height = bbox[3]
                        # If the object is big enough
                        if width*height > min_area_object:
                            # Check if the object is inside a panel
                            dict_intersection = return_intersected_panels(bbox, panel_cordinates_page)
                            # If the object is only inside one panel and the intersection is big enough
                            if (len(dict_intersection) == 1) and (max(dict_intersection.values()) > min_intersection_panel):
                                panel = list(dict_intersection.keys())[0]
                            else:
                                panel = None

                            if panel is not None:
                                # If there was not any object in the panel, create the list
                                if panel not in panel_objects.keys():
                                    panel_objects[panel] = []
                                # Add the bbox and the stability score to the list of objects in the panel
                                panel_objects[panel].append((bbox, stability_score))

                    # Iterate over the panels and objects in the page
                    for panel, bboxs_stability in panel_objects.items():
                        id_panel = f"{book_id}_{page_id}_{panel}"
                        # If the panel is not in the h5 file, add it
                        if id_panel not in keys:
                            # Get number the objects with the highest stability score up to max_number_objects
                            bbox_sorted = [bbox for bbox, stability in sorted(bboxs_stability, key=lambda x: x[1], reverse=True)][:max_number_objects]

                            # Save the features and the bounding boxes in the h5 file
                            with torch.no_grad():
                                imgs = torch.stack([transform(img.crop(bbox)) for bbox in bbox_sorted]).to(device)
                                features = feature_extractor.encode(imgs)
                                del imgs
                            grp = f.create_group(id_panel)
                            grp['features'] = features.cpu().numpy()  # [num_objects, 2048]
                            grp['boxes'] = bbox_sorted # [num_objects, 4]
                            grp['img_w'] = width_img
                            grp['img_h'] = height_img
                            f.flush()

                    # Free memory
                    del csv_page
                    del panel_cordinates_page
                    del img
                    del panel_objects
                    libc.malloc_trim(0)
                            
                # If there is an error, continue with the next page
                except Exception as e:
                    print(f"Error in book {book_id} page {page_id}")
                    print(e)
                    continue
            