# Visual features extraction

To extract the visual features for the model we have tested different posibilities. Here there is the code to extract the features using the different implementations.


* **comics_proposal.py**: Extract features of the detected objects using detectron2 with a Resnet50 trained with SimCLR.
* **get_features_SAM_BB.py**: Extract features of the detected objects using Segment anything (Previously detected) with a Resnet50 trained with SimCLR.
* **extract_features_VIT.py**: Extract features of the panels using a pretrained VIT from pytorch.
* **Resnet50_panels_features.py**: Extract features of the panels using a pretrained Resnet50 from pytorch.
* **extract_features_blip2.py**: Extract features of the panels using Blip2 qformer features.
* **extract_features_blip2_not_mask_text.py**: Extract features of the panels using Blip2 qformer features, but not masking the text on the panels.
* **Sim_CLR_panels_features_finetuning.py**: Extract features of the panels using a finetuned Resnet50 with SimCLR.
* **Sim_CLR_panels_features_not_mask_text.py**: Extract features of the panels using a Resnet50 trained with SimCLR, but not masking the text on the panels.


## How to extract features in the detectron approach
As [VL-T5 authors](https://github.com/j-min/VL-T5) do, we tested using [Hao Tan's Detectron2 implementation of 'Bottom-up feature extractor'](https://github.com/airsplay/py-bottom-up-attention) to detect the objects on panels.

### 1. Install Detectron2

Please follow [the original installation guide](https://github.com/airsplay/py-bottom-up-attention#installation).

*We strongly recommend that you create a new Conda environment for this task, we provide the enviroment that we used on detectron2.yml.*


### 2. Download the panel images

If you have not done so, please download the extracted panel images from [here](https://obj.umiacs.umd.edu/comics/index.html).


### 3. Manually extract & convert features

Just run the following command:

```sh
python comics_proposal.py 
    --batch_size=<batch size> 
    --dataset_path=<path to the dataset (defaults to 'datasets/COMICS/data')> 
    --out_dir=<path to the output directory (defaults to 'datasets/COMICS/frcnn_features')>
```
