# Multimodal Transformer for Comics Text-Cloze
* Authors: Emanuele Vivoli*, Joan Lafuente Baeza*, Ernest Valveny Llobet and Dimosthenis Karatzas
* Contact: evivoli@cvc.uab.cat, joan.lafuente@autonoma.cat
* The paper is available [here](https://arxiv.org/pdf/2403.03719.pdf) for more information.

## Example
In the next figure it can be seen two instances of the text-cloze task, as well as an example of the dialogue generation task.
![Example_task](./assets/example.png)


## Project Description
The purpose of this project is to perform a text-cloze task in the multimodal context of comics. To do this, we used the database provided in [COMICS](https://github.com/miyyer/comics). We have also explored the posibility of dialogues generation.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).

```
git clone https://github.com/joanlafuente/ComicVT5.git
cd ComicVT5
```

2. Download and prepare the dataset.  

    a. Download our preprocessed version of the original dataset as well as a version with textract OCR from [here](https://uab-my.sharepoint.com/:f:/g/personal/1523726_uab_cat/EtPWggOeDjpAvDKV3b3zb1oBDheUMHDOT1KBGmKeofiK8g?e=calKi6) and place it in ```datasets/COMICS```. If you would like to use the original dataset, you can download it from the [COMICS repository](https://github.com/miyyer/comics).

    *Note that there are two files for the test set. The original authors filtered some dialogues from this set based on their tokenizer. To be able to compare the results of our model with the original ones, we maintain this filter in the test set. The "full" version, on the other hand, is not filtered.*

    b. Extract the visual features using one of the explained methods [here](https://github.com/joanlafuente/ComicVT5/blob/main/tools/extract_visual_features), within this repo.

3. Install the dependencies.

```sh
# Create a conda environment with the depnedencies needed (optional but recommended)
conda env create -f environment.yml
```

4. [Optional] Download the pre-trained weights of the best performing models, on the easy and hard task, and CRN-Scratch from [here](https://uab-my.sharepoint.com/:f:/g/personal/1523726_uab_cat/EmVN5Cd11QtEtrcgV6Hz3mQBLnyfL3WtD-26mEAt7BiyNg?e=Y7OUbl). If you use the VL-T5 model, download the pretrained weights from their repository [here](https://github.com/j-min/VL-T5).


## Configuration

Every model, dataset, and trainer is configured in a configuration file. The configuration file is a YAML file. The configuration files are located in the ```configs``` folder. In case you want to add a new model, dataset, or trainer, you should create a new configuration file and add it to the ```configs``` folder, as well as the corresponding model or dataset script in ```src```.

### Dataset configuration
For each dataset, you need a configuration file in the ```configs/datasets``` folder. The file must contain the "name" parameter, which is the same as the name of the dataset script in ```src/datasets``` that will be used to load the dataset.

### Model configuration
For each model, you need a configuration file in the ```configs/models``` folder. The name of the file must be the same as the name of the model script in ```src/models``` that will be used to load the model. The file must contain the the following parameters:
``` YAML
classname: <class name of the model>

tokenizer: <name of the tokenizer (we use the AutoTokenizer class from HuggingFace)>
# or
feature_extractor: <name of the feature extractor>
```

### Trainer configuration
For the trainer, you need a configuration file in the ```configs/trainers``` folder. The file must contain the the following parameters:

``` YAML
epochs: <number of epochs>
runs_path: <path to the runs folder>
report_path: <path to the report folder>

optimizer:
    type: <type of optimizer>
    # ... parameters of the optimizer
```

The runs folder is where the training logs will be saved. The report folder is where the evaluation reports will be saved.


## Code structure

```sh
# Run dataset setup and feature extraction
./tools
    extract_visual_features/
    setup_dataset/

# Store dataset
./datasets
    COMICS/
        frcnn_features/
            boxes36.h5
        text_cloze_train_easy.csv
        ...
    ...

# Store pre-trained weights
./pretrained_weights
    vlt5_epoch30.pth           <= The VL-T5 model expects this file to be in this folder (It can be changed in its config file).
    ...

# Store configuration
./config
    datasets/
    models/
    trainers/

# Create you own models or datasets
.src/
    models/                    <= This folder is where it is expected to add new models.
        base_model.py
    datasets/                  <= This folder is where it is expected to add new datasets.
        base_dataset.py

# Run the model
./main.py
# A version of "main.py" to evaluate dialogue generation models
./main_inf.py

# A notebook to generate examples of the text-cloze models
./plot_samples.ipynb 
# A notebook to generate examples of dialogue generation models
./plot_samples_gen.ipynb 
```


## Training and evaluation

To train the model, run the following command:

```sh
python main.py
  --mode "train"
  --model               Model to run
  --dataset_config      Dataset config to use
  --trainer_config      Trainer params to use
  --dataset_dir         Dataset directory path
  --load_checkpoint     Path to model checkpoint
  --batch_size          Batch size
  --seed                Seed to use
```

To evaluate the model, change the ```--mode``` to "eval".


## Reference
```sh
@misc{vivoli2024multimodal,
      author = {Emanuele Vivoli and Joan Lafuente Baeza and Ernest Valveny Llobet and Dimosthenis Karatzas},
       title = {Multimodal Transformer for Comics Text-Cloze}, 
        year = {2024},
         url = {https://arxiv.org/pdf/2403.03719}
}
```
