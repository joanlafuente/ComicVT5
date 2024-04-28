import os
import yaml
from typing import Any
from munch import DefaultMunch


def get_configuration(yaml_path: str) -> Any:
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
        return DefaultMunch.fromDict(yaml_dict)


def get_model_configuration(yaml_path: str) -> Any: 
    yaml_path = os.path.join("configs/models", yaml_path + ".yaml")
    return get_configuration(yaml_path)
    

def get_dataset_configuration(yaml_path: str) -> Any: 
    yaml_path = os.path.join("configs/datasets", yaml_path + ".yaml")
    return get_configuration(yaml_path)


def get_trainer_configuration(yaml_path: str) -> Any: 
    yaml_path = os.path.join("configs/trainers", yaml_path + ".yaml")
    return get_configuration(yaml_path)