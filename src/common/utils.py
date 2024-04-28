"""
Some functions from https://github.com/ArjanCodes/2021-data-science-refactor/blob/main/after/ds/utils.py
"""
import pathlib
import datetime

from typing import List, Tuple

from src.common.registry import Registry


def create_experiment_dir(root: str, experiment_name: str,
                          parents: bool = True) -> Tuple[str, str]:
    root_path = pathlib.Path(root).resolve()
    # child = (
    #     create_from_missing(root_path, experiment_name)
    #     if not root_path.exists()
    #     else create_from_existing(root_path, experiment_name)
    # )
    child = root_path / experiment_name
    child.mkdir(parents=parents)
    models_path = child / "models"
    models_path.mkdir()
    return child.as_posix(), models_path.as_posix()


def create_from_missing(root: pathlib.Path,
                        experiment_name: str = "") -> pathlib.Path:
    return root / f"0-{experiment_name}"


def create_from_existing(root: pathlib.Path,
                         experiment_name: str = "") -> pathlib.Path:
    children = [
        int(c.name.split("-")[0]) for c in root.glob("*")
        if (c.is_dir() and c.name.split("-")[0].isnumeric())
    ]
    if is_first_experiment(children):
        child = create_from_missing(root, experiment_name)
    else:
        child = root / \
            f"{increment_experiment_number(children)}-{experiment_name}"
    return child


def is_first_experiment(children: List[int]) -> bool:
    return len(children) == 0


def increment_experiment_number(children: List[int]) -> str:
    return str(len(children) + 1)


def generate_experiment_name():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_config = Registry.get("model_config")
    dataset_config = Registry.get("dataset_config")
    experiment_name = f"{model_config.classname}_{dataset_config.name}_{now}"
    return experiment_name
