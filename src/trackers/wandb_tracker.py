import json
import os
import numpy as np
import torch
import wandb

from pathlib import Path
from typing import List, Tuple
from matplotlib import pyplot as plt

from src.trackers.tracker import Stage
from src.common.registry import Registry


class WandbTracker:
    """
    Creates a tracker that implements the ExperimentTracker protocol and logs to wandb.
    """

    def __init__(self) -> None:
        wandb.init(project="comics-dialogue-generation")
        self.stage = Stage.TRAIN

    def set_stage(self, stage: Stage):
        self.stage = stage

    

    
        
    

