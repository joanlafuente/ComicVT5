import torch

from typing import Any
from torch.utils.data import Dataset

from src.common.sample import Sample


class BaseDataset(Dataset[Any]):
    """
    Base Dataset.
    """

    def __init__(self,
                 device: torch.device,
                 config: Any,
                 ) -> None:
        """
        Constructor of the Dataset.
        """
        self.device = device
        self.config = config

    def getitem(self, idx: int) -> Sample:
        """
        Returns the item at the given index as a Sample object.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Sample:
        """
        Returns the item at the given index.

        .. warning::
            This method must not be overridden.
        """
        sample = self.getitem(idx)
        return sample.to(self.device)
        



