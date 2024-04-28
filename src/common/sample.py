
import logging
import torch

from collections import OrderedDict


class Sample(OrderedDict):
    """
    Sample class is a data structure that contains the data of a sample.

    Every dataset will return a Sample object when calling the __getitem__ method.
    """

    def __init__(self, sample_id: str, data: dict) -> None:
        """
        Initialize a Sample object.

        Args:
            sample_id: The id of the sample.
            data: The data of the sample.
        """
        self.sample_id = sample_id
        self.data = data

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def to(self, device: torch.device):
        """
        Move the sample data to the specified device.

        Args:
            device: The device to move the sample to.

        Returns:
            The sample data moved to the specified device.
        """
        new_sample = Sample(self.sample_id, {})

        for key, value in self.data.items():
            if type(value) is torch.Tensor:
                new_sample.data[key] = value.to(device)
            elif (
                type(value) is dict
                or type(value) is OrderedDict
            ):
                new_sample.data[key] = {k: v.to(device)
                                        for k, v in value.items()}
            else:
                # logging.warning(
                #     f"Value of type {type(value)} could not be moved to device.", 
                #     norepeat=True
                #     )
                new_sample.data[key] = value

        return new_sample
