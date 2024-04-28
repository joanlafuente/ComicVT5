import os
import torch
import logging
import h5py as h5

from tqdm import tqdm
from typing import Any
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_outputs import Seq2SeqLMOutput


class InferenceEngine:
    """
    Inference Engine
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        """
        Constructor of the InferenceEngine.

        Args:
            model: The model to use.
            device: The device to use.
        """
        self.model = model
        self.device = device
        self.output_dir = "inference_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dataloader: DataLoader[Any]) -> None:
        """
        Run the inference engine through the dataloader and
        save the results to the output directory.

        Args:
            dataloader: The dataloader to use.
        """
        logging.info(f"Running inference on dataset.")
        self.model.eval()

        h5_file = h5.File(os.path.join(
            self.output_dir, "inference_output.h5"), "w")
        features = h5_file.create_group("features")

        for local_batch in tqdm(dataloader):
            sample_ids = local_batch["sample_id"]
            batch = local_batch["data"]
            batch_len = len(sample_ids)

            with torch.no_grad():
                batch_output = self.model(**batch)

            for i in range(batch_len):
                sample_id = sample_ids[i]

                if type(batch_output) is BaseModelOutputWithPooling:
                    output = batch_output.pooler_output[i]
                elif type(batch_output) is Seq2SeqLMOutput:
                    raise NotImplementedError(
                        "Seq2SeqLMOutput is not implemented yet.")
                else:
                    logging.warning(
                        f"Unhandled output type: {type(batch_output)}")
                    output = batch_output[i]

                output = output.detach().cpu().numpy()

                comic_no, page_no, panel_no = sample_id.split("_")
                comic_no = int(comic_no)
                page_no = int(page_no)
                panel_no = int(panel_no)

                if str(comic_no) not in features:
                    comic_group = features.create_group(str(comic_no))
                else:
                    comic_group = features[str(comic_no)]

                if str(page_no) not in comic_group:
                    page_dataset = comic_group.create_dataset(
                        str(page_no),
                        shape=(1, *output.shape),
                        maxshape=(None, *output.shape),
                        dtype=output.dtype
                    )
                else:
                    page_dataset = comic_group[str(page_no)]

                page_dataset.resize(panel_no + 1, axis=0)
                page_dataset[panel_no] = output

                save_path = os.path.join(self.output_dir, f"{sample_id}.pt")
                torch.save(output, save_path)
