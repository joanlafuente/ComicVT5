import torch
import pandas as pd

from typing import Any, Tuple
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from src.common.sample import Sample
from src.datasets.base_dataset import BaseDataset

"""
Dataset class thought for doing an experiment in which we 
do not provide visual information.

As input text we provide the context text and the three posible 
options, and as target the index of the correct answer.

This was done to test the importance of the visual information
in this task.
"""

class ComicsOcrOnlyDataset(BaseDataset):

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 device: torch.device,
                 config: Any
                 ):
        super().__init__(device, config)
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def getitem(self, idx: int) -> Sample:
        sample = self.data.iloc[idx]
        context = [
            "context00",
            sample["context_text_0_0"],
            "context01",
            sample["context_text_0_1"],
            "context02",
            sample["context_text_0_2"],
            "context10",
            sample["context_text_1_0"],
            "context11",
            sample["context_text_1_1"],
            "context12",
            sample["context_text_1_2"],
            "context20",
            sample["context_text_2_0"],
            "context21",
            sample["context_text_2_1"],
            "context22",
            sample["context_text_2_2"],
        ]

        # blank_random_index = torch.randint(9, (1,))
        # context[blank_random_index] = ""
        context = ["text cloze :"] + context

        context = self.tokenizer(context, return_tensors="pt", truncation=True,
                                 max_length=self.config.context_max_speech_size, padding="max_length").input_ids

        answers = [
            "answer0",
            sample["answer_candidate_0_text"],
            "answer1",
            sample["answer_candidate_1_text"],
            "answer2",
            sample["answer_candidate_2_text"],
        ]

        answers = self.tokenizer(answers, return_tensors="pt", truncation=True,
                                 max_length=self.config.answer_max_tokens, padding="max_length").input_ids

        target = torch.zeros(3)
        target[sample["correct_answer"]] = 1.0

        permutation = torch.randperm(3)
        answers = answers[permutation]
        target = target[permutation]

        return Sample(str(idx), {
            "context": context,
            "answers": answers,
            "target":  torch.argmax(target, dim=0)
        })


def create_dataloader(
    batch_size: int,
    dataset_path: str,
    device: torch.device,
    config: Any,
    inference: bool = False,
    dataset_kwargs: dict = {},
) -> Tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    assert not inference, "This dataset cannot be used for inference."

    train_df = pd.read_csv(
        f"{dataset_path}/text_cloze_train_{config.mode}.csv", delimiter=',')
    train_df = train_df.fillna("")
    dev_df = pd.read_csv(
        f"{dataset_path}/text_cloze_dev_{config.mode}.csv", delimiter=',')
    dev_df = dev_df.fillna("")
    test_df = pd.read_csv(
        f"{dataset_path}/text_cloze_test_{config.mode}.csv", delimiter=',')
    test_df = test_df.fillna("")

    train_dataset = ComicsOcrOnlyDataset(
        train_df, dataset_kwargs["tokenizer"], device, config)
    val_dataset = ComicsOcrOnlyDataset(
        dev_df, dataset_kwargs["tokenizer"], device, config)
    test_dataset = ComicsOcrOnlyDataset(
        test_df, dataset_kwargs["tokenizer"], device, config)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_dataloader, val_dataloader, test_dataloader
