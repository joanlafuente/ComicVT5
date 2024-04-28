import argparse
import torch
import numpy as np
import importlib
import logging
import wandb

from transformers import AutoTokenizer

from src.common.registry import Registry
from src.common.configuration import get_dataset_configuration, get_model_configuration, get_trainer_configuration
from src.inference import InferenceEngine
from src.trainer_inf import Trainer
from src.tokenizers.vlt5_tokenizers import VLT5TokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="dialogue_generation_vlt5_inf_start_sent_given",
                        help='Model to run')
    parser.add_argument('--dataset_config', type=str, default="comics_dialogue_generation_textract_Blip2",
                        help='Dataset config to use')
    parser.add_argument('--trainer_config', type=str, default="adam",
                        help='Trainer params to use')
    parser.add_argument('--dataset_dir', type=str, default="/data/data/datasets/COMICS",
                        help='Dataset directory path')
    parser.add_argument('--mode', type=str, default="eval",
                        help='Execution mode ("train", "eval" or "inference")')
    parser.add_argument('--load_checkpoint', type=str, default="runs/DialogueGenerationVLT5Model_comics_dialogue_generation_1_description_panel_2024-02-05_11:53:50/models/epoch_15.pt",
                        help='Checkpoint to load')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    torch.manual_seed(0)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"SELECTED DEVICE: {device}")

    # Configuration loading
    model_config = get_model_configuration(args.model)
    Registry.register("model_config", model_config)
    dataset_config = get_dataset_configuration(args.dataset_config)
    Registry.register("dataset_config", dataset_config)

    logging.info(f"SELECTED MODEL: {model_config.classname}")
    logging.info(f"SELECTED DATASET: {dataset_config.name}")
    
    # Dataset preprocessing
    tokenizer = None
    if model_config.tokenizer:
        if model_config.tokenizer == "vlt5":
            tokenizer = VLT5TokenizerFast.from_pretrained(
                model_config.backbone,
                max_length=model_config.max_text_length,
                do_lower_case=model_config.do_lower_case,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer)

    transform = None
    if model_config.transforms:
        raise NotImplementedError("Transforms are not implemented yet.")

    dataset_kwargs = {}
    if tokenizer:
        dataset_kwargs["tokenizer"] = tokenizer

    if transform:
        dataset_kwargs["transform"] = transform

    # Model loading
    ModelClass = getattr(importlib.import_module(
        f"src.models.{args.model}"), model_config.classname)
    model = ModelClass(model_config, device).to(device)

    if tokenizer:
        model.tokenizer = tokenizer

    # Load model checkpoint
    checkpoint = None

    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint.")

        try:
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
        except Exception as e:
            logging.error("The checkpoint could not be loaded.")
            print(e)
            return
            
        model.load_checkpoint(checkpoint["model_state_dict"])

    print("Available devices:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        # TODO: Change to DistributedDataParallel
        model = torch.nn.DataParallel(model)
        model.to(device)
    
    # Print the model number of parameters
    logging.info(f"MODEL PARAMETERS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if args.mode != "inference":
        # Trainer specific configuration loading
        trainer_config = get_trainer_configuration(args.trainer_config)
        Registry.register("trainer_config", trainer_config)

        all_config_param = {**model_config.__dict__, **dataset_config.__dict__, **trainer_config.__dict__}
        
        wandb.init(project="Dialogs prediction", config=all_config_param)
        wandb.watch(model) 


        # DataLoaders
        create_dataloader = getattr(importlib.import_module(
            f"src.datasets.{dataset_config.name}"), "create_dataloader")
        train_dataloader, val_dataloader, test_dataloader = create_dataloader(
            args.batch_size,
            args.dataset_dir,
            device,
            dataset_config,
            dataset_kwargs=dataset_kwargs
        )

        trainer = Trainer(model, train_dataloader, val_dataloader,
                          test_dataloader, device, trainer_config, checkpoint)

        if args.mode == "train":
            del checkpoint
            trainer.train(trainer_config.epochs)
        elif args.mode == "eval":
            #assert checkpoint is not None, "ERROR: No checkpoint provided."
            trainer.eval()
        else:
            raise ValueError(
                f"Unknown mode: {args.mode}. Please select one of the following: train, eval, inference")

    else:
        # DataLoaders
        create_dataloader = getattr(importlib.import_module(
            f"src.datasets.{dataset_config.name}"), "create_dataloader")
        dataloader, _, _ = create_dataloader(
            args.batch_size,
            args.dataset_dir,
            device,
            dataset_config,
            inference=True,
            dataset_kwargs=dataset_kwargs
        )

        inference_engine = InferenceEngine(model, device)
        inference_engine.run(dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
