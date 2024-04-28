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
from src.trainer import Trainer
from src.tokenizers.vlt5_tokenizers import VLT5TokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="text_cloze_image_text_vlt5",
                        help='Model to run')
    parser.add_argument('--dataset_config', type=str, default="text_cloze_image_text_SimCLR_hard_textract_vlt5_not_mask_context_panels_nor_input_context_text",
                        help='Dataset config to use')
    parser.add_argument('--trainer_config', type=str, default="adam",
                        help='Trainer params to use')
    parser.add_argument('--dataset_dir', type=str, default="/data/data/datasets/COMICS",
                        help='Dataset directory path')
    parser.add_argument('--mode', type=str, default="eval",
                        help='Execution mode ("train", "eval" or "inference")')
    parser.add_argument('--load_checkpoint', type=str, default="/home/jlafuente/Comics_dialogs_generation/runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_BLIP2_not_masked_context_not_context_text_input_2024-02-10_21:07:39/models/epoch_1.pt",
                        help='Checkpoint to load')
    parser.add_argument('--batch_size', type=int, default=15,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed to use')

    args = parser.parse_args()
    return args


# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_BLIP2_not_masked_context_not_context_text_input_2024-02-10_00:10:31/models/epoch_8.pt", # Not context text SimCLR easy

# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_BLIP2_not_masked_context_not_context_text_input_2024-02-04_08:34:01/models/epoch_4.pt", # Not context text blip2 hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_BLIP2_not_masked_context_not_context_text_input_2024-02-02_21:34:55/models/epoch_11.pt", # Not context text blip2 easy
# default="runs/DialogueGenerationVLT5Model_comics_dialogue_generation_with_posible_answers_2024-02-03_23:49:55/models/epoch_3.pt", # Gen hard blip2 with answers
# default="runs/DialogueGenerationVLT5Model_comics_dialogue_generation_with_posible_answers_2024-02-02_16:45:59/models/epoch_10.pt", # Gen easy blip2 with answers
# default="runs/DialogueGenerationVLT5Model_comics_dialogue_generation_with_posible_answers_2024-02-01_15:31:03/models/epoch_3.pt", # Gen hard with answers 
# default="runs/DialogueGenerationVLT5Model_comics_dialogue_generation_with_posible_answers_2024-01-31_13:49:00/models/epoch_6.pt", # Gen easy with answers real epoch +3
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2024-01-30_23:12:52/models/epoch_8.pt", # SimCLR model with linear before encoder easy
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_no_prefix_2024-01-29_23:50:15/models/epoch_4.pt", # Encoder only - ResnetSimCLR - easy - no prefix - no Batch norm, drop head input and in 15 + 4 epochs
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_no_prefix_2024-01-29_12:59:05/models/epoch_4.pt", # Encoder only - ResnetSimCLR - easy - no prefix - Batch norm no drop head 15 epochs + 4 epochs
# +0 seed
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2024-01-20_15:53:03/models/epoch_8.pt", # Encoder_decoder - Resnet - hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2024-01-21_08:53:31/models/epoch_3.pt", # Encoder_decoder - VIT - hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_head_2024-01-20_15:45:20/models/epoch_2.pt", # Head - simclr - hard   
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_head_2024-01-19_13:05:11/models/epoch_8.pt", # Head - simclr - easy
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_head_2024-01-19_12:47:43/models/epoch_4.pt", # Head - blip2 - hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_head_2023-12-22_18:27:11/models/epoch_10.pt" # Head -blip2 - easy
# default="/home/jlafuente/Comics_dialogs_generation/runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-12-18_10:26:20/models/epoch_2.pt", # Objects sam simclr hard epoch2
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-12-13_11:22:31/models/epoch_3.pt", # Textract + blip2 hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-12-11_11:49:19/models/epoch_4.pt", # SAM objects easy - SIMCLR
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-12-11_10:13:25/models/epoch_2.pt", # Textract + blip2 easy second run (epoch 2+6)
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-22_09:56:02/models/epoch_1.pt", # Textract context hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-11-27_11:01:16/models/epoch_2.pt", # SAM objects hard - 15% dropout
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-11-25_00:20:48/models/epoch_5.pt", # SAM objects easy - 10% dropout
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-11-24_22:26:38/models/epoch_6.pt", # SAM objects easy - 15% dropout
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-21_09:23:36/models/epoch_1.pt", # Textract context hard config2 2nd run (10+1) epoch
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-20_11:18:06/models/epoch_10.pt", # Textract context easy config2 1st run
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-19_19:48:31/models/epoch_3.pt", # Blip2 hard
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-18_13:46:14/models/epoch_9.pt", # Blip2 easy
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-17_13:16:12/models/epoch_3.pt", # Textract context hard 
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-15_21:19:18/models/epoch_6.pt", # Textract context easy
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-14_19:11:34/models/epoch_6.pt", # Trained with 9 distractors hard - 2nd run (In reality its like epoch 3+6)
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-13_14:04:32/models/epoch_3.pt", # Trained with 9 distractors hard - 1st run
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-12_13:38:35/models/epoch_4.pt", # Trained with 9 distractors easy
# +10 seed
# default="/runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-11-08_09:46:57/models/epoch_5.pt", # Textract run 1
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-10-23_22:00:07/models/epoch_6.pt", # VIT easy
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-10-22_19:40:44/models/epoch_1.pt", # VL-T5 - Trained on hard data resnet50-finetuned
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-10-20_09:57:15/models/epoch_6.pt", # VL-T5 - Trained on easy data resnet50-finetuned
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-10-07_11:37:10/models/epoch_5.pt", # VL-T5 - Trained on easy data resnet50
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-10-01_22:42:55/models/epoch_3.pt", # VL-T5 - Trained on hard data objects
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_2023-09-30_18:42:31/models/epoch_6.pt", # VL-T5 - Trained on easy data objects
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-09-22_12:54:22/models/epoch_2.pt", # VL-T5 - Trained on hard data
# default="runs/TextClozeImageTextT5Model_comics_images_Sim_CLR_text_2023-09-25_22:43:03/models/epoch_3.pt", # T5 trained on hard data
# default="runs/TextClozeImageTextT5Model_comics_images_Sim_CLR_text_2023-09-13_13:10:30/models/epoch_8.pt", # T5 trained on easy data
# default="runs/TextClozeImageTextVLT5Model_text_cloze_image_text_vlt5_simCLR_2023-09-21_11:41:35/models/epoch_6.pt", # VL-T5 - Trained on easy data

def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    torch.manual_seed(0)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
            assert checkpoint is not None, "ERROR: No checkpoint provided."
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
