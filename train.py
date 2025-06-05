# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.5'

import argparse
from tqdm.auto import tqdm
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_collections import ConfigDict
from typing import List, Callable
import loralib as lora

from utils.audio_utils import prepare_data
from utils.settings import parse_args_train, initialize_environment, wandb_init, get_model_from_config
from utils.model_utils import bind_lora_to_model, load_start_checkpoint, save_weights, normalize_batch, \
    initialize_model_and_device, get_optimizer, save_last_weights

from utils.losses import choice_loss
from valid import valid_multi_gpu, valid, check_validation


import warnings
warnings.filterwarnings("ignore")


def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace, optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool, scaler: torch.cuda.amp.GradScaler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        config: Configuration object containing training parameters.
        args: Command-line arguments with specific settings (e.g., model type).
        optimizer: Optimizer used for training.
        device: Device to run the model on (CPU or GPU).
        device_ids: List of GPU device IDs if using multiple GPUs.
        epoch: The current epoch number.
        use_amp: Whether to use automatic mixed precision (AMP) for training.
        scaler: Scaler for AMP to manage gradient scaling.
        gradient_accumulation_steps: Number of gradient accumulation steps before updating the optimizer.
        train_loader: DataLoader for the training dataset.
        multi_loss: The loss function to use during training.

    Returns:
        None
    """

    model.train().to(device)
    print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
    loss_val = 0.
    total = 0

    normalize = getattr(config.training, 'normalize', False)

    pbar = tqdm(train_loader)
    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)  # mixture
        y = batch.to(device)

        if normalize:
            x, y = normalize_batch(x, y)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if 'roformer' in args.model_type:
                # loss is computed in forward pass
                loss = model(x, y)
                if isinstance(device_ids, (list, tuple)):
                    # If it's multiple GPUs sum partial loss
                    loss = loss.mean()
            else:
                y_ = model(x)
                loss = multi_loss(y_, y, x)

        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()
        if config.training.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        li = loss.item() * gradient_accumulation_steps
        loss_val += li
        total += 1
        pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
        wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})
        loss.detach()

    print(f'Training loss: {loss_val / total}')
    wandb.log({'train_loss': loss_val / total, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], best_metric: float,
                          epoch: int, scheduler: torch.optim.lr_scheduler._LRScheduler) -> float:
    """
    Compute and log the metrics for the current epoch, and save model weights if the metric improves.

    Args:
        model: The model to evaluate.
        args: Command-line arguments containing configuration paths and other settings.
        config: Configuration dictionary containing training settings.
        device: The device (CPU or GPU) used for evaluation.
        device_ids: List of GPU device IDs when using multiple GPUs.
        best_metric: The best metric value seen so far.
        epoch: The current epoch number.
        scheduler: The learning rate scheduler to adjust the learning rate.

    Returns:
        The updated best_metric.
    """

    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
    else:
        metrics_avg, all_metrics = valid(model, args, config, device, verbose=False)
    metric_avg = metrics_avg[args.metric_for_scheduler]
    if metric_avg > best_metric:
        store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt'
        print(f'Store weights: {store_path}')
        train_lora = args.train_lora
        save_weights(store_path, model, device_ids, train_lora)
        best_metric = metric_avg
    scheduler.step(metric_avg)
    wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
    for metric_name in metrics_avg:
        wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric


def train_model(args: argparse.Namespace) -> None:
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """

    args = parse_args_train(args)

    initialize_environment(args.seed, args.results_path)
    model, config = get_model_from_config(args.model_type, args.config_path)
    use_amp = getattr(config.training, 'use_amp', True)
    device_ids = args.device_ids
    batch_size = config.training.batch_size * len(device_ids)

    wandb_init(args, config, device_ids, batch_size)

    train_loader = prepare_data(config, args, batch_size)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='train')

    if args.train_lora:
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)

    device, model = initialize_model_and_device(model, args.device_ids)

    if args.pre_valid:
        if torch.cuda.is_available() and len(device_ids) > 1:
            valid_multi_gpu(model, args, config, args.device_ids, verbose=True)
        else:
            valid(model, args, config, device, verbose=True)

    optimizer = get_optimizer(config, model)
    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))

    # Reduce LR if no metric improvements for several epochs
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                  factor=config.training.reduce_factor)

    multi_loss = choice_loss(args, config)
    scaler = GradScaler()
    best_metric = float('-inf')

    print(
        f"Instruments: {config.training.instruments}\n"
        f"Losses for training: {args.loss}\n"
        f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
        f"Patience: {config.training.patience} "
        f"Reduce factor: {config.training.reduce_factor}\n"
        f"Batch size: {batch_size} "
        f"Grad accum steps: {gradient_accumulation_steps} "
        f"Effective batch size: {batch_size * gradient_accumulation_steps}\n"
        f"Dataset type: {args.dataset_type}\n"
        f"Optimizer: {config.training.optimizer}"
    )

    print(f'Train for: {config.training.num_epochs} epochs')

    for epoch in range(config.training.num_epochs):

        train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, gradient_accumulation_steps, train_loader, multi_loss)
        save_last_weights(args, model, device_ids)
        best_metric = compute_epoch_metrics(model, args, config, device, device_ids, best_metric, epoch, scheduler)


    import os
    import re
    import glob
# --- Final validation step: Find the best checkpoint from results_path and call check_validation ---
    print("\n--- Running final validation after training ---")

    # Ensure args has 'valid_path' and 'metrics' attributes set for validation
    if not hasattr(args, 'valid_path') or not args.valid_path:
        print("Error: args.valid_path is not set for final validation. Please provide it via command line or defaults.")
        return # Exit or handle error

    if not hasattr(args, 'metrics') or not args.metrics:
        print("Error: args.metrics is not set for final validation. Please provide it via command line or defaults.")
        return # Exit or handle error

    final_validation_checkpoint = None
    if os.path.exists(args.results_path):
        pattern = re.compile(rf'model_{args.model_type}_ep_\d+_{args.metric_for_scheduler}_([0-9.]+)\.ckpt')

        all_checkpoints = glob.glob(os.path.join(args.results_path, f'model_{args.model_type}_ep_*.ckpt'))

        best_score = float('-inf')

        for ckpt_path in all_checkpoints:
            match = pattern.search(os.path.basename(ckpt_path))
            if match:
                try:
                    score = float(match.group(1))
                    if score > best_score:
                        best_score = score
                        final_validation_checkpoint = ckpt_path
                except ValueError:
                    print(f"Warning: Could not parse metric from filename: {os.path.basename(ckpt_path)}")

        if final_validation_checkpoint:
            print(f"Found best checkpoint for final validation: {final_validation_checkpoint} (Score: {best_score:.4f})")
        else:
            last_ckpt_path = os.path.join(args.results_path, 'last.ckpt')
            if os.path.exists(last_ckpt_path):
                final_validation_checkpoint = last_ckpt_path
                print(f"No metric-specific checkpoints found, using last saved checkpoint for final validation: {final_validation_checkpoint}")
            else:
                print("Warning: No suitable checkpoint found in results directory. Final validation will run on the current model state (after all epochs).")
    else:
        print(f"Warning: Results path '{args.results_path}' does not exist. Cannot find best checkpoint. Final validation will run on the current model state.")

    # Prepare a dictionary of arguments to pass to check_validation
    # check_validation expects a dictionary that it will parse with parse_args_valid
    dict_args_for_final_valid = vars(args).copy() # Convert Namespace to dict and copy it

    # --- Modify start_check_point ---
    dict_args_for_final_valid['start_check_point'] = final_validation_checkpoint

    # --- Modify valid_path to point to the 'test' split ---
    # Assuming args.valid_path is a list, and contains something like '/path/to/dataset/validation'
    # We want to change 'validation' to 'test'. This handles multiple paths in the list.
    modified_valid_paths = []
    for path in args.valid_path:
        # Using os.path.normpath to handle varying path separators and redundancies
        normalized_path = os.path.normpath(path)
        if normalized_path.endswith('validation'):
            # Replace the last component
            modified_path = os.path.join(os.path.dirname(normalized_path), 'test')
            modified_valid_paths.append(modified_path)
        elif normalized_path.endswith('valid'): # Also handle just 'valid' as a common split name
            modified_path = os.path.join(os.path.dirname(normalized_path), 'test')
            modified_valid_paths.append(modified_path)
        else:
            # If the path doesn't end with 'validation' or 'valid', print a warning or handle as needed
            print(f"Warning: The validation path '{path}' does not end with 'validation' or 'valid'. Attempting to append 'test' if it's a directory, or you may need to adjust this logic.")
            # A more robust solution might be to check if 'test' subdirectory exists
            # For now, we'll just append it assuming the base path is correct.
            modified_path = os.path.join(normalized_path, 'test') # Simple append if no specific 'validation' suffix
            if not os.path.isdir(modified_path):
                 print(f"Warning: Derived test path '{modified_path}' does not seem to be a directory. Please check your valid_path configuration.")
            modified_valid_paths.append(modified_path)

    dict_args_for_final_valid['valid_path'] = modified_valid_paths
    print(f"Original valid_path for training: {args.valid_path}")
    print(f"Modified valid_path for final validation: {dict_args_for_final_valid['valid_path']}")

    # Call check_validation directly with the dictionary of arguments
    # check_validation will handle all model loading and validation execution.
    print(f"Calling check_validation for final validation with start_check_point: {dict_args_for_final_valid['start_check_point']}")
    check_validation(dict_args_for_final_valid)

    print("\n--- Final Validation Completed ---")


if __name__ == "__main__":
    train_model(None)