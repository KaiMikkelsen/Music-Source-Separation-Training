# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.4'

import random

import argparse
from tqdm.auto import tqdm
import os
import torch
import wandb
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ml_collections import ConfigDict
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Callable
from datetime import datetime
import uuid

from utils.dataset import MSSDataset
from utils.settings import get_model_from_config, parse_args_inference
from valid import valid_multi_gpu, valid

from utils.model_utils import bind_lora_to_model, load_start_checkpoint, save_weights, normalize_batch, \
    initialize_model_and_device, get_optimizer, save_last_weights
import loralib as lora
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler  # You can choose other samplers if needed
from optuna.trial import TrialState

import warnings

warnings.filterwarnings("ignore")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--use_multistft_loss", action='store_true', help="Use MultiSTFT Loss (from auraloss package)")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora", action='store_true', help="Train with LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def manual_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    Initialize the environment by setting the random seed, configuring PyTorch settings,
    and creating the results directory.

    Args:
        seed: The seed value for reproducibility.
        results_path: Path to the directory where results will be stored.
    """

    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)

def wandb_init(args: argparse.Namespace, config: Dict, device_ids: List[int], batch_size: int) -> None:
    """
    Initialize the Weights & Biases (wandb) logging system.

    Args:
        args: Parsed command-line arguments containing the wandb key.
        config: Configuration dictionary for the experiment.
        device_ids: List of GPU device IDs used for training.
        batch_size: Batch size for training.
    """
    import datetime
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    run_name = f"{args.model_type}_{args.data_path}_{current_date}"

    print(f"Initializing wandb: {args}")

    if args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(project='msst', 
                   config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size },
                   name=run_name,
                   group="Optuna")


def prepare_data(config: Dict, args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    Prepare the training dataset and data loader.

    Args:
        config: Configuration dictionary for the dataset.
        args: Parsed command-line arguments containing dataset paths and settings.
        batch_size: Batch size for training.

    Returns:
        DataLoader object for the training dataset.
    """

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, f'metadata_{args.dataset_type}.pkl'),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader


def initialize_model_and_device(model: torch.nn.Module, device_ids: List[int]) -> Tuple[Union[torch.device, str], torch.nn.Module]:
    """
    Initialize the model and assign it to the appropriate device (GPU or CPU).

    Args:
        model: The PyTorch model to be initialized.
        device_ids: List of GPU device IDs to use for parallel processing.

    Returns:
        A tuple containing the device and the model moved to that device.
    """

    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")

    return device, model


def get_optimizer(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Initializes an optimizer based on the configuration.

    Args:
        config: Configuration object containing training parameters.
        model: PyTorch model whose parameters will be optimized.

    Returns:
        A PyTorch optimizer object configured based on the specified settings.
    """

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        print(f'Optimizer params from config:\n{optim_params}')

    name_optimizer = getattr(config.training, 'optimizer',
                             'No optimizer in config')

    if name_optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        print(f'Unknown optimizer: {name_optimizer}')
        exit()
    return optimizer


def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    """
    Compute the masked loss, which applies a quantile-based mask to the MSE loss.

    Args:
        y_: Predicted tensor of shape [num_sources, batch_size, num_channels, chunk_size].
        y: Ground truth tensor of the same shape as y_.
        q: Quantile value for the mask.
        coarse: If True, computes the mean loss over the last two dimensions (channels and chunk_size).

    Returns:
        The masked mean loss.
    """

    # shape = [num_sources, batch_size, num_channels, chunk_size]
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def multistft_loss(y: torch.Tensor, y_: torch.Tensor, loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Compute the multi-STFT loss between the predicted and ground truth tensors.

    Args:
        y: Ground truth tensor, shape [num_sources, batch_size, num_channels, chunk_size] (or [num_sources, batch_size, chunk_size] for models like Apollo).
        y_: Predicted tensor with the same shape as y.
        loss_multistft: A function that computes the multi-STFT loss.

    Returns:
        The multi-STFT loss value.
    """

    if len(y_.shape) == 4:
        y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
        y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
    elif len(y_.shape) == 3:
        # For models like apollo no need to reshape
        y1_ = y_
        y1 = y
    else:
        raise ValueError(f"Invalid shape for predicted array: {y_.shape}. Expected 3 or 4 dimensions.")

    return loss_multistft(y1_, y1)


def choice_loss(args: argparse.Namespace, config: ConfigDict) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Select and return the appropriate loss function based on the configuration and arguments.

    Args:
        args: Parsed command-line arguments containing flags for different loss functions.
        config: Configuration object containing loss settings and parameters.

    Returns:
        A loss function that can be applied to the predicted and ground truth tensors.
    """

    if args.use_multistft_loss:
        loss_options = dict(getattr(config, 'loss_multistft', {}))
        print(f'Loss options: {loss_options}')
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(**loss_options)

        if args.use_mse_loss and args.use_l1_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(y_, y) + F.l1_loss(y_, y)
        elif args.use_mse_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + nn.MSELoss()(y_, y)
        elif args.use_l1_loss:
            def multi_loss(y_, y):
                return (multistft_loss(y_, y, loss_multistft) / 1000) + F.l1_loss(y_, y)
        else:
            def multi_loss(y_, y):
                return multistft_loss(y_, y, loss_multistft) / 1000
    elif args.use_mse_loss:
        if args.use_l1_loss:
            def multi_loss(y_, y):
                return nn.MSELoss()(y_, y) + F.l1_loss(y_, y)
        else:
            multi_loss = nn.MSELoss()
    elif args.use_l1_loss:
        multi_loss = F.l1_loss
    else:
        def multi_loss(y_, y):
            return masked_loss(y_, y, q=config.training.q, coarse=config.training.coarse_loss_clip)
    return multi_loss


def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a batch of tensors (x and y) by subtracting the mean and dividing by the standard deviation.

    Args:
        x: Tensor to normalize.
        y: Tensor to normalize (same as x, typically).

    Returns:
        A tuple of normalized tensors (x, y).
    """

    mean = x.mean()
    std = x.std()
    if std != 0:
        x = (x - mean) / std
        y = (y - mean) / std
    return x, y


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
            if args.model_type in ['mel_band_roformer', 'bs_roformer']:
                # loss is computed in forward pass
                loss = model(x, y)
                if isinstance(device_ids, (list, tuple)):
                    # If it's multiple GPUs sum partial loss
                    loss = loss.mean()
            else:
                y_ = model(x)
                loss = multi_loss(y_, y)

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


def save_weights(store_path, model, device_ids, train_lora):

    if train_lora:
        torch.save(lora.lora_state_dict(model), store_path)
    else:
        print("saving weights to ", store_path)
        state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
        torch.save(
            state_dict,
            store_path
        )


def save_last_weights(args: argparse.Namespace, model: torch.nn.Module, device_ids: List[int]) -> None:
    """
    Save the model's state_dict to a file for later use.

    Args:
        args: Command-line arguments containing the results path and model type.
        model: The model whose weights will be saved.
        device_ids: List of GPU device IDs if using multiple GPUs.

    Returns:
        None
    """

    store_path = f'{args.results_path}/last_{args.model_type}.ckpt'
    train_lora = args.train_lora
    save_weights(store_path, model, device_ids, train_lora)


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

    print("printing epoch metrics")
    print(f"metrics for scheduler {args.metric_for_scheduler}")
    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg, all_metrics = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
    else:
        metrics_avg, all_metrics = valid(model, args, config, device, verbose=False)
    print(f"metrics_avg {metrics_avg}")
    print(f"metrics for scheduler {args.metric_for_scheduler}")
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

    args = parse_args(args)

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
        




def objective(trial: Trial, args: argparse.Namespace) -> float:
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        Best metric achieved during training.
    """

    args = parse_args(args)
    initialize_environment(args.seed, args.results_path)
    model, config = get_model_from_config(args.model_type, args.config_path)

    # do optuna here i think
   # do optuna here i think

   # **Model Capacity**:
    channels = trial.suggest_int("channels", 24, 64, step=8)
    growth = trial.suggest_int("growth", 1, 3)
    bottom_channels = trial.suggest_int("bottom_channels", 256, 1024, step=128)
    num_subbands = trial.suggest_categorical("num_subbands", [1, 2, 4])
    freq_emb = trial.suggest_uniform("freq_emb", 0.1, 0.5)

    # **Convolution Parameters**:
    encoder_kernel_size = trial.suggest_categorical("encoder_kernel_size", [8, 16, 32])
    decoder_kernel_size = trial.suggest_categorical("decoder_kernel_size", [8, 16, 32])

    # **Transformer Parameters**:
    t_heads = trial.suggest_int("t_heads", 4, 16, step=4)
    t_hidden_scale = trial.suggest_uniform("t_hidden_scale", 2.0, 8.0)
    dconv_mode = trial.suggest_categorical("dconv_mode", [1, 2, 3])

    # **Latent Space Processing**:
    t_emb_scale = trial.suggest_uniform("t_emb_scale", 0.1, 1.0)
    t_layers = trial.suggest_int("t_layers", 1, 8)

    # **Normalization & Activation**:
    norm_type = trial.suggest_categorical("norm_type", ["batchnorm", "layernorm", "groupnorm"])
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "swish"])

    # **Dropout & Regularization**:
    t_dropout = trial.suggest_uniform("t_dropout", 0.0, 0.3)
    t_weight_decay = trial.suggest_loguniform("t_weight_decay", 1e-6, 1e-3)

    # **Loss Weighting**:
    loss_alpha = trial.suggest_uniform("loss_alpha", 0.1, 1.0)
    loss_beta = trial.suggest_uniform("loss_beta", 0.1, 1.0)

    # **Learning-related**:
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-4)
    reduce_factor = trial.suggest_uniform("reduce_factor", 0.1, 0.5)
    ema_momentum = trial.suggest_uniform("ema_momentum", 0.95, 0.99)
    optimizer = trial.suggest_categorical("optimizer", ["rmsprop", "adam", "adamw", "sgd"])
    batch_size = trial.suggest_int("batch_size", 1, 10, step=1)
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 10, step=1)

    # **Audio Chunking**:
    hop_length = trial.suggest_categorical("hop_length", [512, 1024, 2048])
    segment = trial.suggest_int("segment", 5, 15)

    chunk_size = 44100 * segment

    # **Augmentation settings**:
    augmentation_loudness_min = trial.suggest_loguniform("augmentation_loudness_min", 0.2, 0.8)
    augmentation_loudness_max = trial.suggest_loguniform("augmentation_loudness_max", 1.2, 2.0)
    random_phase_shift = trial.suggest_uniform("random_phase_shift", 0, 180)

    # **Inference settings**:
    num_overlap = trial.suggest_int("num_overlap", 1, 8, step=1)

    # Apply sampled values from Optuna to the configuration
    config.htdemucs.channels = channels
    config.htdemucs.growth = growth
    config.htdemucs.bottom_channels = bottom_channels
    config.htdemucs.num_subbands = num_subbands
    config.htdemucs.freq_emb = freq_emb
    config.htdemucs.encoder_kernel_size = encoder_kernel_size
    config.htdemucs.decoder_kernel_size = decoder_kernel_size

    # Transformer parameters
    config.htdemucs.t_heads = t_heads
    config.htdemucs.t_hidden_scale = t_hidden_scale
    config.htdemucs.dconv_mode = dconv_mode
    config.htdemucs.t_emb_scale = t_emb_scale
    config.htdemucs.t_layers = t_layers

    # Normalization & Activation
    # config.model.norm_type = norm_type
    # config.model.activation = activation

    # Regularization
    config.htdemucs.t_dropout = t_dropout
    config.htdemucs.t_weight_decay = t_weight_decay

    # Loss Weights
    # config.training.loss_alpha = loss_alpha
    # config.training.loss_beta = loss_beta

    # Learning-related parameters
    config.training.lr = lr
    config.training.reduce_factor = reduce_factor
    config.training.ema_momentum = ema_momentum
    config.training.optimizer = optimizer
    config.training.batch_size = batch_size
    config.gradient_accumulation_steps = gradient_accumulation_steps
    config.training.num_steps = 1000
    config.training.num_epochs = 50


    # # Audio Chunking
    config.training.chunk_size = chunk_size
    config.training.hop_length = hop_length
    config.training.segment = segment

    # Augmentation settings
    config.augmentations.loudness_min = augmentation_loudness_min
    config.training.loudness_max = augmentation_loudness_max
    # config.training.random_phase_shift = random_phase_shift

    # Inference settings
    config.inference.num_overlap = num_overlap

    # end optuna


    # initialize_environment(args.seed, args.results_path)
    # model, config = get_model_from_config(args.model_type, args.config_path)
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

        try:
            train_one_epoch(model, config, args, optimizer, device, device_ids, epoch,
                        use_amp, scaler, gradient_accumulation_steps, train_loader, multi_loss)
        except torch.cuda.OutOfMemoryError as e:
            print(f"Error occurred during training: {e}")
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
                
        except Exception as e:
            print(f"An unexpected error occurred during training: {e}")
            #torch.cuda.empty_cache()  # Optional, only relevant if it's CUDA-related
            raise optuna.exceptions.TrialPruned()  # Re-raise the exception to propagate it up
            #continue
        save_last_weights(args, model, device_ids)
        best_metric = compute_epoch_metrics(model, args, config, device, device_ids, best_metric, epoch, scheduler)
        trial.report(best_metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_metric
    


#if __name__ == "__main__":
#    train_model(None)

if __name__ == "__main__":
    # Define an Optuna study
    
    # Define an Optuna study
    print("Optuna study started")

    db_folder = "optunadb"
    os.makedirs(db_folder, exist_ok=True)  # Create folder if it doesn't exist


    unique_id = uuid.uuid4().hex[:8]  # Generate a short unique ID
    # Generate a unique filename for each study
    db_path = os.path.join(db_folder, f"htdemucs_optimization_{datetime.now().strftime('%Y-%m-%d')}_{unique_id}.sqlite3")

    # Create the study with the new database path
    study = optuna.create_study(
        direction="maximize",  # Change to "minimize" if optimizing a loss
        sampler=TPESampler(),  # TPE sampler for efficient search
        storage=f"sqlite:///{db_path}",  # Save in "optunadb" folder
        study_name=f"htdemucs_optimization_{datetime.now().strftime('%Y-%m-%d')}_{unique_id}"
    )

    #train_model(None)
    # Run the optimization
    #study.optimize(objective, n_trials=100)  # Adjust n_trials as needed
    study.optimize(lambda trial: objective(trial, None), n_trials=150)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))