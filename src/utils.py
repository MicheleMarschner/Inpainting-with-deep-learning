from datetime import datetime
import wandb
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seeds(seed: int = 51):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seeds set to {seed}")


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M")


def init_wandb(model, opt_name, name, sample_number, num_params=-1, batch_size=64, epochs=15):

    config = {
        "optimizer_type": opt_name,
        "model_architecture": model.__class__.__name__,
        "batch_size": batch_size,
        "num_epochs": epochs,
        "num_parameters": num_params,
        "sample_number": sample_number
    }

    timestamp = get_timestamp()

    run = wandb.init(
        project="Inpainting",
        name=f"{name}_run_{timestamp}",
        config=config,
        reinit='finish_previous',                 # allows starting a new run inside one script
    )

    return run


def compute_dataset_mean_std(
    dataset,
    batch_size: int = 32,
):
    """
    Compute per-channel mean and std on the train dataset that returns
    images as tensors in [0,1], shape (C,H,W).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),       ## TODO: what does it do?
    )

    channel_sum = 0.0               # sum of all pixel values in channel c
    channel_sq_sum = 0.0            # sum of squared pixel values in channel c
    pixel_count = 0                 # total number of pixels per channel

    for img in tqdm(loader):
        img = img.float()               # (B,C,H,W)
        b, c, h, w = img.shape

        pixel_count += b * h * w                # total pixel count across all batches with height x width
        channel_sum += img.sum(dim=(0, 2, 3))   # sum over batch_dim, height and width -> (C,)
        channel_sq_sum += (img ** 2).sum(dim=(0, 2, 3)) # accumulate squared sum per channel

    mean = channel_sum / pixel_count
    var = channel_sq_sum / pixel_count - mean ** 2
    std = torch.sqrt(var.clamp(min=1e-12))          # 1e-12 ensures numerical stability and that var is not-negative

    return mean.tolist(), std.tolist()


def format_time(seconds):
    """
    Convert a duration in seconds to a human-readable 'MMm SSs' string.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted duration, e.g. "02m 15s".
    """
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}m {s:02d}s"


def cleanup(*objs):
    import gc, torch
    for o in objs:
        try:
            del o
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


