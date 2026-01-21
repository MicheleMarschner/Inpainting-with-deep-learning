import matplotlib.pyplot as plt
import math
import torch
from typing import Sequence, Optional


def show_dataset_grid(dataset, n=8, cols=4):
    """
    Show n images from a dataset in a grid.
    Return: tensor (C,H,W) in [0,1].
    """
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    ax = ax.flatten()

    # Show the first n images from the dataset.
    for idx in range(n):
        img = dataset[idx]          # (C,H,W)
        img = img.permute(1, 2, 0)  # reorders dim: (C,H,W) -> (H,W,C)
        ax[idx].imshow(img)
        ax[idx].axis("off")

    # Hide unused axes
    for j in range(n, len(ax)):
        ax[j].axis("off")

    plt.tight_layout()

    return fig, ax


def show_img_grid(
    tensors: Sequence[torch.Tensor],
    titles: Optional[Sequence[str]] = None,
    cols: int = 3,
    cmap_mask: str = "gray",
):
    """
    Show an arbitrary number of tensors in a grid with fixed number of columns.

    tensors:
      - list of tensors
      - RGB: (3,H,W) or (4,H,W)
      - Mask: (1,H,W) or (H,W)

    titles:
      - optional list of titles (same length as tensors)

    cols:
      - number of images per row (default: 3)
    """
    n = len(tensors)
    rows = math.ceil(n / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    ax = ax.flatten() if n > 1 else [ax]

    for i, tensor in enumerate(tensors):
        x = tensor.detach().cpu()

        if x.dim() == 4:
            x.squeeze(0)
        
        # RGB image
        if x.dim() == 3 and x.size(0) == 4:
            x = x[:3].detach().cpu().clamp(0, 1).permute(1, 2, 0)
            ax[i].imshow(x)

        # RGB image
        elif x.dim() == 3 and x.size(0) == 3:
            x = x.detach().cpu().clamp(0, 1).permute(1, 2, 0)
            ax[i].imshow(x)

        # Mask: (1,H,W)
        elif x.dim() == 3 and x.size(0) == 1:
            ax[i].imshow(x[0], cmap=cmap_mask, vmin=0, vmax=1)

        else:
            raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

        ax[i].axis("off")
        if titles is not None:
            ax[i].set_title(titles[i])

    # Hide unused axes
    for j in range(i + 1, len(ax)):
        ax[j].axis("off")

    plt.tight_layout()
    return fig, ax


## !!TODO bleiben oder mit oberen mergen und unnormalisierung außerhalb?
def show_norm_img_grid(
    tensors: Sequence[torch.Tensor],
    titles: Optional[Sequence[str]] = None,
    cols: int = 3,
    cmap_mask: str = "gray",
):
    """
    Show an arbitrary number of tensors in a grid with fixed number of columns.

    tensors:
      - list of tensors
      - RGB: (3,H,W) or (4,H,W)
      - Mask: (1,H,W) or (H,W)

    titles:
      - optional list of titles (same length as tensors)

    cols:
      - number of images per row (default: 3)
    """
    n = len(tensors)
    rows = math.ceil(n / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    ax = ax.flatten() if n > 1 else [ax]

    for i, tensor in enumerate(tensors):
        x = tensor.detach().cpu()
        
        # RGB image
        if x.dim() == 3 and x.size(0) == 4:
            mask = x[3].unsqueeze(2)
            x = unnormalize(x[:3], MEAN, STD).clamp(0, 1).permute(1, 2, 0)
            x = x * mask
            ax[i].imshow(x)

        # RGB image
        elif x.dim() == 3 and x.size(0) == 3:
            x = unnormalize(x, MEAN, STD).clamp(0, 1).permute(1, 2, 0)
            ax[i].imshow(x)

        # Mask: (1,H,W)
        elif x.dim() == 3 and x.size(0) == 1:
            ax[i].imshow(x[0], cmap=cmap_mask, vmin=0, vmax=1)

        # Mask: (H,W)
        elif x.dim() == 2:
            ax[i].imshow(x, cmap=cmap_mask, vmin=0, vmax=1)

        else:
            raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

        ax[i].axis("off")
        if titles is not None:
            ax[i].set_title(titles[i])

    # Hide unused axes
    for j in range(i + 1, len(ax)):
        ax[j].axis("off")

    plt.tight_layout()
    return fig, ax


def plot_val_losses(loss_dict, title="Validation Loss per Model", figsize=(8,5)):
    """
    Plots validation loss curves for multiple models for comparison.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, losses in loss_dict.items():
        ax.plot(losses, label=model_name)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax

