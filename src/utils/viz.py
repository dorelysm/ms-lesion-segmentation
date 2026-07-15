"""Visualization helpers: overlay lesion masks (ground truth / prediction) on MRI slices."""

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(ax, image: np.ndarray, mask: np.ndarray, color: tuple = (1, 0, 0), alpha: float = 0.4) -> None:
    """Draw `image` (grayscale, H x W) with `mask` (binary, H x W) overlaid in `color`."""
    norm_image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    rgb = np.stack([norm_image] * 3, axis=-1)
    overlay = rgb.copy()
    for c in range(3):
        overlay[..., c] = np.where(mask > 0, color[c], rgb[..., c])
    blended = (1 - alpha) * rgb + alpha * overlay
    ax.imshow(blended)
    ax.axis("off")


def plot_prediction_grid(images: np.ndarray, gt_masks: np.ndarray, pred_masks: np.ndarray, n: int = 4, save_path=None):
    """images: (N, C, H, W) taking channel 0 for display; gt_masks/pred_masks: (N, H, W)."""
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = images[i, 0]
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title("MRI slice" if i == 0 else "")
        axes[i, 0].axis("off")

        overlay_mask(axes[i, 1], img, gt_masks[i], color=(0, 1, 0))
        axes[i, 1].set_title("Ground truth" if i == 0 else "")

        overlay_mask(axes[i, 2], img, pred_masks[i], color=(1, 0, 0))
        axes[i, 2].set_title("Prediction" if i == 0 else "")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
