"""Segmentation metrics for binary lesion masks. All functions take raw model
logits (`pred`) and a binary ground-truth mask (`target`), both shaped
(B, 1, H, W) or (1, H, W), and return a Python float.
"""

import torch


def _binarize(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(pred)
    return (probs > threshold).float()


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bin = _binarize(pred)
    intersection = (pred_bin * target).sum()
    denom = pred_bin.sum() + target.sum()
    return ((2 * intersection + eps) / (denom + eps)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bin = _binarize(pred)
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return ((intersection + eps) / (union + eps)).item()


def sensitivity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Recall: fraction of true lesion pixels correctly predicted."""
    pred_bin = _binarize(pred)
    true_positive = (pred_bin * target).sum()
    return ((true_positive + eps) / (target.sum() + eps)).item()


def precision(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bin = _binarize(pred)
    true_positive = (pred_bin * target).sum()
    return ((true_positive + eps) / (pred_bin.sum() + eps)).item()


def compute_all(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "dice": dice_coefficient(pred, target),
        "iou": iou_score(pred, target),
        "sensitivity": sensitivity(pred, target),
        "precision": precision(pred, target),
    }


class DiceBCELoss(torch.nn.Module):
    """Combined Dice + BCE loss, standard for imbalanced medical segmentation."""

    def __init__(self, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum()
        dice_loss = 1 - (2 * intersection + self.eps) / (probs.sum() + target.sum() + self.eps)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
