"""PyTorch Dataset over preprocessed slices, plus patient-level k-fold splitting.

Splitting is done on patient IDs (not slice indices) so that no patient's
slices ever appear in both the train and validation sets of a fold.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def patient_kfold_splits(index_df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> list[dict]:
    """Return a list of {'train': [...patient_ids], 'val': [...patient_ids]} per fold."""
    patient_ids = sorted(index_df["patient_id"].unique())
    if len(patient_ids) < n_folds:
        raise ValueError(f"Only {len(patient_ids)} patients available, cannot make {n_folds} folds.")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(patient_ids):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]
        assert set(train_ids).isdisjoint(val_ids), "Patient leakage between train and val!"
        splits.append({"train": train_ids, "val": val_ids})
    return splits


class MSLesionDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, patient_ids: list[str], augment: bool = False):
        self.records = index_df[index_df["patient_id"].isin(patient_ids)].reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]
        image = np.load(PROJECT_ROOT / row["image_path"])  # (C, H, W) float32
        mask = np.load(PROJECT_ROOT / row["mask_path"])  # (H, W) uint8

        if self.augment:
            image, mask = self._augment(image, mask)

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float().unsqueeze(0),  # (1, H, W)
            "patient_id": row["patient_id"],
        }

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:  # horizontal flip
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]
        if np.random.rand() < 0.5:  # vertical flip
            image = image[:, ::-1, :]
            mask = mask[::-1, :]
        k = np.random.choice([0, 1, 2, 3])  # random 90-degree rotation
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(0, 1))
        return image, mask
