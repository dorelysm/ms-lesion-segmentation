"""Train the 2D U-Net on preprocessed MS lesion slices.

Config-driven and device-agnostic: runs on CPU (slow, for development/smoke
tests) or CUDA (fast, for real training) with no code changes.

Usage:
    python src/train.py --config configs/baseline.yaml
    python src/train.py --config configs/baseline.yaml --smoke-test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # allow `python src/train.py` as well as `python -m src.train`

from src.data.dataset import MSLesionDataset, patient_kfold_splits
from src.models.unet import UNet2D
from src.utils.metrics import DiceBCELoss, compute_all


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, metric_sums, n_batches = 0.0, None, 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, masks)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_metrics = compute_all(logits.detach(), masks)
        if metric_sums is None:
            metric_sums = {k: 0.0 for k in batch_metrics}
        for k, v in batch_metrics.items():
            metric_sums[k] += v

        total_loss += loss.item()
        n_batches += 1

    avg = {"loss": total_loss / n_batches}
    avg.update({k: v / n_batches for k, v in metric_sums.items()})
    return avg


def train_fold(fold_idx: int, train_ids: list, val_ids: list, index_df: pd.DataFrame, cfg: dict, device: torch.device) -> dict:
    train_cfg = cfg["train"]
    smoke = cfg.get("_smoke_test", False)
    batch_size = cfg["smoke_test"]["batch_size"] if smoke else train_cfg["batch_size"]
    epochs = cfg["smoke_test"]["epochs"] if smoke else train_cfg["epochs"]

    train_ds = MSLesionDataset(index_df, train_ids, augment=train_cfg["augment"])
    val_ds = MSLesionDataset(index_df, val_ids, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=train_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=train_cfg["num_workers"])

    in_channels = len(cfg["data"]["modalities"])
    model = UNet2D(in_channels=in_channels, out_channels=1, base_channels=cfg["model"]["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    criterion = DiceBCELoss(bce_weight=train_cfg["bce_weight"])

    checkpoint_dir = PROJECT_ROOT / cfg["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice, history = -1.0, []

    print(f"\n=== Fold {fold_idx}: {len(train_ids)} train patients, {len(val_ids)} val patients, "
          f"{len(train_ds)} train slices, {len(val_ds)} val slices, device={device} ===")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "seconds": elapsed})
        print(f"[fold {fold_idx}] epoch {epoch}/{epochs} "
              f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} ({elapsed:.1f}s)")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), checkpoint_dir / f"fold{fold_idx}_best.pt")

    history_path = checkpoint_dir / f"fold{fold_idx}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {"fold": fold_idx, "best_val_dice": best_val_dice, "history_path": str(history_path)}


def main(config_path: Path, smoke_test: bool) -> None:
    cfg = load_config(config_path)
    cfg["_smoke_test"] = smoke_test

    index_df = pd.read_csv(PROJECT_ROOT / cfg["data"]["index_csv"])

    if smoke_test:
        patient_limit = cfg["smoke_test"]["patient_limit"]
        keep_ids = sorted(index_df["patient_id"].unique())[:patient_limit]
        index_df = index_df[index_df["patient_id"].isin(keep_ids)].reset_index(drop=True)
        print(f"[smoke test] restricted to {len(keep_ids)} patients ({len(index_df)} slices)")

    n_folds = min(cfg["split"]["n_folds"], index_df["patient_id"].nunique())
    splits = patient_kfold_splits(index_df, n_folds=n_folds, seed=cfg["split"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds_to_run = [0] if smoke_test else cfg["split"]["folds_to_run"]
    results = []
    for fold_idx in folds_to_run:
        split = splits[fold_idx]
        result = train_fold(fold_idx, split["train"], split["val"], index_df, cfg, device)
        results.append(result)

    print("\n=== Summary ===")
    for r in results:
        print(f"fold {r['fold']}: best val dice = {r['best_val_dice']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny run on a few patients/epochs to validate the pipeline.")
    args = parser.parse_args()
    main(args.config, args.smoke_test)
