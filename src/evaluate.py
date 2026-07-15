"""Evaluate trained fold checkpoints on their held-out validation patients and
report Dice / IoU / sensitivity / precision, averaged across folds (mean ± std).

Usage:
    python src/evaluate.py --config configs/baseline.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MSLesionDataset, patient_kfold_splits
from src.models.unet import UNet2D
from src.utils.metrics import compute_all
from src.utils.viz import plot_prediction_grid


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_fold(fold_idx: int, val_ids: list, index_df: pd.DataFrame, cfg: dict, device: torch.device) -> tuple[dict, dict]:
    checkpoint_path = PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / f"fold{fold_idx}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint for fold {fold_idx} at {checkpoint_path}. Run train.py first.")

    in_channels = len(cfg["data"]["modalities"])
    model = UNet2D(in_channels=in_channels, out_channels=1, base_channels=cfg["model"]["base_channels"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    val_ds = MSLesionDataset(index_df, val_ids, augment=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0)

    metric_sums, n_batches = None, 0
    sample_batch = None
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)

            batch_metrics = compute_all(logits, masks)
            if metric_sums is None:
                metric_sums = {k: 0.0 for k in batch_metrics}
            for k, v in batch_metrics.items():
                metric_sums[k] += v
            n_batches += 1

            if sample_batch is None:
                sample_batch = {
                    "images": images.cpu().numpy(),
                    "gt": masks.cpu().numpy()[:, 0],
                    "pred": (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[:, 0],
                }

    avg_metrics = {k: v / n_batches for k, v in metric_sums.items()}
    return avg_metrics, sample_batch


def main(config_path: Path, folds: list | None, smoke_test: bool) -> None:
    cfg = load_config(config_path)
    index_df = pd.read_csv(PROJECT_ROOT / cfg["data"]["index_csv"])

    if smoke_test:
        # Must mirror the patient restriction in train.py's smoke-test mode exactly,
        # otherwise the k-fold splits computed here won't match the ones used to
        # produce the checkpoints (same index_df + same seed -> same splits).
        patient_limit = cfg["smoke_test"]["patient_limit"]
        keep_ids = sorted(index_df["patient_id"].unique())[:patient_limit]
        index_df = index_df[index_df["patient_id"].isin(keep_ids)].reset_index(drop=True)

    n_folds = min(cfg["split"]["n_folds"], index_df["patient_id"].nunique())
    splits = patient_kfold_splits(index_df, n_folds=n_folds, seed=cfg["split"]["seed"])
    folds = folds or ([0] if smoke_test else cfg["split"]["folds_to_run"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows, first_sample = [], None
    for fold_idx in folds:
        metrics, sample = evaluate_fold(fold_idx, splits[fold_idx]["val"], index_df, cfg, device)
        rows.append({"fold": fold_idx, **metrics})
        print(f"fold {fold_idx}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        if first_sample is None:
            first_sample = sample

    results_df = pd.DataFrame(rows)
    summary = results_df.drop(columns="fold").agg(["mean", "std"])
    print("\n=== Cross-fold summary ===")
    print(summary)

    figures_dir = PROJECT_ROOT / cfg["output"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(figures_dir / "metrics_by_fold.csv", index=False)
    summary.to_csv(figures_dir / "metrics_summary.csv")

    if first_sample is not None:
        plot_prediction_grid(
            first_sample["images"], first_sample["gt"], first_sample["pred"],
            n=min(4, len(first_sample["images"])),
            save_path=figures_dir / "prediction_examples.png",
        )
    print(f"\nSaved results to {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--smoke-test", action="store_true", help="Match the patient subset used by `train.py --smoke-test`.")
    args = parser.parse_args()
    main(args.config, args.folds, args.smoke_test)
