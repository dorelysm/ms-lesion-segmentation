"""Run MS lesion segmentation inference on a new patient's NIfTI files.

The patient directory must contain T1, T2, and FLAIR volumes (matched by keyword,
same as preprocessing.py). A ground-truth mask is optional: if found, Dice is
reported; if not, the script runs in prediction-only mode.

Usage:
    # Single checkpoint
    python src/predict.py --patient-dir data/raw/patient_001 --checkpoint outputs/checkpoints/fold0_best.pt

    # Ensemble of all 5 folds (recommended)
    python src/predict.py --patient-dir data/raw/patient_001 --ensemble

    # Skip N4 bias correction for a quick test
    python src/predict.py --patient-dir data/raw/patient_001 --ensemble --skip-bias-correction
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml
from skimage.transform import resize as sk_resize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as _nib
from scipy.ndimage import zoom as _nd_zoom

import SimpleITK as sitk

from src.data.preprocessing import (
    MODALITY_KEYWORDS,
    REFERENCE_MODALITY,
    _find_file,
    _n4_bias_correct_sitk,
    _register_sitk,
    load_patient_volumes,
    normalize_volume,
)
from src.models.unet import UNet2D
from src.utils.viz import overlay_mask

MIN_BRAIN_PIXELS = 100


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    in_channels = len(cfg["data"]["modalities"])
    model = UNet2D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=cfg["model"]["base_channels"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_ensemble(checkpoint_dir: Path, n_folds: int, cfg: dict, device: torch.device) -> list:
    models = []
    for fold in range(n_folds):
        ckpt = checkpoint_dir / f"fold{fold}_best.pt"
        if not ckpt.exists():
            print(f"  Warning: checkpoint not found for fold {fold} ({ckpt}), skipping.")
            continue
        models.append(load_model(ckpt, cfg, device))
    if not models:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}.")
    return models


def _resize_stack(stack: np.ndarray, image_size: int) -> np.ndarray:
    return np.stack(
        [sk_resize(c, (image_size, image_size), preserve_range=True, anti_aliasing=True) for c in stack],
        axis=0,
    ).astype(np.float32)


def _load_volumes_no_mask(patient_dir: Path, modalities: list, bias_correct: bool, register: bool = True) -> dict:
    """Load modality volumes without requiring a lesion mask (prediction-only mode)."""
    from src.data.preprocessing import MASK_KEYWORDS
    raw_sitk: dict = {}
    for modality in modalities:
        path = _find_file(patient_dir, MODALITY_KEYWORDS[modality], exclude=MASK_KEYWORDS)
        if path is None:
            raise FileNotFoundError(f"Could not find '{modality}' volume in {patient_dir}")
        img = sitk.ReadImage(str(path), sitk.sitkFloat32)
        if bias_correct:
            img = _n4_bias_correct_sitk(img)
        raw_sitk[modality] = img

    ref_mod = REFERENCE_MODALITY if REFERENCE_MODALITY in modalities else modalities[0]
    fixed_sitk = raw_sitk[ref_mod]
    ref_arr = np.moveaxis(sitk.GetArrayFromImage(fixed_sitk).astype(np.float32), 0, -1)
    fixed_shape_nib = ref_arr.shape  # (H, W, Z) in nibabel convention

    volumes: dict = {}
    for modality, img in raw_sitk.items():
        if img.GetSize() == fixed_sitk.GetSize():
            volumes[modality] = np.moveaxis(sitk.GetArrayFromImage(img).astype(np.float32), 0, -1)
        elif register:
            resampled = _register_sitk(img, fixed_sitk)
            volumes[modality] = np.moveaxis(sitk.GetArrayFromImage(resampled).astype(np.float32), 0, -1)
        else:
            arr_nib = np.moveaxis(sitk.GetArrayFromImage(img).astype(np.float32), 0, -1)
            zoom_factors = [t / s for t, s in zip(fixed_shape_nib, arr_nib.shape)]
            volumes[modality] = _nd_zoom(arr_nib, zoom_factors, order=1)
    volumes["mask"] = np.zeros(fixed_shape_nib, dtype=np.uint8)
    return volumes


def predict_patient(
    patient_dir: Path,
    models: list,
    cfg: dict,
    device: torch.device,
    threshold: float,
    bias_correct: bool,
    register: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Preprocess patient and run inference. Returns (pred_3d, prob_3d, volumes)."""
    modalities = cfg["data"]["modalities"]
    image_size = cfg["data"]["image_size"]

    has_mask = True
    try:
        volumes = load_patient_volumes(patient_dir, modalities, bias_correct=bias_correct, register=register)
    except FileNotFoundError as e:
        if "mask" in str(e).lower():
            print("  No ground-truth mask found — running in prediction-only mode.")
            has_mask = False
            volumes = _load_volumes_no_mask(patient_dir, modalities, bias_correct, register=register)
        else:
            raise

    ref_vol = volumes[modalities[0]]
    n_slices = ref_vol.shape[2]
    h, w = ref_vol.shape[:2]

    normalized = {m: normalize_volume(volumes[m]) for m in modalities}

    prob_3d = np.zeros((n_slices, image_size, image_size), dtype=np.float32)
    active_slices = []

    for z in range(n_slices):
        stack = np.stack([normalized[m][:, :, z] for m in modalities], axis=0)
        if np.count_nonzero(stack[0]) < MIN_BRAIN_PIXELS:
            continue
        active_slices.append(z)
        stack_resized = _resize_stack(stack, image_size)
        tensor = torch.from_numpy(stack_resized).unsqueeze(0).to(device)

        if len(models) == 1:
            with torch.no_grad():
                logit = models[0](tensor)
            prob_3d[z] = torch.sigmoid(logit).squeeze().cpu().numpy()
        else:
            # Soft ensemble: average probabilities across all fold models
            probs = []
            for m in models:
                with torch.no_grad():
                    logit = m(tensor)
                probs.append(torch.sigmoid(logit).squeeze().cpu().numpy())
            prob_3d[z] = np.mean(probs, axis=0)

    pred_3d = (prob_3d > threshold).astype(np.uint8)
    volumes["_active_slices"] = active_slices
    volumes["_has_mask"] = has_mask
    return pred_3d, prob_3d, volumes


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    # pred: (Z, 256, 256); gt from NIfTI: (H, W, Z) native resolution
    gt = np.moveaxis(gt, -1, 0)  # -> (Z, H, W)
    if gt.shape != pred.shape:
        # resize pred to native resolution for an apples-to-apples comparison
        native_hw = gt.shape[1:]
        pred = np.stack(
            [sk_resize(pred[z], native_hw, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
             for z in range(pred.shape[0])],
            axis=0,
        )
    intersection = (pred * gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * intersection / denom) if denom > 0 else 1.0


def save_outputs(
    pred_3d: np.ndarray,
    volumes: dict,
    out_dir: Path,
    modalities: list,
    checkpoint_label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save predicted mask as NIfTI (in FLAIR-resampled space)
    nib.save(
        nib.Nifti1Image(pred_3d.astype(np.uint8), affine=np.eye(4)),
        str(out_dir / "pred_mask.nii.gz"),
    )

    active_slices: list = volumes.get("_active_slices", [])
    has_mask: bool = volumes.get("_has_mask", True)
    gt_mask = volumes.get("mask", np.zeros_like(pred_3d))

    # Choose which slices to display: prefer lesion-positive predictions, else first 6 active
    display_slices = [z for z in active_slices if pred_3d[z].any()]
    if not display_slices:
        display_slices = active_slices[:6]
    display_slices = display_slices[:8]  # cap at 8 rows

    n = len(display_slices)
    n_cols = 3 if has_mask else 2
    if n > 0:
        fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        ref_mod = modalities[-1]  # FLAIR as display channel
        for row, z in enumerate(display_slices):
            img_slice = volumes[ref_mod][:, :, z]
            norm_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            axes[row, 0].imshow(norm_slice, cmap="gray")
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("MRI (FLAIR)")

            native_hw = img_slice.shape[:2]
            pred_native = sk_resize(pred_3d[z], native_hw, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

            col = 1
            if has_mask:
                overlay_mask(axes[row, col], img_slice, gt_mask[:, :, z], color=(0, 1, 0))
                if row == 0:
                    axes[row, col].set_title("Ground truth")
                col += 1

            overlay_mask(axes[row, col], img_slice, pred_native, color=(1, 0, 0))
            if row == 0:
                axes[row, col].set_title("Prediction")

        fig.tight_layout()
        fig.savefig(out_dir / "overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved overlay.png ({n} slices)")

    # Summary JSON
    summary: dict = {
        "checkpoint_used": checkpoint_label,
        "n_slices_with_brain": len(active_slices),
        "n_slices_with_predicted_lesion": int(pred_3d.any(axis=(1, 2)).sum()),
        "total_lesion_voxels": int(pred_3d.sum()),
    }
    if has_mask:
        summary["dice"] = _dice(pred_3d, gt_mask)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Summary: {summary}")


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = PROJECT_ROOT / cfg["output"]["checkpoint_dir"]

    if args.ensemble:
        n_folds = cfg["split"]["n_folds"]
        models = load_ensemble(checkpoint_dir, n_folds, cfg, device)
        checkpoint_label = f"ensemble of {len(models)} folds"
        print(f"Loaded ensemble: {len(models)} fold models")
    else:
        ckpt = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "fold0_best.pt"
        models = [load_model(ckpt, cfg, device)]
        checkpoint_label = str(ckpt)
        print(f"Loaded checkpoint: {ckpt}")

    patient_dir = Path(args.patient_dir)
    print(f"Preprocessing {patient_dir} ...")
    pred_3d, prob_3d, volumes = predict_patient(
        patient_dir,
        models,
        cfg,
        device,
        threshold=args.threshold,
        bias_correct=not args.skip_bias_correction,
        register=not args.skip_registration,
    )

    out_dir = Path(args.out_dir)
    print(f"Saving outputs to {out_dir} ...")
    save_outputs(pred_3d, volumes, out_dir, cfg["data"]["modalities"], checkpoint_label)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", required=True, type=str, help="Folder with T1/T2/FLAIR NIfTI files.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a single fold checkpoint.")
    parser.add_argument("--ensemble", action="store_true", help="Use all fold checkpoints (soft voting).")
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "outputs" / "predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-bias-correction", action="store_true")
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Use proportional zoom instead of SimpleITK rigid registration (faster, lower quality).",
    )
    args = parser.parse_args()

    if not args.ensemble and not args.checkpoint:
        print("No --checkpoint specified and --ensemble not set; defaulting to fold0_best.pt.")
    main(args)
