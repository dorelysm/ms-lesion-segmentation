"""Turn raw per-patient NIfTI volumes (T1/T2/FLAIR + consensus lesion mask) into
normalized, resized 2D axial slices saved as .npy, plus an index CSV describing
every slice (patient id, slice index, path, whether it contains lesion pixels).

The exact file naming inside each patient folder varies by dataset release, so
files are matched by keyword rather than a fixed filename.
"""

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom as nd_zoom
from skimage.transform import resize as sk_resize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

MODALITY_KEYWORDS = {
    "t1": ["t1"],
    "t2": ["t2"],
    "flair": ["flair"],
}
MASK_KEYWORDS = ["mask", "consensus", "lesion", "gt", "label", "seg"]

# T1/T2/FLAIR are not co-registered in this dataset -- each modality has its own
# native resolution and slice count per patient. The lesion mask is provided
# per-modality-space; we treat the reference modality's space as the common grid
# and resample the other modalities onto it (see load_patient_volumes).
REFERENCE_MODALITY = "flair"

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def _find_file(patient_dir: Path, keywords: list[str], exclude: list[str] | None = None) -> Path | None:
    exclude = exclude or []
    candidates = [
        p
        for p in patient_dir.rglob("*")
        if p.is_file()
        and p.name.lower().endswith(NIFTI_SUFFIXES)
        and any(k in p.name.lower() for k in keywords)
        and not any(e in p.name.lower() for e in exclude)
    ]
    if not candidates:
        return None
    return sorted(candidates)[0]


def discover_patients(raw_dir: Path) -> list[Path]:
    """A patient folder is any directory containing at least one NIfTI file."""
    patient_dirs = sorted(
        {p.parent for p in raw_dir.rglob("*") if p.name.lower().endswith(NIFTI_SUFFIXES)}
    )
    return patient_dirs


def patient_id_from_dir(patient_dir: Path) -> str:
    match = re.search(r"(\d+)", patient_dir.name)
    return match.group(1).zfill(3) if match else patient_dir.name


def _find_mask_file(patient_dir: Path, reference_modality: str) -> Path | None:
    """Find the lesion mask registered to `reference_modality`'s native space
    (mask filenames embed the modality they were resampled to, e.g. LesionSeg-Flair)."""
    ref_keywords = MODALITY_KEYWORDS.get(reference_modality, [reference_modality])
    candidates = [
        p
        for p in patient_dir.rglob("*")
        if p.is_file()
        and p.name.lower().endswith(NIFTI_SUFFIXES)
        and any(k in p.name.lower() for k in MASK_KEYWORDS)
        and any(k in p.name.lower() for k in ref_keywords)
    ]
    if not candidates:
        return None
    return sorted(candidates)[0]


def load_patient_volumes(
    patient_dir: Path,
    modalities: list[str] | None = None,
    reference_modality: str = REFERENCE_MODALITY,
) -> dict[str, np.ndarray]:
    """Load each requested modality volume plus a consensus lesion mask.

    T1/T2/FLAIR are each acquired/stored at their own native resolution and
    slice count in this dataset (not mutually co-registered), so every
    modality other than `reference_modality` is resampled (scipy.ndimage.zoom,
    trilinear) onto the reference modality's grid -- which is also the space
    the lesion mask is provided in. This is an approximation (proportional
    resampling, not true anatomical registration).
    """
    modalities = modalities or list(MODALITY_KEYWORDS.keys())
    reference_modality = reference_modality if reference_modality in modalities else modalities[0]

    raw_volumes: dict[str, np.ndarray] = {}
    for modality in modalities:
        path = _find_file(patient_dir, MODALITY_KEYWORDS[modality], exclude=MASK_KEYWORDS)
        if path is None:
            raise FileNotFoundError(f"Could not find '{modality}' volume in {patient_dir}")
        raw_volumes[modality] = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)

    mask_path = _find_mask_file(patient_dir, reference_modality)
    if mask_path is None:
        raise FileNotFoundError(f"Could not find lesion mask in {patient_dir}")
    mask = (np.asarray(nib.load(str(mask_path)).dataobj) > 0).astype(np.uint8)

    target_shape = raw_volumes[reference_modality].shape
    volumes: dict[str, np.ndarray] = {}
    for modality, volume in raw_volumes.items():
        if volume.shape == target_shape:
            volumes[modality] = volume
        else:
            zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
            volumes[modality] = nd_zoom(volume, zoom_factors, order=1)
    volumes["mask"] = mask
    return volumes


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization using only non-background (nonzero) voxels."""
    brain_voxels = volume[volume > 0]
    if brain_voxels.size == 0:
        return volume
    mean, std = brain_voxels.mean(), brain_voxels.std()
    std = std if std > 1e-6 else 1.0
    normalized = (volume - mean) / std
    normalized[volume == 0] = 0
    return normalized


def extract_patient_slices(
    patient_id: str,
    volumes: dict[str, np.ndarray],
    out_dir: Path,
    modalities: list[str],
    image_size: int,
    min_brain_pixels: int = 100,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_slices = volumes[modalities[0]].shape[2]
    records = []

    normalized = {m: normalize_volume(volumes[m]) for m in modalities}

    for z in range(n_slices):
        stack = np.stack([normalized[m][:, :, z] for m in modalities], axis=0)  # (C, H, W)
        mask_slice = volumes["mask"][:, :, z]

        if np.count_nonzero(stack[0]) < min_brain_pixels:
            continue  # skip empty (no-brain) slices

        stack_resized = np.stack(
            [sk_resize(c, (image_size, image_size), preserve_range=True, anti_aliasing=True) for c in stack],
            axis=0,
        ).astype(np.float32)
        mask_resized = sk_resize(
            mask_slice, (image_size, image_size), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

        image_path = out_dir / f"{patient_id}_slice{z:03d}_img.npy"
        mask_path = out_dir / f"{patient_id}_slice{z:03d}_mask.npy"
        np.save(image_path, stack_resized)
        np.save(mask_path, mask_resized)

        records.append(
            {
                "patient_id": patient_id,
                "slice_idx": z,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "has_lesion": bool(mask_resized.sum() > 0),
            }
        )
    return records


def run(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    modalities: list[str] | None = None,
    image_size: int = 256,
    patient_limit: int | None = None,
) -> pd.DataFrame:
    modalities = modalities or ["t1", "t2", "flair"]
    patient_dirs = discover_patients(raw_dir)
    if not patient_dirs:
        raise FileNotFoundError(
            f"No NIfTI files found under {raw_dir}. Run src/data/download.py first."
        )
    if patient_limit:
        patient_dirs = patient_dirs[:patient_limit]

    all_records = []
    for patient_dir in patient_dirs:
        patient_id = patient_id_from_dir(patient_dir)
        print(f"Processing patient {patient_id} ({patient_dir}) ...")
        volumes = load_patient_volumes(patient_dir, modalities)
        records = extract_patient_slices(patient_id, volumes, out_dir, modalities, image_size)
        all_records.extend(records)
        print(f"  -> {len(records)} slices ({sum(r['has_lesion'] for r in records)} with lesion)")

    index_df = pd.DataFrame(all_records)
    index_path = out_dir / "index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"\nWrote index with {len(index_df)} slices from {index_df['patient_id'].nunique()} patients to {index_path}")
    return index_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--modalities", nargs="+", default=["t1", "t2", "flair"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patient-limit", type=int, default=None, help="For smoke tests.")
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir, args.modalities, args.image_size, args.patient_limit)
