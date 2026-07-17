"""Turn raw per-patient NIfTI volumes (T1/T2/FLAIR + consensus lesion mask) into
normalized, resized 2D axial slices saved as .npy, plus an index CSV describing
every slice (patient id, slice index, path, whether it contains lesion pixels).

The exact file naming inside each patient folder varies by dataset release, so
files are matched by keyword rather than a fixed filename.
"""

import argparse
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
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


def _register_sitk(moving: sitk.Image, fixed: sitk.Image) -> sitk.Image:
    """Register `moving` sitk.Image onto `fixed` sitk.Image's grid (rigid, Mattes MI).

    Reads spacing/origin/direction from the sitk images directly, preserving the full
    NIfTI spatial metadata (including different FOVs and origins across modalities).

    Pyramid depth is capped so the smallest z-extent (fixed or moving) is never shrunk
    below 8 slices — smaller z-extents cause the optimizer to diverge and lose overlap
    with the next pyramid level. Falls back to GEOMETRY-only resampling (no optimization)
    if the MI metric still fails despite these precautions.
    """
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Only use a multi-resolution pyramid when both images are large enough in z.
    # Shrink=2 requires min_z >= 16 to keep >= 8 slices at the coarse level.
    min_z = min(fixed.GetSize()[2], moving.GetSize()[2])
    if min_z >= 16:
        shrink_factors = [2, 1]
        smooth_sigmas = [1, 0]
    else:
        shrink_factors = [1]
        smooth_sigmas = [0]

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100, convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smooth_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    try:
        transform = registration.Execute(fixed, moving)
    except RuntimeError:
        # Registration failed (e.g. extreme FOV mismatch with tiny z-extent).
        # Fall back to the GEOMETRY initializer alone — center alignment without
        # optimization. Better than zoom and preserves spatial metadata.
        transform = initial_transform

    return sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID())


def _n4_bias_correct_sitk(image: sitk.Image, shrink_factor: int = 4) -> sitk.Image:
    """N4 bias field correction on a sitk.Image, preserving full spatial metadata.

    Runs on a shrunk copy for speed, reconstructs the corrected full-resolution image
    from the estimated log bias field. Uses nonzero voxels as the foreground mask.
    """
    array = sitk.GetArrayFromImage(image)
    mask = sitk.GetImageFromArray((array > 0).astype(np.uint8))
    mask.CopyInformation(image)
    if not array.any():
        return image

    image_ds = sitk.Shrink(image, [shrink_factor] * image.GetDimension())
    mask_ds = sitk.Shrink(mask, [shrink_factor] * image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(image_ds, mask_ds)

    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected = image / sitk.Exp(log_bias_field)

    # Zero out background while keeping spatial metadata
    corrected_array = sitk.GetArrayFromImage(corrected).astype(np.float32)
    corrected_array[array <= 0] = 0
    corrected_sitk = sitk.GetImageFromArray(corrected_array)
    corrected_sitk.CopyInformation(image)
    return corrected_sitk


def _n4_bias_correct(volume: np.ndarray, shrink_factor: int = 4) -> np.ndarray:
    """Correct MRI intensity inhomogeneity (bias field) with N4ITK.

    Wrapper around _n4_bias_correct_sitk for callers that work with numpy arrays
    (no spatial metadata needed — internal use only when registration is skipped).
    """
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    corrected = _n4_bias_correct_sitk(image, shrink_factor)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def load_patient_volumes(
    patient_dir: Path,
    modalities: list[str] | None = None,
    reference_modality: str = REFERENCE_MODALITY,
    bias_correct: bool = True,
    register: bool = True,
) -> dict[str, np.ndarray]:
    """Load each requested modality volume plus a consensus lesion mask.

    T1/T2/FLAIR are each acquired/stored at their own native resolution and
    slice count in this dataset (not mutually co-registered), so every
    modality other than `reference_modality` is brought onto the reference
    modality's grid.

    When `register=True` (default), uses SimpleITK rigid registration with
    Mattes mutual information -- true anatomical alignment. When `register=False`,
    falls back to proportional scipy.ndimage.zoom (faster but approximate).

    Each modality is N4 bias-field-corrected in its own native resolution
    (before registration/resampling) unless `bias_correct=False`.
    """
    modalities = modalities or list(MODALITY_KEYWORDS.keys())
    reference_modality = reference_modality if reference_modality in modalities else modalities[0]

    raw_sitk: dict[str, sitk.Image] = {}
    for modality in modalities:
        path = _find_file(patient_dir, MODALITY_KEYWORDS[modality], exclude=MASK_KEYWORDS)
        if path is None:
            raise FileNotFoundError(f"Could not find '{modality}' volume in {patient_dir}")
        # ReadImage preserves spacing, origin, and direction cosines — required for correct
        # inter-modality registration when FOVs or origins differ across modalities.
        img = sitk.ReadImage(str(path), sitk.sitkFloat32)
        if bias_correct:
            img = _n4_bias_correct_sitk(img)
        raw_sitk[modality] = img

    mask_path = _find_mask_file(patient_dir, reference_modality)
    if mask_path is None:
        raise FileNotFoundError(f"Could not find lesion mask in {patient_dir}")
    mask = (np.asarray(nib.load(str(mask_path)).dataobj) > 0).astype(np.uint8)

    fixed_sitk = raw_sitk[reference_modality]
    # sitk.GetArrayFromImage returns (Z, Y, X); convert to nibabel convention (X, Y, Z)
    # so the rest of the pipeline can slice with volume[:, :, z] unchanged.
    ref_arr = np.moveaxis(sitk.GetArrayFromImage(fixed_sitk).astype(np.float32), 0, -1)
    fixed_shape_nib = ref_arr.shape  # (H, W, Z) in nibabel convention

    volumes: dict[str, np.ndarray] = {}
    for modality, img in raw_sitk.items():
        if img.GetSize() == fixed_sitk.GetSize():
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            volumes[modality] = np.moveaxis(arr, 0, -1)
        elif register:
            resampled = _register_sitk(img, fixed_sitk)
            arr = sitk.GetArrayFromImage(resampled).astype(np.float32)
            volumes[modality] = np.moveaxis(arr, 0, -1)
        else:
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            arr_nib = np.moveaxis(arr, 0, -1)  # (H, W, Z)
            zoom_factors = [t / s for t, s in zip(fixed_shape_nib, arr_nib.shape)]
            volumes[modality] = nd_zoom(arr_nib, zoom_factors, order=1)
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
    neg_ratio: float | None = 1.5,
    rng: random.Random | None = None,
) -> list[dict]:
    """Extract 2D axial slices for one patient.

    `neg_ratio` caps how many lesion-free brain slices are kept per patient,
    as a multiple of that patient's lesion-containing slice count (e.g. 1.5
    keeps at most 1.5x as many empty slices as lesion slices, randomly
    sampled). Only ~54% of brain slices in this dataset contain any lesion
    pixel, and lesion pixels are a tiny fraction even within those -- capping
    the empty-slice majority reduces that imbalance at the dataset level.
    Set neg_ratio=None to keep every brain slice (previous behaviour).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = rng or random.Random()
    n_slices = volumes[modalities[0]].shape[2]

    normalized = {m: normalize_volume(volumes[m]) for m in modalities}

    candidates = []
    for z in range(n_slices):
        stack = np.stack([normalized[m][:, :, z] for m in modalities], axis=0)  # (C, H, W)
        mask_slice = volumes["mask"][:, :, z]

        if np.count_nonzero(stack[0]) < min_brain_pixels:
            continue  # skip empty (no-brain) slices

        candidates.append((z, stack, mask_slice, bool(mask_slice.sum() > 0)))

    positive = [c for c in candidates if c[3]]
    negative = [c for c in candidates if not c[3]]
    if neg_ratio is not None and negative:
        n_keep = min(len(negative), round(len(positive) * neg_ratio))
        negative = rng.sample(negative, n_keep)
    selected = sorted(positive + negative, key=lambda c: c[0])

    records = []
    for z, stack, mask_slice, _ in selected:
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
    bias_correct: bool = True,
    register: bool = True,
    neg_ratio: float | None = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    modalities = modalities or ["t1", "t2", "flair"]
    patient_dirs = discover_patients(raw_dir)
    if not patient_dirs:
        raise FileNotFoundError(
            f"No NIfTI files found under {raw_dir}. Run src/data/download.py first."
        )
    if patient_limit:
        patient_dirs = patient_dirs[:patient_limit]

    rng = random.Random(seed)
    all_records = []
    for patient_dir in patient_dirs:
        patient_id = patient_id_from_dir(patient_dir)
        print(f"Processing patient {patient_id} ({patient_dir}) ...")
        volumes = load_patient_volumes(patient_dir, modalities, bias_correct=bias_correct, register=register)
        records = extract_patient_slices(
            patient_id, volumes, out_dir, modalities, image_size, neg_ratio=neg_ratio, rng=rng
        )
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
    parser.add_argument(
        "--skip-bias-correction",
        action="store_true",
        help="Skip N4 bias field correction (faster, e.g. for smoke tests).",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Use proportional zoom instead of SimpleITK rigid registration (faster, lower quality).",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=1.5,
        help="Max lesion-free slices to keep per patient, as a multiple of that "
        "patient's lesion-containing slice count. Use a negative value (e.g. -1) "
        "to keep every brain slice (no balancing).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for empty-slice subsampling.")
    args = parser.parse_args()
    run(
        args.raw_dir,
        args.out_dir,
        args.modalities,
        args.image_size,
        args.patient_limit,
        bias_correct=not args.skip_bias_correction,
        register=not args.skip_registration,
        neg_ratio=None if args.neg_ratio < 0 else args.neg_ratio,
        seed=args.seed,
    )
