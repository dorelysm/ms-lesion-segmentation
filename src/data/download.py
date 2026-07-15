"""Download the MS Brain MRI Lesion Segmentation dataset from Kaggle into data/raw/.

Requires a Kaggle API token (kaggle.json) configured as described in the README
(either at ~/.kaggle/kaggle.json or via the KAGGLE_USERNAME/KAGGLE_KEY env vars).
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

DATASET_SLUG = "orvile/multiple-sclerosis-brain-mri-lesion-segmentation"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    if any(raw_dir.iterdir()) and not force:
        print(f"{raw_dir} is not empty, skipping download (use --force to re-download).")
        return

    zip_path = raw_dir / "dataset.zip"
    print(f"Downloading {DATASET_SLUG} to {zip_path} ...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(raw_dir),
            "--force" if force else "--skip",
        ],
        check=True,
    )

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()

    print("Done. Top-level contents of data/raw:")
    for p in sorted(raw_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Where to extract the dataset."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if raw-dir is not empty."
    )
    args = parser.parse_args()
    download(args.raw_dir, args.force)
