# MS Lesion Segmentation

2D U-Net study for multiple sclerosis lesion segmentation on brain MRI, using the Kaggle dataset
[`orvile/multiple-sclerosis-brain-mri-lesion-segmentation`](https://www.kaggle.com/datasets/orvile/multiple-sclerosis-brain-mri-lesion-segmentation)
(60 patients, T1/T2/FLAIR + consensus lesion masks, NIfTI format).

## Project structure

```
data/            raw NIfTI downloads and preprocessed 2D slices (gitignored)
notebooks/       EDA, preprocessing dev, results report
src/data/        download + preprocessing + PyTorch Dataset/split
src/models/      2D U-Net
src/utils/       metrics (Dice/IoU/sensitivity/precision) and visualization
src/train.py     training loop (config-driven, device-agnostic)
src/evaluate.py  cross-fold evaluation + prediction figures
configs/         hyperparameters (baseline.yaml)
outputs/         checkpoints (gitignored) and figures (versioned)
```

## Two-machine workflow

This project is developed across two computers, kept in sync via this Git repo:

- **Dev machine** (no NVIDIA GPU): write/debug code here, run small smoke tests on CPU.
- **Training machine** (16GB VRAM GPU): `git pull`, download the real dataset, run full training.

```
# on the dev machine
git add -A && git commit -m "..." && git push

# on the GPU machine
git pull
python src/data/download.py          # only needed once, or after wiping data/raw
python src/data/preprocessing.py
python src/train.py --config configs/baseline.yaml

# after training, sync results back
git add outputs/figures configs
git commit -m "Add training results"
git push

# back on the dev machine
git pull   # inspect outputs/figures in notebooks/03_results_report.ipynb
```

Data (`data/raw/`, `data/processed/`) and model checkpoints (`outputs/checkpoints/`) are gitignored —
they're regenerated locally on each machine, not pushed through Git.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Kaggle API credentials

`src/data/download.py` uses the `kaggle` CLI, which needs an API token:

1. Go to your Kaggle account settings → "Create New API Token" → downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars).
3. Never commit `kaggle.json` (already in `.gitignore`).

## Running the pipeline

```
python src/data/download.py                       # -> data/raw/
python src/data/preprocessing.py                   # -> data/processed/ (2D slices + index.csv)
python src/train.py --config configs/baseline.yaml  # -> outputs/checkpoints/
python src/evaluate.py --config configs/baseline.yaml  # -> outputs/figures/
```

### Smoke test (fast, CPU-friendly, run on the dev machine)

```
python src/data/preprocessing.py --patient-limit 4
python src/train.py --config configs/baseline.yaml --smoke-test
python src/evaluate.py --config configs/baseline.yaml --smoke-test
```

Runs on 4 patients / 2 epochs to validate the full pipeline runs end-to-end before
launching a real training run on the GPU machine.

## Notebooks

- `01_eda.ipynb` — inspect volumes, visualize modalities + lesion masks, lesion burden distribution.
- `02_preprocessing_dev.ipynb` — debug preprocessing on 2-3 patients before running it on all 60.
- `03_results_report.ipynb` — training curves, metrics table, prediction overlays, discussion.

## Method notes

- **Task**: 2D axial-slice binary segmentation (lesion vs. background) with a from-scratch U-Net.
- **Split**: patient-level k-fold (default 5-fold) — slices from the same patient never span
  train/val within a fold, enforced by an assertion in `src/data/dataset.py`.
- **Loss**: combined Dice + BCE, standard for imbalanced medical segmentation.
- **Metrics**: Dice, IoU, sensitivity, precision, reported per fold and as mean ± std across folds.
