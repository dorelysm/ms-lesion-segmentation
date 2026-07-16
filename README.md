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
.kaggle/         project-local Kaggle API token (gitignored, see below)
```

## GPU setup (AMD ROCm on Windows)

This project trains on an **AMD Radeon RX 9060 XT** via AMD's official "PyTorch on Windows"
ROCm 7.2.1 build, which preserves the standard `torch.cuda` API (`train.py`/`evaluate.py`
need no code changes to use the GPU — they already do
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

Requirements:
- **Python 3.12** exactly (the ROCm wheels are built for `cp312`; other Python versions won't work).
- **AMD graphics driver 26.2.2+** (check via Windows Settings → About, or the AMD Software / Adrenalin app).
- **Visual Studio Build Tools**, "Desktop development with C++" workload + a Windows SDK — MIOpen's
  HIPRTC JIT kernel compiler needs the MSVC standard library headers on Windows. Without this, GPU ops
  like `BatchNorm` fail with `miopenStatusUnknownError` / `'type_traits' file not found`. Install via:
  ```
  # download https://aka.ms/vs/17/release/vs_buildtools.exe, then:
  vs_buildtools.exe --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended
  ```

Environment setup:
```powershell
py -3.12 -m venv .venv-rocm
.venv-rocm\Scripts\activate        # Windows

# ROCm SDK (~1.4 GB)
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

# PyTorch/torchvision/torchaudio with ROCm support (~823 MB) -- do this BEFORE `pip install -r requirements.txt`,
# since the plain `torch`/`torchvision` entries there would otherwise install CPU-only wheels over these.
pip install --no-cache-dir `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl `
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl

pip install -r requirements.txt
```

Verify:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True, AMD Radeon RX 9060 XT
```

Other AMD Windows GPUs (RX 7000/9000 series) supported by the same ROCm 7.2.1 build should work the
same way — check AMD's [Windows compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
for your card. On a machine with an **NVIDIA GPU** instead, skip this whole section and just
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` (or the current
CUDA index) before `pip install -r requirements.txt` — the rest of the pipeline is unchanged either way.

Without any GPU, everything still runs on CPU (`--smoke-test` is fast enough for that); a full 5-fold
real run is only practical with a GPU.

## Kaggle API credentials

`src/data/download.py` uses the `kaggle` CLI (v2.x), which needs an API token. Kaggle's current token
format (`KGAT_...`) is read from an `access_token` file, resolved via `KAGGLE_CONFIG_DIR` (defaults to
`~/.kaggle`).

This repo scopes it to the project instead of your global `~/.kaggle`: `.venv-rocm\Scripts\Activate.ps1`
sets `KAGGLE_CONFIG_DIR` to `<project root>\.kaggle` automatically on activation.

1. Go to your Kaggle account → Settings → API → **Create New Token**.
2. Save the token value into `.kaggle/access_token` at the project root (create the folder if needed):
   ```
   echo YOUR_TOKEN > .kaggle/access_token
   ```
3. Never commit this file — `.kaggle/` is already in `.gitignore`.

(The classic `kaggle.json` with `KAGGLE_USERNAME`/`KAGGLE_KEY` still works too, if you have one from an
older token.)

## Running the pipeline

```
python src/data/download.py                        # -> data/raw/
python src/data/preprocessing.py                    # -> data/processed/ (2D slices + index.csv)
python src/train.py --config configs/baseline.yaml  # -> outputs/checkpoints/
python src/evaluate.py --config configs/baseline.yaml  # -> outputs/figures/
```

`configs/baseline.yaml` runs the full 5-fold cross-validation by default (`folds_to_run: [0,1,2,3,4]`).
Set it back to e.g. `[0]` for a single-fold run.

`preprocessing.py` options (all optional):
- `--skip-bias-correction` — skip N4 bias field correction (faster; useful for smoke tests).
- `--neg-ratio 1.5` — cap lesion-free slices per patient at this multiple of that patient's
  lesion-containing slice count (pass a negative value to disable and keep every brain slice).
- `--seed 42` — seed for the empty-slice subsampling.
- `--patient-limit N` — for smoke tests.

### Smoke test (fast, CPU-friendly)

```
python src/data/preprocessing.py --patient-limit 4 --skip-bias-correction
python src/train.py --config configs/baseline.yaml --smoke-test
python src/evaluate.py --config configs/baseline.yaml --smoke-test
```

Runs on 4 patients / 2 epochs to validate the full pipeline runs end-to-end before launching a real run.

## Notebooks

- `01_eda.ipynb` — inspect volumes, visualize modalities + lesion masks, lesion burden distribution.
- `02_preprocessing_dev.ipynb` — debug preprocessing on 2-3 patients before running it on all 60.
- `03_results_report.ipynb` — training curves, metrics table, prediction overlays, critical analysis
  of results, and prioritized future-improvement notes.

## Method notes

- **Task**: 2D axial-slice binary segmentation (lesion vs. background) with a from-scratch U-Net.
- **Modalities**: T1/T2/FLAIR are *not* co-registered in this dataset (each has its own native
  resolution/slice count per patient) — `preprocessing.py` N4-corrects each modality in its own space,
  then resamples onto the reference (FLAIR) grid via `scipy.ndimage.zoom` (a proportional approximation,
  not true anatomical registration).
- **Split**: patient-level k-fold (default 5-fold) — slices from the same patient never span
  train/val within a fold, enforced by an assertion in `src/data/dataset.py`.
- **Loss**: combined Dice + BCE, standard for imbalanced medical segmentation.
- **Metrics**: Dice, IoU, sensitivity, precision, reported per fold and as mean ± std across folds.

## Current results

Full 5-fold run on GPU (see `notebooks/03_results_report.ipynb` for the full critical analysis):

| Metric | Mean ± std |
|---|---|
| Dice | 0.553 ± 0.051 |
| IoU | 0.433 ± 0.042 |
| Sensitivity | 0.578 ± 0.040 |
| Precision | 0.661 ± 0.027 |
