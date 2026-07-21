# MS Lesion Segmentation

La esclerosis múltiple (EM) es una enfermedad neurológica en la que el sistema inmunitario ataca la
mielina del cerebro y la médula espinal, dejando cicatrices visibles en la resonancia magnética
llamadas lesiones. Los neurólogos monitorean el avance de la enfermedad contando y midiendo esas
lesiones en cada nueva resonancia — un proceso que puede tomar 20–40 minutos por paciente, y donde
dos radiólogos distintos a veces no coinciden exactamente. Este proyecto entrena una red neuronal
para hacer esa misma delimitación automáticamente, a partir de ejemplos previamente anotados por
expertos.

**En concreto**: un 2D U-Net entrenado sobre MRI cerebrales (T1/T2/FLAIR) de 60 pacientes con EM,
con validación cruzada de 5 pliegues a nivel de paciente. La U-Net es un tipo de red neuronal
diseñada específicamente para segmentación de imágenes médicas: aprende a identificar patrones
visuales en miles de ejemplos anotados y luego aplica ese conocimiento en imágenes nuevas.

**Result**: Dice 0.563 ± 0.047, sensitivity 0.604 across 5 folds (60 patients). See
[Results](#results) below and the full critical analysis in
[`notebooks/03_results_report.ipynb`](notebooks/03_results_report.ipynb).

> **This is a research/educational project, not a clinical tool.** It has not been validated
> externally, was trained on a single 60-patient public dataset from ~20 centers, and should not be
> used to inform any real diagnostic or treatment decision. See [Limitations](#limitations--future-work).

## Results

![Prediction examples: MRI slice, ground truth, and model prediction overlays](outputs/figures/prediction_examples.png)

*Cada fila es un corte axial del cerebro. Izquierda: imagen MRI (canal FLAIR). Centro: anotación del
radiólogo (lesión en blanco). Derecha: predicción del modelo (lesión en blanco). Las áreas donde
ambos coinciden son las lesiones detectadas correctamente.*

![Training and validation loss/Dice curves per fold](outputs/figures/training_curves.png)

| Metric | Mean ± std (5-fold) |
|---|---|
| Dice | 0.563 ± 0.047 |
| IoU | 0.441 ± 0.040 |
| Sensitivity | 0.604 ± 0.038 |
| Precision | 0.640 ± 0.034 |

> **¿Qué significa Dice = 0.563?** El coeficiente Dice mide el solapamiento entre la predicción del
> modelo y la anotación del radiólogo: 1.0 sería coincidencia perfecta, 0.0 significa que no detecta
> nada. Un valor de 0.563 indica que el modelo y el especialista coinciden en aproximadamente el 56%
> del área de lesión — suficiente para asistir como herramienta de screening, pero no para reemplazar
> la revisión clínica. Para contexto, la variabilidad entre radiólogos humanos en esta tarea suele
> dar Dice de 0.60–0.70.

Best result from 6 training experiments (see [`outputs/experiments.md`](outputs/experiments.md)). Note: a preprocessing axis bug (nibabel vs SimpleITK convention mismatch) was discovered and fixed after Exp 5; the numbers above reflect the corrected retraining.

## What we found

The best model (Exp 4) uses Tversky loss with α=0.6, which penalizes missed lesions more than false positives. This raised sensitivity by +0.048 over the baseline — meaning the model correctly identifies more of the actual lesion area — while keeping precision high enough to avoid flooding predictions with false alarms. Adding a learning rate scheduler and early stopping prevented overfitting and stabilized results across folds.

**Two experiments that didn't work — and why they're interesting:**

- *Anatomical registration* (Exp 5, Dice 0.132): replacing the simple image rescaling with proper 3D registration collapsed performance. The reason: this dataset's MRI files have placeholder coordinate headers (all zeros), so the registration optimizer had no real patient-space information to work with. Lesson: anatomical registration only helps when the NIfTI files contain real scanner coordinates.
- *Intensity augmentation* (Exp 6, Dice 0.175): randomly distorting image contrast during training made things worse, not better. The distortion range was too wide — training images looked so different from validation images that the model learned the wrong thing. Lesson: data augmentation needs to stay close enough to the real distribution.

**The persistent outlier (fold 1):** one validation fold consistently scores ~0.07 lower than the others. Investigation shows this fold happens to contain two patients with unusually tiny lesions (~4×5 pixel blobs at 256×256 resolution), which are near-impossible to detect reliably regardless of the model. This is a data split artifact, not a model failure.

**Next steps (prioritized):**

1. Intensity augmentation with conservative ranges (gamma ±10%, brightness ±5%) — same idea as Exp 6, but without the distribution mismatch that caused the regression.
2. 2.5D architecture — feed adjacent slices as extra input channels to give the model spatial context across slices, which should help with small lesions that only appear in 1–2 consecutive slices.
3. Stratified split — distribute the hard small-lesion patients evenly across folds so per-fold variance better reflects model quality rather than data luck.

Full analysis with numbers in [`notebooks/03_results_report.ipynb`](notebooks/03_results_report.ipynb).

## Project structure

```
data/            raw NIfTI downloads and preprocessed 2D slices (gitignored)
notebooks/       EDA, preprocessing dev, results report
src/data/        download + preprocessing (N4 bias correction, resampling) + PyTorch Dataset/split
src/models/      2D U-Net
src/utils/       metrics (Dice/IoU/sensitivity/precision) and visualization
src/train.py     training loop (config-driven, device-agnostic)
src/evaluate.py  cross-fold evaluation + prediction figures
configs/         hyperparameters (baseline.yaml)
outputs/         checkpoints (gitignored) and figures (versioned)
docs/            GPU setup details (AMD ROCm / NVIDIA CUDA)
.kaggle/         project-local Kaggle API token (gitignored, see below)
```

## Quickstart

```
python -m venv .venv-rocm   # or .venv -- see docs/SETUP_GPU.md for GPU-specific setup (AMD ROCm / NVIDIA)
.venv-rocm\Scripts\activate
pip install -r requirements.txt

python src/data/download.py                        # -> data/raw/
python src/data/preprocessing.py                    # -> data/processed/ (2D slices + index.csv)
python src/train.py --config configs/baseline.yaml  # -> outputs/checkpoints/
python src/evaluate.py --config configs/baseline.yaml  # -> outputs/figures/
```

**GPU setup** (required for a full run in reasonable time) is hardware-specific — see
[`docs/SETUP_GPU.md`](docs/SETUP_GPU.md) for AMD ROCm (what this project used) and NVIDIA CUDA
instructions. Without a GPU, everything still runs on CPU via the smoke test below.

`configs/baseline.yaml` runs the full 5-fold cross-validation by default (`folds_to_run: [0,1,2,3,4]`).

`preprocessing.py` options (all optional):
- `--skip-bias-correction` — skip N4 bias field correction (faster; useful for smoke tests).
- `--skip-registration` — use proportional zoom instead of SimpleITK rigid registration (much faster; useful for smoke tests).
- `--neg-ratio 1.5` — cap lesion-free slices per patient at this multiple of that patient's
  lesion-containing slice count (pass a negative value to disable and keep every brain slice).
- `--seed 42` — seed for the empty-slice subsampling.
- `--patient-limit N` — for smoke tests.

### Smoke test (fast, CPU-friendly)

```
python src/data/preprocessing.py --patient-limit 4 --skip-bias-correction --skip-registration
python src/train.py --config configs/baseline.yaml --smoke-test
python src/evaluate.py --config configs/baseline.yaml --smoke-test
```

Runs on 4 patients / 2 epochs to validate the full pipeline end-to-end before launching a real run.

## Kaggle API credentials

`src/data/download.py` uses the `kaggle` CLI (v2.x), which needs an API token. Kaggle's current token
format (`KGAT_...`) is read from an `access_token` file, resolved via `KAGGLE_CONFIG_DIR` (defaults to
`~/.kaggle`).

This repo scopes it to the project instead of your global `~/.kaggle`:
`.venv-rocm\Scripts\Activate.ps1` sets `KAGGLE_CONFIG_DIR` to `<project root>\.kaggle` automatically
on activation.

1. Go to your Kaggle account → Settings → API → **Create New Token**.
2. Save the token value into `.kaggle/access_token` at the project root (create the folder if needed):
   ```
   echo YOUR_TOKEN > .kaggle/access_token
   ```
3. Never commit this file — `.kaggle/` is already in `.gitignore`.

(The classic `kaggle.json` with `KAGGLE_USERNAME`/`KAGGLE_KEY` still works too, if you have one from an
older token.)

## Inference on a new patient

Once you have trained checkpoints (`outputs/checkpoints/fold{0-4}_best.pt`), you can run the model
on any patient folder that contains T1, T2, and FLAIR NIfTI files:

```
# Ensemble of all 5 folds (recommended — soft probability average before thresholding)
python src/predict.py \
  --patient-dir data/raw/patient_001 \
  --config configs/baseline.yaml \
  --ensemble \
  --out-dir outputs/predictions/patient_001/

# Single checkpoint
python src/predict.py \
  --patient-dir data/raw/patient_001 \
  --checkpoint outputs/checkpoints/fold0_best.pt

# Skip N4 bias correction for a quick test
python src/predict.py --patient-dir data/raw/patient_001 --ensemble --skip-bias-correction
```

Outputs written to `--out-dir`:
- `pred_mask.nii.gz` — 3D binary lesion mask in the FLAIR-resampled space
- `overlay.png` — grid of lesion-positive slices with prediction overlay in red
  (3-column with ground-truth overlay if a mask file is found, 2-column otherwise)
- `summary.json` — slice counts, total lesion voxels, Dice (only if ground-truth mask present)

The script applies the same N4 bias correction and FLAIR-space resampling as the training
preprocessing. If no ground-truth mask is found in the patient folder, it runs in
prediction-only mode without computing Dice.

## Notebooks

- `01_eda.ipynb` — inspect volumes, visualize modalities + lesion masks, lesion burden distribution.
- `02_preprocessing_dev.ipynb` — debug preprocessing on 2-3 patients before running it on all 60.
- `03_results_report.ipynb` — training curves, metrics table, prediction overlays, **critical
  analysis of results, and prioritized future-improvement notes** (start here for the full story
  behind the headline numbers).

## Method notes

- **Task**: 2D axial-slice binary segmentation (lesion vs. background) with a from-scratch U-Net.
- **Modalities**: Each MRI scan comes in three "flavors" that highlight different tissue properties —
  T1 shows general brain anatomy, T2 is sensitive to water and inflammation, and FLAIR suppresses
  cerebrospinal fluid so lesions stand out more clearly. The model uses all three simultaneously as
  input channels. T1/T2/FLAIR are *not* co-registered in this dataset (each has its own native
  resolution/slice count per patient) — `preprocessing.py` N4-corrects each modality in its own
  space, then registers T1/T2 onto the FLAIR grid via SimpleITK rigid registration (Euler3D
  transform, Mattes mutual information metric, multi-resolution pyramid 4→2→1). Use
  `--skip-registration` for fast smoke tests (falls back to proportional zoom).
- **Class balance**: lesion pixels are a small minority even within lesion-containing slices;
  `preprocessing.py` caps lesion-free slices per patient to reduce slice-level imbalance, and
  training uses a combined Dice + BCE loss.
- **Split**: patient-level k-fold (default 5-fold) — slices from the same patient never span
  train/val within a fold, enforced by an assertion in `src/data/dataset.py`.
- **Metrics**: Dice, IoU, sensitivity, precision, reported per fold and as mean ± std across folds.

## Limitations & future work

Full analysis in [`notebooks/03_results_report.ipynb`](notebooks/03_results_report.ipynb):

- n=60 patients from a single public dataset — no external validation on other institutions' data.
- 2D per-slice segmentation ignores inter-slice context; lesions spanning only 1–2 slices are harder to detect consistently.
- Smallest lesions (~4×5 pixels at 256×256) are near the detection limit of any standard conv model.
- All metrics are from internal cross-validation; real-world Dice may differ.
- No held-out test set outside the 5-fold CV.

## Dataset & license

Trained on [`orvile/multiple-sclerosis-brain-mri-lesion-segmentation`](https://www.kaggle.com/datasets/orvile/multiple-sclerosis-brain-mri-lesion-segmentation)
(Kaggle), 60 patients, T1/T2/FLAIR + consensus lesion masks from ~20 centers, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset itself is not redistributed
in this repo (`data/` is gitignored) — download it yourself via `src/data/download.py`.

Code in this repository is licensed under the [MIT License](LICENSE).
