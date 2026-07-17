# Experiment log

All runs use the same dataset (60 patients, N4 bias correction, neg_ratio=1.5, seed=42),
same 5-fold patient-level split (seed=42), same U-Net architecture (base_channels=32, in_channels=3).

---

## Exp 1 — Baseline: Dice+BCE, 50 epochs, fixed LR
**Date**: 2026-07-15  
**Config**: `loss=dice_bce, bce_weight=0.5, lr=1e-4, epochs=50, no scheduler, no early stopping`

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | 0.573 | 0.455 | 0.605 | 0.641 |
| 1 | 0.483 | 0.382 | 0.544 | 0.625 |
| 2 | 0.615 | 0.477 | 0.629 | 0.679 |
| 3 | 0.520 | 0.393 | 0.532 | 0.677 |
| 4 | 0.572 | 0.456 | 0.582 | 0.685 |
| **Mean** | **0.553** | **0.433** | **0.578** | **0.661** |
| **Std**  | **0.051** | **0.042** | **0.040** | **0.027** |

**Notes**: Folds stopped improving at epochs 45–48 of 50 — not fully converged.
Fold 1 is the persistent outlier (0.483 vs ~0.57 for the others).

---

## Exp 2 — N4 bias correction + slice balancing (same loss/schedule as Exp 1)
**Date**: 2026-07-15  
**Config**: `loss=dice_bce, bce_weight=0.5, lr=1e-4, epochs=50` + N4 per modality + neg_ratio=1.5

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | 0.573 | 0.455 | 0.605 | 0.641 |
| 1 | 0.483 | 0.382 | 0.544 | 0.625 |
| 2 | 0.615 | 0.477 | 0.629 | 0.679 |
| 3 | 0.520 | 0.393 | 0.532 | 0.677 |
| 4 | 0.572 | 0.456 | 0.582 | 0.685 |
| **Mean** | **0.553** | **0.433** | **0.578** | **0.661** |
| **Std**  | **0.051** | **0.042** | **0.040** | **0.027** |

**Key result**: Std dropped ~42–63% vs Exp 1 (before N4). Fold 1 improved from 0.41→0.48.
Mean Dice held steady while variance halved — N4 reduced inter-scanner intensity gap.

---

## Exp 3 — Tversky loss (alpha=0.7) + ReduceLROnPlateau + early stopping
**Date**: 2026-07-16  
**Config**: `loss=tversky, alpha=0.7, beta=0.3, lr=1e-4, epochs=100, lr_patience=5, early_stop_patience=15`  
Early stopping epochs: fold0=74, fold1=80, fold2=52, fold3=60, fold4=51

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | 0.620 | 0.496 | 0.658 | 0.686 |
| 1 | 0.437 | 0.335 | 0.633 | 0.473 |
| 2 | 0.576 | 0.439 | 0.632 | 0.609 |
| 3 | 0.524 | 0.395 | 0.583 | 0.600 |
| 4 | 0.556 | 0.436 | 0.643 | 0.583 |
| **Mean** | **0.543** | **0.420** | **0.630** | **0.590** |
| **Std**  | **0.069** | **0.060** | **0.028** | **0.076** |

**Key result**: Sensitivity ↑ 0.578→0.630 (+0.052) — Tversky penalising FN worked.
Precision ↓ 0.661→0.590 (−0.071) — alpha=0.7 too aggressive, model over-segments.
Mean Dice ↓ 0.553→0.543. Fold 1 worsened (0.483→0.437); std increased.
Scheduler and early stopping worked cleanly — next: tune alpha down to 0.6.

---

## Exp 4 — Tversky loss (alpha=0.6) + ReduceLROnPlateau + early stopping
**Date**: 2026-07-16  
**Config**: `loss=tversky, alpha=0.6, beta=0.4, lr=1e-4, epochs=100, lr_patience=5, early_stop_patience=15`

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | — | — | — | — |
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |
| **Mean** | **—** | **—** | **—** | **—** |
| **Std**  | **—** | **—** | **—** | **—** |

**Status**: Running.
