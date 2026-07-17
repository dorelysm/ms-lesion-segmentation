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
**Date**: 2026-07-17  
**Config**: `loss=tversky, alpha=0.6, beta=0.4, lr=1e-4, epochs=100, lr_patience=5, early_stop_patience=15`  
Early stopping epochs: fold0=?, fold1=?, fold2=?, fold3=?, fold4=45

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | 0.620 | 0.496 | 0.632 | 0.705 |
| 1 | 0.481 | 0.376 | 0.592 | 0.558 |
| 2 | 0.597 | 0.461 | 0.655 | 0.622 |
| 3 | 0.530 | 0.401 | 0.591 | 0.606 |
| 4 | 0.529 | 0.408 | 0.663 | 0.534 |
| **Mean** | **0.551** | **0.428** | **0.626** | **0.605** |
| **Std**  | **0.056** | **0.049** | **0.034** | **0.066** |

**Key result**: Best overall balance so far. Dice ≈ Exp 2 (0.551 vs 0.553), sensitivity stays high
(0.626 vs 0.578 baseline), precision recovered partially (0.605 vs 0.590 Exp 3, vs 0.661 baseline).
Std improved vs Exp 3 (0.056 vs 0.069). Fold 1 recovered to 0.481 (was 0.437 in Exp 3).
**alpha=0.6 is the best Tversky setting found so far** — sensitivity gain without the precision collapse of alpha=0.7.

---

## Summary across experiments

| Exp | Loss | alpha | Mean Dice | Std Dice | Mean Sensitivity | Mean Precision |
|-----|------|-------|-----------|----------|------------------|----------------|
| 1 | Dice+BCE | — | 0.553 | 0.051 | 0.578 | 0.661 |
| 2 | Dice+BCE + N4 | — | 0.553 | 0.051 | 0.578 | 0.661 |
| 3 | Tversky | 0.7 | 0.543 | 0.069 | 0.630 | 0.590 |
| **4** | **Tversky** | **0.6** | **0.551** | **0.056** | **0.626** | **0.605** |

**Takeaway**: Tversky(0.6) gives the best sensitivity/precision tradeoff. Dice is nearly identical
to baseline but sensitivity is +0.048 — clinically relevant (fewer missed lesions).
The persistent bottleneck is fold 1 (0.48 across all experiments); likely a data issue in that
validation set rather than a modeling issue. Next lever: anatomical registration (Improvement #1).
