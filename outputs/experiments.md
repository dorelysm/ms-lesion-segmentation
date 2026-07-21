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

## Exp 5 — Registro anatómico real (SimpleITK rigid) + Tversky α=0.6
**Date**: 2026-07-17  
**Motivación**: El zoom proporcional (`scipy.ndimage.zoom`) no alinea anatómicamente las modalidades — un vóxel en [x,y,z] de FLAIR no corresponde al mismo tejido en T1/T2 tras el rescalado. El registro rígido con Mattes MI debería corregir este desalineamiento, reduciendo ruido de canal en todas las métricas.  
**Cambio**: Reemplazar `nd_zoom` en `load_patient_volumes` por `_register_sitk` (SimpleITK Euler3D + Mattes MI, pirámide dinámica, NONE sampling, fallback geometry-only). Sin cambios en loss, arquitectura ni split. Commits: `e720b47`, `e315a2e`, `ff6b56e`.  
**Config**: igual que Exp 4 (Tversky α=0.6, early stopping patience=15, ReduceLROnPlateau patience=5).  
Early stopping epochs: fold0=48, fold1=31, fold2=53, fold3=37, fold4=33

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | 0.1432 | 0.1295 | 0.4304 | 0.4158 |
| 1 | 0.1479 | 0.1356 | 0.3681 | 0.3785 |
| 2 | 0.1613 | 0.1452 | 0.2675 | 0.3049 |
| 3 | 0.0931 | 0.0780 | 0.2768 | 0.3842 |
| 4 | 0.1137 | 0.0911 | 0.4573 | 0.1250 |
| **Mean** | **0.132** | **0.116** | **0.360** | **0.322** |
| **Std**  | **0.028** | **0.029** | **0.087** | **0.117** |

**Key result**: Regresión severa — Dice 0.132 vs 0.551 en Exp 4 (−0.419). El registro, tal como está implementado, empeora drásticamente el rendimiento. 

**Diagnóstico probable**:
1. **Geometría NIfTI uniforme**: Todos los volúmenes de este dataset tienen `origin=(0,0,0)`, `spacing=(1,1,1,1)` y `direction=identity` en los headers NIfTI — es decir, no hay metadatos de posición real en el espacio del paciente. El inicializador por geometría centra los volúmenes en el mismo punto, y la optimización de MI sobre volúmenes con headers idénticos no tiene información posicional real que explotar. El registro termina produciendo alineaciones aleatorias o subóptimas (muchos folds hacen fallback geometry-only porque el optimizer diverge con pyrámide shrink=[1]).
2. **FOV asimétrico**: T1 (512×512mm) se resamplea al grid de FLAIR (256×256mm), recortando la mitad del campo de visión de T1. Con zoom proporcional, toda la imagen escalaba suavemente; con registro+resample, se pierde información en el borde.
3. **Pérdida de 2 pacientes**: Patients 5 y 11 producen 0 slices tras el registro (posiblemente por resampling fuera del FOV), dejando 58 pacientes (vs 60 en Exp 4).

**Conclusión**: El registro anatómico real solo aporta valor cuando los headers NIfTI contienen coordenadas espaciales reales (posición del paciente en el escáner). En este dataset, los headers son sintéticos/vacíos — el zoom proporcional sigue siendo la mejor estrategia de resampling para este caso de uso.

---

## Exp 6 — Intensity augmentation (gamma jitter + brightness shift) + Tversky α=0.6
**Date**: 2026-07-18  
**Motivación**: El diagnóstico de fold 1 (Dice persistente ≈0.48) identificó que el val set del fold 1 contiene 2 pacientes con lesiones microscópicas (mediana ~17–19 vóxeles a 256×256). La hipótesis era que añadir jitter de intensidad —gamma U[0.7, 1.5] y brightness shift U[−0.2, 0.2]— forzaría al modelo a detectar la señal de lesión independientemente del nivel de contraste absoluto, mejorando la generalización entre scanners.  
**Cambio**: Añadir dos bloques al final de `_augment` en `src/data/dataset.py`: `sign(x)*|x|^gamma` para manejar valores negativos (z-score), y shift aditivo por imagen. Sin cambios en arquitectura, loss, split, ni preprocessing.  
**Config**: `configs/exp6_intensity_aug.yaml` — igual que Exp 4 excepto `epochs=200`, `early_stop_patience=30` (extended para descartar problema de convergencia lenta).  
Early stopping epochs: fold0=42, fold1=109, fold2=95, fold3=51, fold4=64

| Fold | Dice | IoU | Sensitivity | Precision |
|------|------|-----|-------------|-----------|
| 0 | 0.2171 | 0.2040 | 0.3961 | 0.5294 |
| 1 | 0.2072 | 0.1934 | 0.3091 | 0.5515 |
| 2 | 0.1470 | 0.1319 | 0.2577 | 0.3358 |
| 3 | 0.1317 | 0.1077 | 0.4133 | 0.3252 |
| 4 | 0.1703 | 0.1612 | 0.3677 | 0.2534 |
| **Mean** | **0.175** | **0.160** | **0.349** | **0.399** |
| **Std**  | **0.037** | **0.041** | **0.064** | **0.133** |

**Key result**: Regresión severa — Dice 0.175 vs 0.551 en Exp 4 (−0.376). El intensity augmentation empeoró drásticamente el rendimiento.

**Diagnóstico**:
1. **Val dice erráticodesde el inicio**: el val dice oscila sin patrón (ej. fold 4: 0.006 → 0.079 → 0.006 → 0.110 → 0.164) en lugar de converger suavemente. El pico se alcanza en épocas tempranas (12–34) y el modelo no mejora después, señal de que aprendió a sobreajustarse a los patrones augmentados sin transferir a las imágenes de validación limpias.
2. **Distribución de entrenamiento vs. validación divergen**: los rangos gamma U[0.7, 1.5] y brightness U[−0.2, 0.2] con p=0.5 por cada transform son suficientemente agresivos para crear una distribución de entrenamiento marcadamente distinta a la de validación (sin augmentation). El modelo optimiza para imágenes con artefactos de intensidad en lugar de aprender features robustos.
3. **Más épocas no ayudan**: extender de epochs=100/patience=15 a epochs=200/patience=30 no mejoró los resultados — los folds pararon igualmente en 42–109 épocas sin mejora sustancial.

**Conclusión**: El intensity augmentation con estos rangos es contraproducente en este dataset. El augmentation geométrico (Exp 4) es suficiente. Si se quiere explorar augmentation de intensidad en el futuro, empezar con rangos mucho más conservadores (gamma U[0.9, 1.1], brightness ±0.05, p≤0.3).

---

## Summary across experiments

| Exp | Loss | alpha | Augmentation | Mean Dice | Std Dice | Mean Sensitivity | Mean Precision |
|-----|------|-------|-------------|-----------|----------|------------------|----------------|
| 1 | Dice+BCE | — | geométrico | 0.553 | 0.051 | 0.578 | 0.661 |
| 2 | Dice+BCE + N4 | — | geométrico | 0.553 | 0.051 | 0.578 | 0.661 |
| 3 | Tversky | 0.7 | geométrico | 0.543 | 0.069 | 0.630 | 0.590 |
| **4** | **Tversky** | **0.6** | **geométrico** | **0.551** | **0.056** | **0.626** | **0.605** |
| 5 | Tversky | 0.6 | geométrico (sitk rigid) | 0.132 | 0.028 | 0.360 | 0.322 |
| 6 | Tversky | 0.6 | geométrico + gamma + brightness | 0.175 | 0.037 | 0.349 | 0.399 |

**Takeaway**: Exp 4 sigue siendo el mejor resultado. Exp 5 (registro rígido) y Exp 6 (intensity augmentation) producen regresiones severas. El registro falla porque los headers NIfTI del dataset son sintéticos. El intensity augmentation falla porque los rangos gamma U[0.7,1.5] y brightness ±0.2 crean una distribución de entrenamiento demasiado distinta a la de validación. Próximos levers potenciales: arquitectura 2.5D (U-Net con contexto inter-slice), augmentation de intensidad con rangos conservadores (gamma U[0.9,1.1], brightness ±0.05), o cambiar la estrategia de split para distribuir mejor los pacientes difíciles.

---

## Future Experiments

Los experimentos siguientes no se han ejecutado. Se documentan aquí para que un investigador pueda
retomar el trabajo a partir del estado actual (Exp 4, Dice 0.563 ± 0.047 tras el fix de eje de
preprocesamiento). El punto de partida en todos los casos es `configs/baseline.yaml` como referencia.

---

### Exp 7 — Split estratificado por lesion burden

**Motivación**: El fold 1 tiene Dice persistentemente menor (~0.496 vs ~0.57 en otros folds). La
investigación mostró que este fold contiene 2 pacientes con lesiones microscópicas (pacientes 20 y
53, mediana ~17–19 vóxeles a 256×256 — aproximadamente un blob de 4×5 píxeles). Estos pacientes
cayeron en el mismo fold de validación por azar, no por diseño. Con `StratifiedKFold` sobre bins de
lesion burden, se distribuirían uniformemente entre folds, haciendo que la std inter-fold refleje
la variabilidad real del modelo en lugar de la variabilidad del split.

**Cambios**:
- `src/data/dataset.py`: reemplazar `KFold` por `StratifiedKFold` en `patient_kfold_splits`.
  Calcular un label por paciente: `lesion_ratio = n_lesion_slices / total_slices`; discretizar en
  3–4 bins con `pd.qcut`; pasar como `y` a `StratifiedKFold.split`.
- `configs/exp7_stratified_split.yaml`: copia de `baseline.yaml` (sin cambios de hiperparámetros).
- `outputs/experiments.md`: registrar resultados tras el run.

**Señal de éxito**: std inter-fold baja respecto a Exp 4 (actualmente 0.047); el Dice de fold 1
sube hacia el rango de los demás folds (≥0.54). El mean Dice no debería bajar.

**Nota**: este cambio no mejora la capacidad del modelo — mejora la validez de la estimación de
rendimiento. Conviene correrlo antes de Exp 8 y 9 para tener un baseline más limpio.

**Esfuerzo estimado**: ~1 hora (cambio de 10–15 líneas + run completo de entrenamiento).

---

### Exp 8 — Intensity augmentation conservador

**Motivación**: Exp 6 demostró que gamma U[0.7, 1.5] + brightness ±0.2 con p=0.5 crea distribution
shift severo entre train y val. El objetivo de intensity augmentation sigue siendo válido: el dataset
proviene de ~20 centros con escáneres distintos, y el modelo debería ser robusto a diferencias de
contraste inter-escáner. La hipótesis es que rangos mucho más conservadores evitan el distribution
shift mientras aún aportan regularización.

**Cambios**:
- `src/data/dataset.py` — modificar `_augment` con los rangos conservadores (revertir Exp 6 si
  `dataset.py` aún contiene los rangos agresivos):
  ```python
  # gamma jitter conservador
  if np.random.rand() < 0.3:
      gamma = np.random.uniform(0.9, 1.1)
      image = np.sign(image) * (np.abs(image) ** gamma)
  # brightness jitter conservador
  if np.random.rand() < 0.3:
      delta = np.random.uniform(-0.05, 0.05)
      image = image + delta
  ```
- `configs/exp8_conservative_aug.yaml`: copia de `baseline.yaml` con comentarios doc sobre los rangos.
- `outputs/experiments.md`: registrar resultados.

**Señal de éxito**: val dice converge suavemente desde epoch 1 (sin el comportamiento errático de
Exp 6); mean Dice ≥ 0.563 (Exp 4 actual); std inter-fold baja. Si val dice oscila desde epoch 1,
reducir más los rangos (gamma ±0.05, brightness ±0.02) antes de concluir que el concepto no funciona.

**Nota**: verificar primero que `dataset.py` está en el estado de Exp 4 (solo augmentation geométrico)
antes de aplicar este cambio. `git diff HEAD src/data/dataset.py` para confirmarlo.

**Esfuerzo estimado**: ~2 horas (cambio pequeño + run completo).

---

### Exp 9 — Arquitectura 2.5D (slices adyacentes como canales extra)

**Motivación**: El modelo actual ve cada slice axial de forma independiente. Una lesión que ocupa
2–3 slices consecutivos se ve como manchas sin conexión entre ellas — la red no puede usar la
continuidad espacial inter-slice para confirmar o rechazar una detección. Incluir los slices ±1
como canales extra (de 3 a 9 canales de entrada: T1/T2/FLAIR × 3 slices) da contexto anatómico
sin el coste de una U-Net 3D completa. Es el cambio con mayor potencial de mejora en lesiones
pequeñas (1–2 slices de grosor).

**Cambios**:
- `src/data/preprocessing.py` — sin cambios en el preprocessing. Los slices se apilan en el Dataset.
- `src/data/dataset.py` — modificar `__getitem__` para cargar el slice central ± n_context slices
  y apilarlos como canales:
  ```python
  # en lugar de image = np.load(row["image_path"])  # (3, H, W)
  # cargar slice central + vecinos y concatenar en el eje de canal → (9, H, W) con context=1
  ```
  Manejar los bordes (primer y último slice del volumen) con padding por reflexión o replicación
  del slice más cercano.
- `src/models/unet.py` — solo cambiar `in_channels` en `UNet2D.__init__`: el resto de la
  arquitectura es agnóstico al número de canales de entrada. Con context=1: `in_channels=9`;
  con context=2: `in_channels=15`.
- `configs/exp9_2d5_context1.yaml`: `model.in_channels: 9`, `model.context_slices: 1`.
- `outputs/experiments.md`: registrar resultados para context=1. Si mejora, probar context=2.

**Señal de éxito**: Dice en fold 1 sube hacia ≥0.53 (las lesiones microscópicas que solo aparecen
en 1–2 slices son el caso donde más debería ayudar el contexto inter-slice); mean Dice ≥ 0.58.

**Precaución**: asegurarse de que `patient_kfold_splits` sigue siendo patient-level — los slices
de un mismo paciente deben mantenerse todos en train o todos en val, incluyendo los slices vecinos
que se usan como contexto. Esto ya está garantizado por la lógica actual del split, pero verificarlo
con el assertion existente en `dataset.py`.

**Esfuerzo estimado**: ~4–6 horas (cambio en Dataset + UNet + run completo). Es el experimento de
mayor complejidad y mayor potencial de los tres.
