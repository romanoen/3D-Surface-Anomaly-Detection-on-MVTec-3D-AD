# Interim Report Notes

## Project Goal

The project investigates anomaly detection on MVTec 3D-AD using a shared patch-based pipeline. The goal is to compare a classical one-class baseline and a deep autoencoder under the same preprocessing, patch extraction, split, scoring, and visualization setup.

The current implementation focuses on depth or height maps first, while keeping the pipeline extensible for RGB and multimodal inputs.

## Current Pipeline

### Dataset Indexing

The dataset index is built from the official MVTec 3D-AD directory structure. Each sample row stores:

- object category,
- split,
- defect type,
- label,
- sample id,
- RGB path,
- XYZ path,
- ground-truth mask path when available.

The repository uses the dataset-provided `train`, `validation`, and `test` folders directly and normalizes the split names to `train`, `val`, and `test`.

### Depth Preprocessing

The current shared representation is a normalized depth map derived from the `Z` channel of the raw XYZ files.

Preprocessing currently performs:

- extraction of the depth channel,
- valid-depth masking,
- foreground/object estimation,
- object crop with configurable margin,
- aspect-preserving resize,
- snapping of processed dimensions to the configured patch grid,
- per-image depth normalization over object pixels only,
- saving of processed depth and foreground mask arrays.

The processed maps may be rectangular. This is important for elongated categories such as `rope`, because forcing every crop into a square image introduced misleading empty regions and distorted the patch grid.

### Patch Extraction

Both classical and future autoencoder pipelines use the same patch extraction logic:

- patch size: `32 x 32`,
- stride: `16 x 16`,
- deterministic row-major patch order,
- fixed patch sizes,
- overlap-aware aggregation back into image space.

The preprocessing step snaps image dimensions to the patch grid so patches fit cleanly into the processed crop.

### Ground-Truth Visualization

Ground-truth defect masks are transformed into processed-image coordinates using the same crop and resize geometry as the depth map. Heatmap figures draw these masks as cyan contours over:

- the processed depth map,
- the anomaly heatmap,
- the depth/heatmap overlay.

This makes qualitative interpretation much easier, especially when anomaly maps appear visually noisy.

## Classical Baseline

### Initial Feature Baseline

The first classical version used nine handcrafted patch descriptors:

- mean,
- standard deviation,
- min,
- max,
- range,
- gradient magnitude mean,
- gradient magnitude standard deviation,
- roughness proxy,
- Laplacian variance.

This was useful as an initial baseline, but qualitative results showed that the representation was too compressed. Important spatial depth structure inside a patch was lost. Small defects could look like normal sensor noise, and larger defects could require context across multiple patches.

### Current Raw-Depth Patch Baseline

The classical pipeline has therefore been changed to use the normalized patch height values directly.

The current classical feature path is:

```text
normalized 32x32 depth patch
-> mask-aware fill of invalid/outside-object pixels
-> flatten to 1024 raw height values
-> StandardScaler
-> PCA(64)
-> category-specific One-Class SVM
-> patch anomaly score
```

The invalid/outside-object fill strategy is `patch_mean`. This avoids introducing artificial zero-depth structure at object borders.

The model is trained per category. This follows the project plan: every object category gets its own normality model, scaler, and PCA transform.

### Training Artifacts

The current training run produced one model directory per category under:

```text
outputs/models/classical/<category>/
```

Each directory contains:

- `ocsvm.joblib`,
- `scaler.joblib`,
- `pca.joblib`,
- `features.json`.

The summary file is:

```text
outputs/logs/classical_training_summary.json
```

Current training summary:

- category-specific models: 10,
- training patches: 200000,
- validation patches: 47913,
- input feature count per patch: 1024,
- model feature count after PCA: 64.

### Classical Test Results

The classical model was evaluated on the test split:

- test images: 1197,
- scored patches: 238113.

Image-level results:

```text
overall AUROC: 0.5517
overall AP:    0.8293
macro AUROC:   0.5415
macro AP:      0.8307
```

Per-category AUROC:

```text
rope:         0.8071
cookie:       0.6969
peach:        0.6705
cable_gland:  0.5145
carrot:       0.5017
potato:       0.4773
bagel:        0.4597
foam:         0.4350
dowel:        0.4301
tire:         0.4221
```

The raw-depth patch representation is methodologically cleaner than the nine handcrafted descriptors and is directly comparable to the planned autoencoder input. However, the performance gain is limited. This supports the current interpretation that the main challenge is not only feature choice, but also the limited context of local patch-based anomaly detection.

## Current Interpretation

The patch-based approach is still a useful baseline because many defects do alter local geometry. However, several qualitative examples show a clear limitation:

- Some defects are visually difficult to distinguish from sensor noise even for a human observer.
- Some defects are only anomalous when their global shape, location, or relation to other regions is considered.
- Individual patches from a larger structural defect can still look locally plausible.

This gives a useful storyline for the report:

> The local patch baseline is a reasonable first step, but it reveals that some MVTec 3D-AD defects are not locally distinctive in depth alone. This motivates richer representations and more context-aware models.

## RGB Investigation

Because some defects are hard to identify in depth alone, RGB images were checked as a possible additional modality.

Technical check:

- RGB files are present for all 1197 test samples.
- RGB and ground-truth mask sizes match for all checked test samples.
- RGB can therefore be transformed with the same crop and resize geometry as the depth maps.

A quick signal analysis compared RGB values inside the ground-truth defect region with a local surrounding context ring. This is not a trained model, but it estimates whether RGB carries visible defect information.

Median RGB color-effect signal by category:

```text
cookie:       1.965
potato:       1.163
peach:        1.145
foam:         0.998
tire:         0.859
carrot:       0.791
rope:         0.758
bagel:        0.736
dowel:        0.728
cable_gland:  0.717
```

Strong RGB signal was especially visible for:

```text
cookie / contamination: 5.419
cookie / hole:          3.116
potato / contamination: 2.379
peach / contamination:  2.265
foam / color:           1.415
tire / contamination:   1.370
```

Weak RGB signal was found for some structural or subtle cases:

```text
tire / combined:     0.315
cable_gland / hole:  0.491
rope / open:         0.534
bagel / crack:       0.568
carrot / crack:      0.611
```

Conclusion:

RGB is likely useful as an additional modality, especially for contamination, color changes, holes, and some surface defects. It should not replace depth, because many defects remain geometric or contextual. The next step should therefore be a controlled modality ablation.

## Next Steps

### 1. Add Processed RGB Maps

The next implementation step should add RGB preprocessing using the same geometry as depth:

```text
raw RGB
-> same crop box as depth
-> same resize geometry as depth
-> save processed RGB map
```

Important: RGB should not be normalized with per-image min/max in the same way as depth, because that could remove color-based anomaly information. A safer first version is to convert RGB to float values in `[0, 1]` and optionally standardize features only during model training.

The split files should be extended with:

```text
processed_rgb_path
```

### 2. Run Classical Modality Ablations

The classical pipeline should support three feature modes:

```text
depth
rgb
depth_rgb
```

Suggested feature paths:

```text
depth:
  32x32 depth patch -> 1024 values -> PCA -> OCSVM

rgb:
  32x32 RGB patch -> 3072 values -> PCA -> OCSVM

depth_rgb:
  depth + RGB patch -> 4096 values -> PCA -> OCSVM
```

All three should use the same train/validation/test split and the same image-level scoring logic. This will show whether RGB actually improves the classical baseline.

### 3. Implement the Autoencoder

The autoencoder should use the same normalized patch system.

Initial depth-only autoencoder:

```text
input: 1 x 32 x 32 depth patch
output: reconstructed depth patch
score: reconstruction residual
```

Multimodal autoencoder extension:

```text
input: 4 x 32 x 32 patch
channels: depth, R, G, B
output: reconstructed multimodal patch
score: depth/RGB residual or weighted residual
```

This keeps the comparison fair:

- classical model: one-class boundary in feature/PCA space,
- autoencoder: reconstruction model over the same local patch representation.

### 4. Add More Context If Needed

If both classical and autoencoder approaches remain weak, the likely reason is missing context rather than only model capacity.

Possible context extensions:

- larger patches,
- multi-scale patches,
- image-position features,
- category-specific normal templates,
- full-image or region-level autoencoders.

## Current Commands

Data preparation:

```bash
conda run -n mvtec-3d-ad python scripts/prepare_data.py
```

Classical training:

```bash
conda run -n mvtec-3d-ad python scripts/train_classical.py
```

Classical inference:

```bash
conda run -n mvtec-3d-ad python scripts/infer_classical.py --split test
```

Tests:

```bash
conda run -n mvtec-3d-ad python -m unittest discover tests
```

## Current Verification

The current codebase passed:

```text
36 unit tests
git diff --check
```

The classical pipeline has been retrained and test inference has been regenerated after the raw-depth patch feature change.
