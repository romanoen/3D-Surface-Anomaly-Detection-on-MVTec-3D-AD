# MVTec 3D-AD Surface Anomaly Detection

This repository implements a reproducible anomaly detection pipeline for the
MVTec 3D-AD dataset. The project studies industrial surface inspection in the
unsupervised setting: models are trained only on defect-free samples and must
detect anomalous test samples, ideally with spatial heatmaps that indicate
where the defect appears.

The central comparison is between:

1. A classical category-specific baseline based on raw normalized depth
   patches, PCA, and One-Class SVM.
2. A category-specific compact convolutional autoencoder trained on the same
   patch representation.
3. A planned RGB/depth modality ablation motivated by the interim results.

The current repository already contains the shared data pipeline, the classical
training and inference path, an implemented depth-only autoencoder path, saved
metrics, qualitative heatmaps, and project documentation for the interim
report.

## Current Status

Implemented:

- Dataset indexing for the locally available MVTec 3D-AD categories.
- Depth preprocessing from raw XYZ files.
- Foreground estimation, object cropping, aspect-preserving resize, and
  patch-grid snapping.
- Overlapping patch extraction with `32 x 32` patches and `16 x 16` stride.
- Category-specific classical baseline:
  `raw depth patch -> StandardScaler -> PCA(64) -> One-Class SVM`.
- Classical inference with image scores and overlap-averaged anomaly heatmaps.
- Category-specific depth-only convolutional autoencoder training on the same
  normalized patch representation with masked reconstruction loss and saved
  checkpoints.
- Autoencoder inference with residual heatmaps and image-level anomaly scores.
- Shared benchmark outputs with per-image, per-category, and summary metrics.
- Diagnostic analysis figures for ROC/PR behavior, category deltas, and
  representative error cases.
- Ground-truth contour visualization in processed-image coordinates.
- Unit tests for the main data, patching, feature, model, training,
  inference, benchmark, and visualization components.

Planned next:

- Add processed RGB maps using the same crop and resize geometry as depth.
- Run controlled modality ablations: depth, RGB, and depth+RGB.
- Build on the shared benchmark outputs to explain failure modes and motivate
  the RGB or multimodal extension.

## Current Benchmark Results

The latest full benchmark compares the two completed depth-only baselines on
the same shared test split. The run used:

- 10 object categories.
- 1197 test images.
- 238113 scored test patches for each method.

Image-level test performance:

| Method | Overall AUROC | Overall AP | Macro AUROC | Macro AP |
| --- | ---: | ---: | ---: | ---: |
| Classical | 0.5517 | 0.8293 | 0.5415 | 0.8307 |
| Autoencoder | 0.5837 | 0.8227 | 0.5812 | 0.8402 |

Per-category AUROC deltas show where the autoencoder helps most and where the
classical baseline still holds an advantage:

| Autoencoder Better | Delta AUROC | Classical Better | Delta AUROC |
| --- | ---: | --- | ---: |
| bagel | +0.1999 | peach | -0.1165 |
| foam | +0.1600 | cable_gland | -0.0772 |
| cookie | +0.1221 | tire | -0.0211 |
| rope | +0.0539 |  |  |
| dowel | +0.0514 |  |  |

Interpretation: both methods capture non-random anomaly signal, but neither is
reliably strong across all categories. The classical baseline often reacts to
global object geometry, boundary regions, or patch artifacts; the autoencoder
usually produces cleaner and more localized heatmaps but still misses many
defects or responds too weakly. These depth-only baselines are useful as
references and for failure analysis, but they do not yet provide dependable
anomaly detection on the full benchmark.

## Dataset

Place the MVTec 3D-AD dataset under:

```text
data/raw/
```

Expected category folders:

```text
data/raw/
+-- bagel/
+-- cable_gland/
+-- carrot/
+-- cookie/
+-- dowel/
+-- foam/
+-- peach/
+-- potato/
+-- rope/
+-- tire/
```

The indexer expects the official MVTec 3D-AD layout with `train`,
`validation`, and `test` folders, including `rgb/`, `xyz/`, and available
`gt/` masks. If the dataset lives elsewhere, update `paths.dataset_root` in
`configs/base.yaml`.

The generated split metadata lives under `data/splits/`:

- `index.csv`
- `index_summary.csv`
- `train.csv`
- `val.csv`
- `test.csv`

The current split convention is:

- `train`: normal samples only
- `val`: normal samples only
- `test`: normal and anomalous samples

## Environment

Create and activate the conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate mvtec-3d-ad
pip install -e .
```

`environment.yml` is the source of truth for the project environment.
`requirements.txt` is included only as a secondary reference for pip-based
setups.

## Reproducible Run Order

All commands below assume the environment is active and the dataset is under
`data/raw/`.

1. Optional: save one raw depth preview per category.

```bash
python scripts/visualize_depth_samples.py
```

2. Prepare split files, processed metadata, and documentation figures.

```bash
python scripts/prepare_data.py
```

For a quick smoke run:

```bash
python scripts/prepare_data.py --limit 20
```

3. Train the classical One-Class SVM baseline.

```bash
python scripts/train_classical.py
```

Useful smaller run:

```bash
python scripts/train_classical.py --max-train-patches 2000 --max-val-patches 500
```

4. Run classical inference on the test split.

```bash
python scripts/infer_classical.py --split test
```

Optional quick inference:

```bash
python scripts/infer_classical.py --split test --max-images 20
```

5. Train the depth-only autoencoder.

```bash
python scripts/train_autoencoder.py
```

Useful smaller run:

```bash
python scripts/train_autoencoder.py --epochs 5 --max-train-patches 2000 --max-val-patches 500
```

6. Run the shared benchmark on the test split.

```bash
python scripts/run_benchmark.py
```

7. Generate diagnostic analysis figures from the saved benchmark outputs.

```bash
python scripts/generate_analysis_figures.py
```

8. Run unit tests.

```bash
python -m unittest discover tests
```

The remaining work now centers on RGB or multimodal extensions and the final
report analysis built on top of the saved benchmark outputs.

## Tests And CI

The `tests/` suite is designed to protect the pipeline logic without requiring
the full MVTec 3D-AD dataset. The tests use small synthetic fixtures and
temporary directories to verify:

- dataset indexing and split handling
- preprocessing and mask transforms
- patch extraction and feature construction
- OCSVM and autoencoder training/inference behavior
- benchmark table generation and metric calculations
- qualitative figure and selection helpers

Run the full test suite locally with:

```bash
python -m unittest discover tests
```

The repository also includes a GitHub Actions workflow at
`.github/workflows/tests.yml` that creates the conda environment, installs the
package in editable mode, and runs the same unit tests automatically on pushes
and pull requests. The CI intentionally does not run the full dataset
preparation, training, or benchmark steps, because those require the external
dataset and are much heavier than the logic checks covered by the tests.

## Repository Structure

```text
configs/      Shared configuration files
data/         Raw data, processed arrays, and split metadata
docs/         Reports, project plan, data format notes, and decisions
fig/          Visual documentation generated by pipeline stages
outputs/      Models, metrics, heatmaps, and logs
scripts/      Runnable entry points for pipeline stages
src/          Reusable implementation modules
tests/        Unit tests for core pipeline behavior
```

Important modules:

```text
src/data/          Dataset indexing, preprocessing, patch extraction, loaders
src/features/      Raw depth and geometric patch feature construction
src/models/        One-Class SVM, Isolation Forest stub, compact autoencoder
src/training/      Classical and autoencoder training utilities
src/inference/     Classical inference, autoencoder inference, anomaly-map generation
src/evaluation/    Metrics and visualization helpers
src/utils/         Config, IO, logging, and reproducibility helpers
```

## Outputs and Figures

The project saves visual and numeric evidence at each major stage.

Figures:

- `fig/01_dataset/`: raw depth examples and dataset index summary
- `fig/02_preprocessing/`: raw vs. processed depth previews
- `fig/03_patches/`: patch grid, traversal, and coverage checks
- `fig/04_classical/`: classical anomaly heatmaps
- `fig/05_autoencoder/`: training curves, reconstruction examples, and residual heatmaps
- `fig/06_results/`: shared comparison panels and per-category metric figures
- `fig/06_results/analysis/`: ROC/PR curves, score-rank histograms, per-category deltas, and error galleries

The comparison figures now use a clearer layout:

- `depth + GT` for the original processed depth map with the defect annotation
- `classical heatmap` as a standalone method output
- `autoencoder heatmap` as a standalone method output

This avoids confusing depth texture or noise with the model response itself.

Classical model artifacts:

```text
outputs/models/classical/<category>/
+-- ocsvm.joblib
+-- scaler.joblib
+-- pca.joblib
+-- features.json
```

Classical metrics and logs:

```text
outputs/logs/classical_training_summary.json
outputs/metrics/classical_per_category.csv
outputs/metrics/classical_image_scores_test.csv
outputs/metrics/classical_patch_scores_test.csv
outputs/metrics/classical_heatmaps/
```

Autoencoder model artifacts:

```text
outputs/models/autoencoder/
+-- <category>/
    +-- conv_autoencoder.pt
```

Autoencoder metrics and logs:

```text
outputs/logs/autoencoder_training_summary.json
outputs/metrics/autoencoder_per_category.csv
outputs/metrics/autoencoder_histories/
outputs/metrics/autoencoder_image_scores_test.csv
outputs/metrics/autoencoder_patch_scores_test.csv
outputs/metrics/autoencoder_heatmaps/
```

Shared benchmark outputs:

```text
outputs/metrics/per_image.csv
outputs/metrics/per_category.csv
outputs/metrics/summary.csv
fig/06_results/
```

## Documentation

Core project documentation:

- `docs/implementation_plan.md`: implementation-first project plan
- `docs/pipeline.md`: end-to-end execution flow
- `docs/data_format.md`: raw, processed, split, and output data conventions
- `docs/decisions.md`: implementation decisions and rationale
- `docs/interim_report.md`: working notes for the interim report
- `docs/Project_3D_Informe_nr1/`: LaTeX interim report for submission

The report and README are aligned around the same current project story: the
shared patch pipeline and both depth-only baselines are implemented, the saved
benchmark and analysis figures make their limitations visible, and the next
phase is the RGB or multimodal extension under the same evaluation framework.
