# MVTec 3D-AD Surface Anomaly Detection

This repository implements a reproducible anomaly detection pipeline for the
MVTec 3D-AD dataset. The project studies industrial surface inspection in the
unsupervised setting: models are trained only on defect-free samples and must
detect anomalous test samples, ideally with spatial heatmaps that indicate
where the defect appears.

The central comparison is between:

1. A classical category-specific baseline based on raw normalized depth
   patches, PCA, and One-Class SVM.
2. A planned compact convolutional autoencoder trained on the same patch
   representation.
3. A planned RGB/depth modality ablation motivated by the interim results.

The current repository already contains the shared data pipeline, the classical
training and inference path, saved metrics, qualitative heatmaps, and project
documentation for the interim report.

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
- Ground-truth contour visualization in processed-image coordinates.
- Unit tests for the main data, patching, feature, model, training, and
  inference components.

Planned next:

- Complete the convolutional autoencoder training and inference path.
- Add processed RGB maps using the same crop and resize geometry as depth.
- Run controlled modality ablations: depth, RGB, and depth+RGB.
- Compare classical and deep models under the same split, scoring, and
  visualization setup.

## Interim Results

The current classical baseline was trained per category on normal training
patches only. The latest run used:

- 10 object categories.
- 200000 normal training patches.
- 47913 validation patches.
- 1197 test images.
- 238113 scored test patches.
- 1024 raw depth values per patch, reduced to 64 PCA features.

Image-level test performance:

| Metric | Value |
| --- | ---: |
| Overall AUROC | 0.5517 |
| Overall AP | 0.8293 |
| Macro AUROC | 0.5415 |
| Macro AP | 0.8307 |

Per-category AUROC shows that the local depth-only baseline is useful for some
categories but weak for others:

| Category | AUROC | Category | AUROC |
| --- | ---: | --- | ---: |
| rope | 0.8071 | carrot | 0.5017 |
| cookie | 0.6969 | potato | 0.4773 |
| peach | 0.6705 | bagel | 0.4597 |
| cable_gland | 0.5145 | foam | 0.4350 |
| dowel | 0.4301 | tire | 0.4221 |

Interpretation: the first baseline is a useful sanity check and a fair
classical reference, but the results suggest that many MVTec 3D-AD defects are
not reliably separable with local depth-only patches. This motivates the
planned autoencoder and RGB/multimodal experiments.

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

5. Run unit tests.

```bash
python -m unittest discover tests
```

The autoencoder and final benchmark entry points are present but intentionally
not implemented yet:

```bash
python scripts/train_autoencoder.py
python scripts/run_benchmark.py
```

They are part of the remaining work for the final submission.

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
src/models/        One-Class SVM, Isolation Forest stub, autoencoder stub
src/training/      Classical training utilities
src/inference/     Classical inference and anomaly-map generation
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
- `fig/05_autoencoder/`: reserved for autoencoder outputs
- `fig/06_results/`: reserved for final comparisons

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

## Documentation

Core project documentation:

- `docs/implementation_plan.md`: implementation-first project plan
- `docs/pipeline.md`: end-to-end execution flow
- `docs/data_format.md`: raw, processed, split, and output data conventions
- `docs/decisions.md`: implementation decisions and rationale
- `docs/interim_report.md`: working notes for the interim report
- `docs/Project_3D_Informe_nr1/`: LaTeX interim report for submission

The report and README are aligned around the same current project story: the
shared patch pipeline is implemented, the classical depth-only baseline exposes
important representation and context limitations, and the final phase will test
autoencoder and RGB/multimodal extensions under the same framework.
