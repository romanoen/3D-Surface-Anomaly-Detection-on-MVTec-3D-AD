# Pipeline

This project uses one shared pipeline and plugs two anomaly detectors into it.

## Shared Flow

1. Load one raw `xyz/*.tiff` sample per category and save a depth-map preview.
2. Index the selected MVTec 3D-AD categories.
3. Load the raw 3D data and extract a depth or height-map representation.
4. Preprocess the maps and save them under `data/processed/`.
5. Create `train.csv`, `val.csv`, and `test.csv` under `data/splits/`.
6. Extract overlapping patches from processed maps.
7. Train the classical baseline on normal training patches.
8. Train the autoencoder on the same normal training patches.
9. Run both methods on the test split.
10. Aggregate patch scores back to image space.
11. Compute image-level metrics and save visual comparisons.

## Figure Outputs

Every stage should save figures:

- `fig/01_dataset/`: category examples, one sample depth-map preview per class, raw overview
- `fig/02_preprocessing/`: normalization, resizing, valid-mask checks
- `fig/03_patches/`: patch grids, overlap examples, aggregation sanity checks
- `fig/04_classical/`: feature plots, score maps, score distributions
- `fig/05_autoencoder/`: loss curves, reconstructions, residual maps
- `fig/06_results/`: side-by-side comparisons, ROC curves, summary panels

## Entry Points

- `scripts/visualize_depth_samples.py`
- `scripts/prepare_data.py`
- `scripts/train_classical.py`
- `scripts/train_autoencoder.py`
- `scripts/run_benchmark.py`
