# Implementation Plan

## Goal

Implement one shared pipeline for MVTec 3D-AD and plug two anomaly detectors into it:

1. Classical pipeline: handcrafted patch features + one-class model.
2. Deep pipeline: convolutional autoencoder + reconstruction error.

Both methods must use the same:

- categories,
- preprocessing,
- patch extraction,
- train/validation split,
- image-level scoring logic,
- anomaly-map generation path,
- evaluation code.

## Core Implementation Decisions

- Use only the depth or height-map representation.
- Train only on normal samples.
- Work on overlapping local patches.
- Cache processed maps and patch metadata to avoid recomputing.
- Use `environment.yml` as the single source of truth for dependencies.
- Start with `OneClassSVM` as the main classical baseline.
- Keep `IsolationForest` optional and easy to add later.
- Use a small convolutional autoencoder, not a large architecture.
- Generate heatmaps by aggregating patch-level scores back to image space.
- Report image-level metrics first. Add pixel-level metrics only if the pipeline is stable.

## Conda Environment

The project should run from one documented conda environment.

Add at the repository root:

- `environment.yml`
- `pyproject.toml`
- optional `requirements.txt` only as a secondary export, not the main dependency file

Environment rules:

- the main setup command should be `conda env create -f environment.yml`
- the activation command should be `conda activate mvtec-3d-ad`
- the repository should support `pip install -e .` for editable development installs
- dependency versions should be pinned enough to keep the project reproducible
- use one environment for both the classical and deep pipelines
- keep the environment minimal and readable

Recommended environment content:

- `python`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pillow`
- `tifffile`
- `pyyaml`
- `tqdm`
- `jupyter`
- `pytorch`
- `torchvision` only if actually needed

Optional:

- GPU-specific PyTorch packages if the project is run on CUDA
- image IO packages depending on the exact MVTec 3D-AD file format

The README should document:

- environment creation
- environment activation
- dataset location
- the exact order of the main scripts

## Repository Layout

```text
.
├── README.md
├── environment.yml
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── classical.yaml
│   └── autoencoder.yaml
├── docs/
│   ├── pipeline.md
│   ├── data_format.md
│   └── decisions.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── fig/
│   ├── 01_dataset/
│   ├── 02_preprocessing/
│   ├── 03_patches/
│   ├── 04_classical/
│   ├── 05_autoencoder/
│   └── 06_results/
├── outputs/
│   ├── models/
│   ├── metrics/
│   └── logs/
├── scripts/
│   ├── visualize_depth_samples.py
│   ├── prepare_data.py
│   ├── train_classical.py
│   ├── train_autoencoder.py
│   └── run_benchmark.py
└── src/
    ├── data/
    │   ├── dataset_index.py
    │   ├── preprocessing.py
    │   ├── patching.py
    │   └── loaders.py
    ├── features/
    │   └── geometric_features.py
    ├── models/
    │   ├── ocsvm.py
    │   ├── isolation_forest.py
    │   └── autoencoder.py
    ├── training/
    │   ├── classical_trainer.py
    │   └── autoencoder_trainer.py
    ├── inference/
    │   ├── classical_inference.py
    │   ├── autoencoder_inference.py
    │   └── anomaly_maps.py
    ├── evaluation/
    │   ├── metrics.py
    │   ├── benchmark.py
    │   └── visualization.py
    └── utils/
        ├── config.py
        ├── io.py
        ├── logging.py
        └── seed.py
```

## Documentation and Readability Rules

The codebase should be easy to understand without digging through many files.

Rules:

- keep one clear responsibility per file
- put runnable entry points in `scripts/`
- keep reusable logic in `src/`
- keep filenames aligned with pipeline steps
- avoid hidden constants and put important values in config files
- add short module docstrings at the top of important files
- add short function docstrings for public functions
- keep notebooks optional and never make them the only implementation
- prefer explicit names over short clever names

Documentation requirements:

- `README.md` explains how to set up the environment and run the full pipeline
- `docs/pipeline.md` explains the execution flow
- `docs/data_format.md` explains how raw and processed data are stored
- `docs/decisions.md` records important implementation decisions
- each major pipeline step must save visual outputs into `fig/`
- the project should install cleanly in editable mode for local development

## End-to-End Pipeline

1. Index the dataset for the selected categories.
2. Save one raw depth-map preview per category in `fig/01_dataset/`.
3. Convert raw 3D samples into depth or height maps.
4. Preprocess and save the processed maps.
5. Create `train`, `val`, and `test` metadata files.
6. Extract overlapping patches from processed maps.
7. Train the classical model on normal training patches.
8. Train the autoencoder on the same normal training patches.
9. Run both models on test images patch by patch.
10. Aggregate patch scores into full anomaly heatmaps.
11. Convert heatmaps into image-level anomaly scores.
12. Compute metrics and save figures in `fig/`.

## What We Need to Implement

## 1. Raw Depth Sample Visualization

File: `scripts/visualize_depth_samples.py`

Implement:

- load one `xyz/*.tiff` training sample per category
- extract the `Z` channel as the raw depth map
- normalize only for display
- save one figure per category

Requirements:

- use the same selected categories as the rest of the project
- keep the raw values untouched for later preprocessing
- save figures with readable names

Output:

- `fig/01_dataset/<category>_sample_depth.png`

This step comes before indexing so we validate the spatial input format early.

## 2. Dataset Indexing

File: `src/data/dataset_index.py`

Implement:

- `build_index(dataset_root, categories) -> pd.DataFrame`
- `save_index(df, output_path)`
- `load_index(index_path) -> pd.DataFrame`

Each row should contain:

- `category`
- `split`
- `label`
- `sample_id`
- `depth_path`
- `mask_path`

Rules:

- `label=0` for normal, `label=1` for anomalous
- no anomalous samples in training
- keep test untouched

Output:

- `data/splits/index.csv`

## 3. Preprocessing

File: `src/data/preprocessing.py`

Implement:

- `load_depth_map(path) -> np.ndarray`
- `preprocess_depth_map(depth, cfg) -> np.ndarray`
- `save_processed_map(array, output_path)`

Preprocessing should do only the minimum required:

- load the depth or height channel,
- handle invalid values,
- crop or resize to a fixed size,
- normalize values,
- optionally keep a valid-pixel mask if needed.

Keep preprocessing identical for both pipelines.

Output:

- processed maps in `data/processed/<category>/`
- preprocessing figures in `fig/02_preprocessing/`

## 4. Split Creation

File: `scripts/prepare_data.py`

Implement:

- category selection
- indexing
- preprocessing
- normal-only `train/val` split
- export of split CSV files

Output files:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- dataset overview figures in `fig/01_dataset/`

## 5. Patch Extraction

File: `src/data/patching.py`

Implement:

- `extract_patches(image, patch_size, stride) -> tuple[np.ndarray, np.ndarray]`
- `extract_patches_with_mask(image, mask, patch_size, stride)`
- `aggregate_patch_values(values, coords, image_shape) -> np.ndarray`
- `aggregate_patch_maps(patch_maps, coords, image_shape) -> np.ndarray`

Requirements:

- return patch coordinates for reconstruction,
- support overlap,
- support inverse aggregation to image space,
- use deterministic ordering.

Use the same patch size and stride for both pipelines.

Save patch documentation figures:

- sample patch grids
- overlap visualization
- aggregation sanity-check plots

Output:

- figures in `fig/03_patches/`

## 6. Data Loaders

File: `src/data/loaders.py`

Implement:

- `PatchDataset` for training patches
- `ImageDataset` for full-image inference
- `build_patch_dataset(split_df, cfg)`
- `build_image_dataset(split_df, cfg)`

Training data:

- only normal samples

Inference data:

- full test images with labels and masks

## 7. Classical Feature Extraction

File: `src/features/geometric_features.py`

Implement:

- `extract_patch_features(patch) -> np.ndarray`
- `extract_batch_features(patches) -> np.ndarray`

Start with a small fixed feature vector:

- mean
- standard deviation
- min
- max
- range
- gradient magnitude mean
- gradient magnitude std
- roughness proxy
- Laplacian mean or variance

Keep the feature set simple and stable. Do not add many descriptors unless the baseline is clearly too weak.

## 8. Classical Model

Files:

- `src/models/ocsvm.py`
- `src/models/isolation_forest.py`
- `src/training/classical_trainer.py`
- `src/inference/classical_inference.py`

Implement:

- feature scaling with training-normal statistics only
- `OneClassSVM` training
- optional `IsolationForest` training
- model save and load
- patch-level scoring
- image-level score aggregation

Patch score path:

1. patch
2. feature vector
3. scaler
4. anomaly model
5. patch anomaly score

Image score:

- start with `max` score or `top-k mean`

Outputs:

- trained scaler
- trained classical model
- patch scores
- heatmaps
- image scores
- figures in `fig/04_classical/`

## 9. Autoencoder

Files:

- `src/models/autoencoder.py`
- `src/training/autoencoder_trainer.py`
- `src/inference/autoencoder_inference.py`

Implement a compact convolutional autoencoder:

- 2 to 4 encoder blocks
- bottleneck
- symmetric decoder
- output same shape as input patch

Training:

- normal patches only
- `MSE` or `MAE` reconstruction loss
- validation on held-out normal patches
- early stopping
- checkpoint saving

Inference:

- reconstruct each patch
- compute residual map
- convert residual map to patch anomaly score
- aggregate residuals back to image space

Outputs:

- trained checkpoint
- reconstruction loss curve
- residual heatmaps
- image scores
- figures in `fig/05_autoencoder/`

## 10. Anomaly Map Generation

File: `src/inference/anomaly_maps.py`

Implement:

- overlap-weighted aggregation of patch scores
- overlap-weighted aggregation of patch residual maps
- optional smoothing after aggregation
- normalization for visualization only

Important:

- do not normalize raw scores differently for the two methods before metric computation
- keep visualization transforms separate from evaluation values

## 11. Evaluation

Files:

- `src/evaluation/metrics.py`
- `src/evaluation/benchmark.py`
- `src/evaluation/visualization.py`

Implement image-level evaluation first:

- AUROC
- Average Precision or PR-AUC

Optional later:

- pixel-level AUROC
- thresholded localization metrics

Benchmark runner should:

- run both methods on the same test set,
- save one row per image,
- save one summary row per category,
- save one global summary file.
- save comparison figures for representative samples.

Output files:

- `outputs/metrics/per_image.csv`
- `outputs/metrics/per_category.csv`
- `outputs/metrics/summary.csv`
- `fig/06_results/`

## 12. Configuration

Files:

- `configs/base.yaml`
- `configs/classical.yaml`
- `configs/autoencoder.yaml`

Keep config values explicit:

- selected categories
- processed image size
- patch size
- stride
- normalization mode
- train/val split seed
- classical hyperparameters
- autoencoder hyperparameters
- output paths

Everything important should come from config, not hardcoded constants.

Also keep figure output paths in config so the documentation outputs are reproducible.

## 13. Scripts

### `scripts/visualize_depth_samples.py`

Responsibilities:

- load one raw spatial sample per selected category
- extract and visualize the depth map
- save the sample depth figures in `fig/01_dataset/`

### `scripts/prepare_data.py`

Responsibilities:

- index raw data
- preprocess maps
- write split files
- save dataset and preprocessing figures

### `scripts/train_classical.py`

Responsibilities:

- load config
- build training patches
- extract features
- fit scaler and model
- save artifacts
- save feature and score visualizations

### `scripts/train_autoencoder.py`

Responsibilities:

- load config
- build patch dataset
- train model
- save checkpoint and logs
- save training curves and reconstruction examples

### `scripts/run_benchmark.py`

Responsibilities:

- load both trained methods
- run test inference
- save metrics
- save example heatmaps
- save side-by-side comparison panels in `fig/06_results/`

## 14. Development Order

Implement in this order:

1. Repo structure, config loading, logging, seeds.
2. Raw depth sample visualization for one sample per category.
3. Dataset indexing and split files.
4. Preprocessing and processed-map caching.
5. Patch extraction and aggregation.
6. Classical feature extraction and `OneClassSVM`.
7. Classical image scoring and heatmap generation.
8. Autoencoder training and inference.
9. Shared benchmark and visualization code.
10. Optional `IsolationForest`.
11. Optional pixel-level localization metrics.

This order matters. The classical baseline should come first because it validates the shared data pipeline quickly.

## 15. Minimum Tests

Add small tests or sanity checks for:

- raw depth visualization loads one sample per category,
- dataset index contains expected counts,
- preprocessing returns fixed-size arrays,
- patch extraction plus aggregation reconstructs the full image layout correctly,
- feature extractor returns fixed-length vectors,
- classical model can fit on a small sample,
- autoencoder forward pass returns the expected shape,
- benchmark script writes metric files.

## 16. Output Artifacts

Each run should save:

- config snapshot
- model artifacts
- training log
- per-image scores
- per-category summary
- example anomaly maps
- step-by-step figures in `fig/`

Recommended artifact layout:

```text
fig/
  01_dataset/
  02_preprocessing/
  03_patches/
  04_classical/
  05_autoencoder/
  06_results/

outputs/
  models/
    classical/
    autoencoder/
  metrics/
    per_image.csv
    per_category.csv
    summary.csv
  logs/
```

## 17. Definition of Done

The implementation is done when:

- the conda environment is documented and reproducible,
- one sample raw depth figure exists per selected category,
- data preparation runs from one script,
- both models train on normal data only,
- both models run on the same test set,
- both models produce image scores and heatmaps,
- benchmark outputs comparable metrics for both,
- each major step is documented with figures in `fig/`,
- results are saved in a reproducible folder structure,
- the repository is understandable from the README and folder layout alone.

## 18. Immediate Build Plan

If we start implementing now, the first concrete tasks are:

1. Create the repo structure and config files.
2. Add `environment.yml`, `pyproject.toml`, `README.md`, and the `fig/` structure.
3. Implement raw depth sample visualization and save one figure per category.
4. Implement dataset indexing and split export.
5. Implement preprocessing and save processed maps.
6. Implement patch extraction and aggregation.
7. Implement classical features + `OneClassSVM`.
8. Add benchmark output for image-level metrics and figures.
9. Implement the autoencoder on top of the same patch pipeline.
