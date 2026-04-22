# Decisions

This file records implementation decisions so the repository stays understandable.

## Initial Decisions

- The shared representation is a depth or height map.
- Both pipelines use the same processed inputs and patch extraction logic.
- Training uses only normal samples.
- The classical baseline starts with `OneClassSVM`.
- `IsolationForest` is optional and added only after the main baseline is stable.
- The deep baseline is a small convolutional autoencoder.
- Image-level metrics come first. Pixel-level metrics are optional.
- `environment.yml` is the main dependency file.
- `fig/` is required for visual documentation at every major step.
- Before indexing, save one raw depth-map preview per category in `fig/01_dataset/`.

## Updates

- The default category selection in `configs/base.yaml` now includes all 10 locally available MVTec 3D-AD categories so the dataset overview and later pipeline steps cover the full extracted dataset by default.
- Dataset preview figures now mask invalid zero-depth pixels and normalize only over valid object measurements, because the raw unmasked Z maps were visually misleading for human interpretation.
- Dataset preview colorbars use simple tick labels `0 = closer` and `1 = farther` without extra explanatory text, to keep the figures easy to read.
- Dataset indexing now uses the dataset-provided `train`, `validation`, and `test` folders directly, writes normalized split names `train`, `val`, `test` into CSV, and saves an index summary figure in `fig/01_dataset/index_summary.png`.

## Update Rule

Whenever a major implementation choice changes, add a short note here with:

- what changed
- why it changed
- which files or configs were affected
