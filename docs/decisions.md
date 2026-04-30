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
- Preprocessing now extracts the `Z` channel, treats finite positive values as valid depth, resizes depth and mask with aspect-ratio-preserving `preserve_area` geometry by default, normalizes only over valid pixels, and saves `depth.npy` plus `mask.npy` files for later patch extraction. The configured `image_size` now defines the target pixel area, not a mandatory square canvas.
- The `preserve_area` resize snaps processed height and width upward to the configured patch grid, so `(axis_length - patch_size)` is divisible by `stride` and patch extraction does not need extra edge-aligned patches on freshly processed maps.
- Preprocessing now infers a rough object foreground from closer valid depth values, crops to the foreground bounding box plus a configurable margin before resizing, and stores the crop box in the split metadata so later patch extraction can focus more on the object than the background.
- Patch extraction now uses deterministic row-major coordinates with an extra edge-aligned patch when stride alone would leave uncovered borders, so both pipelines reconstruct the full processed image layout consistently.
- Patch aggregation now averages overlapping scalar scores and patch maps in image space instead of summing them, which keeps heatmaps comparable across regions with different overlap counts.
- Patch overview figures now focus on the dense patch grid, sliding-window order, and true per-pixel coverage count, without extra reconstruction or patch-contact-sheet panels.
- Patch overview figures now support rectangular processed maps and lay out very wide crops vertically, so categories like `rope` show the real aspect-adapted patch raster instead of a mostly empty padded square.
- Data loaders now keep patch metadata lightweight, load processed maps lazily, and use a small `min_valid_fraction` threshold from config to skip empty patches. The first classical feature extractor returns a fixed nine-value vector per patch using simple depth statistics, gradient magnitude, roughness, and Laplacian variance.

## Update Rule

Whenever a major implementation choice changes, add a short note here with:

- what changed
- why it changed
- which files or configs were affected
