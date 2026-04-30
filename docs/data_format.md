# Data Format

## Raw Data

Place the MVTec 3D-AD dataset under `data/raw/`.

The exact file structure can vary depending on how the dataset is unpacked, but the indexing code should normalize that into one tabular index.

## Processed Data

Processed maps should be saved under:

```text
data/processed/<category>/
```

Recommended saved artifacts:

- processed depth or height map
- valid-pixel mask
- optional metadata per sample if needed

Current processed file format:

- `<sample_id>_depth.npy`: normalized `float32` depth map
- `<sample_id>_mask.npy`: boolean valid-depth mask

Processed maps may be rectangular. With the default `preserve_area` resize mode,
`configs/base.yaml` uses `image_size` as a target pixel area reference rather
than as a required output shape, so long categories such as `rope` keep their
wide format instead of being padded into a square canvas. The final processed
height and width are snapped upward to the configured patch grid, so fixed-size
patches land exactly on the processed map edges.

## Split Files

Split metadata should live under `data/splits/`:

- `index.csv`
- `index_summary.csv`
- `train.csv`
- `val.csv`
- `test.csv`

Recommended columns:

- `category`
- `split`
- `defect_type`
- `label`
- `sample_id`
- `rgb_path`
- `xyz_path`
- `mask_path`
- `processed_depth_path`
- `processed_mask_path`
- `processed_valid_fraction`
- `processed_height`
- `processed_width`
- `foreground_fraction`
- `crop_top`
- `crop_bottom`
- `crop_left`
- `crop_right`

Label convention:

- `0`: normal
- `1`: anomalous

Current indexing choice:

- use the dataset-provided `train`, `validation`, and `test` folders directly
- normalize the split names in CSV files to `train`, `val`, and `test`
- keep paths relative to the repository root

Current dataset documentation artifact:

- `fig/01_dataset/index_summary.png`
- `fig/02_preprocessing/<category>_raw_vs_processed.png`
- `fig/03_patches/<category>_patch_overview.png`

## Output Data

Save experiment artifacts under `outputs/`:

- `outputs/models/`
- `outputs/metrics/`
- `outputs/logs/`

Save step-by-step visual documentation under `fig/`.
