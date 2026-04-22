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
- optional valid-pixel mask
- optional metadata per sample if needed

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

Label convention:

- `0`: normal
- `1`: anomalous

Current indexing choice:

- use the dataset-provided `train`, `validation`, and `test` folders directly
- normalize the split names in CSV files to `train`, `val`, and `test`
- keep paths relative to the repository root

Current dataset documentation artifact:

- `fig/01_dataset/index_summary.png`

## Output Data

Save experiment artifacts under `outputs/`:

- `outputs/models/`
- `outputs/metrics/`
- `outputs/logs/`

Save step-by-step visual documentation under `fig/`.
