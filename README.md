# MVTec 3D-AD Anomaly Detection

This repository implements a shared anomaly detection pipeline for MVTec 3D-AD with two methods:

1. A classical baseline based on handcrafted patch features and a one-class model.
2. A deep baseline based on a compact convolutional autoencoder.

The main implementation guide is in `implementation_plan.md`. This README is the fast entry point for setup, structure, and execution.

## Environment

Create the conda environment from the root of the repository:

```bash
conda env create -f environment.yml
conda activate mvtec-3d-ad
pip install -e .
```

If you need a plain pip export, `requirements.txt` is included as a secondary reference. The conda environment is the source of truth.

`pyproject.toml` is included so the repository can be installed in editable mode during development.

## Expected Dataset Location

Put the MVTec 3D-AD dataset under:

```text
data/raw/
```

If your dataset lives elsewhere, update the paths in `configs/base.yaml`.

## Project Structure

```text
configs/   Shared configuration files
data/      Raw data, processed maps, and split metadata
docs/      Human-readable documentation for the pipeline and decisions
fig/       Figures created at every major pipeline step
outputs/   Model artifacts, metrics, and logs
pyproject.toml  Packaging and editable-install configuration
scripts/   Runnable entry points
src/       Reusable implementation modules
```

## Figure Policy

Every major step must produce figures in `fig/` so the workflow is documented visually:

- `fig/01_dataset/`
- `fig/02_preprocessing/`
- `fig/03_patches/`
- `fig/04_classical/`
- `fig/05_autoencoder/`
- `fig/06_results/`

The goal is to make debugging, reporting, and review easy without digging through code.

## Planned Run Order

1. Save one sample depth-map visualization per category:

```bash
python scripts/visualize_depth_samples.py
```

2. Prepare data and splits:

```bash
python scripts/prepare_data.py
```

3. Train the classical baseline:

```bash
python scripts/train_classical.py
```

4. Train the autoencoder baseline:

```bash
python scripts/train_autoencoder.py
```

5. Benchmark both methods:

```bash
python scripts/run_benchmark.py
```

The scripts are scaffolded and aligned with the implementation plan, but the full pipeline logic still needs to be implemented.

## Main Docs

- `implementation_plan.md`: implementation-first project plan
- `docs/pipeline.md`: end-to-end execution flow
- `docs/data_format.md`: folder and file expectations
- `docs/decisions.md`: implementation decisions and conventions
