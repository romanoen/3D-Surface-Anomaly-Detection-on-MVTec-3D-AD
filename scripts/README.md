# Scripts

This folder contains the runnable entry points for the project.

- `visualize_depth_samples.py`: save one sample depth-map preview per category
- `prepare_data.py`: dataset indexing, preprocessing, split export, and dataset figures
- `train_classical.py`: classical baseline training and documentation figures
- `infer_classical.py`: classical baseline image scoring and heatmap generation
- `train_autoencoder.py`: autoencoder training and training-curve figures
- `run_benchmark.py`: final evaluation, metrics export, and comparison figures

Keep reusable logic in `src/` and keep these files focused on orchestration.
