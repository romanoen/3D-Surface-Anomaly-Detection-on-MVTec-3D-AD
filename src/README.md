# Source Modules

This folder contains the reusable implementation modules.

- `src/data/`: indexing, preprocessing, patch extraction, loaders
- `src/features/`: handcrafted geometric features
- `src/models/`: model definitions and wrappers
- `src/training/`: training logic
- `src/inference/`: patch scoring and anomaly-map generation
- `src/evaluation/`: metrics and visualization
- `src/utils/`: config, IO, logging, and reproducibility helpers

The goal is to keep one responsibility per file and make the codebase easy to scan.
