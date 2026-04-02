# QDCR-Net

This repository is organized for a paper-first underwater detection project built around
`QDCR-Net` (Quality-aware Dual-branch Cross-Residual Network).

## Directory layout

- `paper/`: LaTeX manuscript, references, figures, and tables.
- `data/`: dataset links, logs, and processed experiment records.
- `src/`: Python source code for datasets, models, engine, utils, and losses.
- `scripts/`: entry scripts for dataset preparation, training, and evaluation.
- `configs/`: experiment configuration files.
- `outputs/`: checkpoints and visualization artifacts.

## Current status

This is the initial project scaffold. The structure is ready for:

- manuscript writing in LaTeX;
- baseline implementation with a YOLO-style detector;
- QDCR-Net module implementation;
- ablation and robustness experiments on URPC2018 and RUOD.

## Next engineering steps

1. Implement the baseline detector and data pipeline.
2. Add the dual-branch backbone and cross-residual modules.
3. Add quality-aware fusion and the small-object neck.
4. Export experiment logs and result tables for the manuscript.
