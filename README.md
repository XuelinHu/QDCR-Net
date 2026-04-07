# QDCR-Net

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10-blue" />
  <img height="20" src="https://img.shields.io/badge/pytorch-2.4.0-%23EE4C2C" />
  <img height="20" src="https://img.shields.io/badge/conda-environment-44A833" />
  <img height="20" src="https://img.shields.io/badge/cuda-12.1-76B900" />
  <img height="20" src="https://img.shields.io/badge/linux-ubuntu-lightgrey" />
  <img height="20" src="https://img.shields.io/badge/latex-paper%20first-008080" />
</p>

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

## Linux environment

The project is intended to run on Linux with NVIDIA CUDA and PyTorch.

- Use the Conda environment file: `environment.linux.yml`
- Target runtime: Linux + RTX 3090 + CUDA-capable PyTorch
- Recommended Python version: `3.10`

Create the environment on Linux with:

```bash
conda env create -f environment.linux.yml
conda activate qdcr-net
```

This repository is currently scaffolded for a pure PyTorch implementation path. No
TensorFlow, Paddle, or other training frameworks are required.

The paper-oriented experiment plan is centralized in
[paper/EXPERIMENT_WORKFLOW.md](/ds1/workspace/ai/QDCR-Net/paper/EXPERIMENT_WORKFLOW.md).

## Minimal smoke pipeline

The repository now includes a minimal trainable smoke pipeline:

- `conda run -n yolo python scripts/train.py` runs a short PyTorch training loop and saves a checkpoint under `outputs/checkpoints/`
- `conda run -n yolo python scripts/eval.py` reloads the latest checkpoint before evaluation
- `conda run -n yolo python scripts/eval_map.py` computes mAP-style metrics and writes prediction artifacts
- evaluation currently reports `loss`, `acc`, `box_iou`, `precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95`, `Params`, `GFLOPs`, and `FPS`
- training configs now support early stopping through `train.early_stopping`, and the best validation checkpoint is saved as `best.pt`
- metrics are written under `runs/` for TensorBoard-compatible workflows when `torch.utils.tensorboard` is available
- if no real dataset is present, the loader falls back to synthetic samples

For real data, use YOLO-style image and label pairs under the dataset roots configured in `configs/qdcr_net.yaml`.
The current implementation trains a fixed-query detector with explicit matching, NMS-based inference, and mAP-style evaluation. A baseline single-branch detector can be trained with `configs/base.yaml`.

## Next engineering steps

1. Implement the baseline detector and data pipeline.
2. Add the dual-branch backbone and cross-residual modules.
3. Add quality-aware fusion and the small-object neck.
4. Export experiment logs and result tables for the manuscript.
