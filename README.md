# QDCR-Net

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10-blue" />
  <img height="20" src="https://img.shields.io/badge/pytorch-2.4.0-%23EE4C2C" />
  <img height="20" src="https://img.shields.io/badge/conda-environment-44A833" />
  <img height="20" src="https://img.shields.io/badge/cuda-12.1-76B900" />
  <img height="20" src="https://img.shields.io/badge/linux-ubuntu-lightgrey" />
  <img height="20" src="https://img.shields.io/badge/latex-paper%20first-008080" />
</p>

QDCR-Net is a paper-first underwater detection scaffold built around a lightweight
PyTorch implementation of `QDCR-Net` (Quality-aware Dual-branch Cross-Residual Network)
and a matching baseline detector.

Chinese documentation is available in [README-CN.md](README-CN.md).

## Encoding and comments

- All Python files under `src/` and `scripts/` include UTF-8 Chinese comments.
- Python files use `# -*- coding: utf-8 -*-` at the top.
- Shell scripts also include Chinese comments and are saved in UTF-8.

## Quick start

Create the environment on Linux:

```bash
conda env create -f environment.linux.yml
conda activate qdcr-net
```

Typical commands:

```bash
conda run -n yolo python scripts/train.py --config configs/qdcr_net.yaml
conda run -n yolo python scripts/eval.py --config configs/qdcr_net.yaml
conda run -n yolo python scripts/eval_map.py --config configs/qdcr_net.yaml
bash scripts/run_download_prepare.sh
bash scripts/run_formal_matrix.sh
bash scripts/continue_overnight.sh
```

## Default dataset, snapshot, and output paths

Default training and validation dataset paths from `configs/qdcr_net.yaml` and `configs/base.yaml`:

- `data/datasets/URPC2018/train/images`
- `data/datasets/URPC2018/train/labels`
- `data/datasets/URPC2018/val/images`
- `data/datasets/URPC2018/val/labels`
- `data/datasets/URPC2018_enhanced`

Downloaded archive and extracted dataset paths used by scripts:

- `data/raw/Brackish.zip`
- `data/raw/RUOD.zip`
- `data/raw/UOv2.zip`
- `data/datasets/downloads/Brackish`
- `data/datasets/downloads/RUOD`
- `data/datasets/downloads/UOv2`

Formal full-data dataset paths from `configs/formal/*.yaml`:

- `data/datasets/downloads/UOv2/train/images`
- `data/datasets/downloads/UOv2/valid/images`
- `data/datasets/downloads/Brackish/train/images`
- `data/datasets/downloads/Brackish/valid/images`
- `data/datasets/downloads/RUOD/RUOD/train/images`
- `data/datasets/downloads/RUOD/RUOD/valid/images`

Default checkpoint snapshot and evaluation artifact paths:

- `outputs/checkpoints/qdcr_net_smoke/latest.pt`
- `outputs/checkpoints/qdcr_net_smoke/best.pt`
- `outputs/checkpoints/qdcr_net_smoke/predictions.json`
- `outputs/checkpoints/qdcr_net_smoke/metrics.json`
- `outputs/checkpoints/baseline_detector/latest.pt`
- `outputs/checkpoints/baseline_detector/best.pt`
- `outputs/checkpoints/baseline_detector/predictions.json`
- `outputs/checkpoints/baseline_detector/metrics.json`
- `runs/qdcr_net_smoke/scalars.tsv`
- `runs/baseline_detector/scalars.tsv`

Automation, visualization, and experiment record paths:

- `outputs/automation/`
- `outputs/visualizations/`
- `data/logs/`
- `data/results/`

Formal full-data experiment snapshot paths:

- `outputs/checkpoints/baseline_uov2_full/latest.pt`
- `outputs/checkpoints/baseline_uov2_full/best.pt`
- `outputs/checkpoints/qdcr_uov2_full/latest.pt`
- `outputs/checkpoints/qdcr_uov2_full/best.pt`
- `outputs/checkpoints/baseline_brackish_full/latest.pt`
- `outputs/checkpoints/baseline_brackish_full/best.pt`
- `outputs/checkpoints/qdcr_brackish_full/latest.pt`
- `outputs/checkpoints/qdcr_brackish_full/best.pt`
- `outputs/checkpoints/baseline_ruod_full/latest.pt`
- `outputs/checkpoints/baseline_ruod_full/best.pt`
- `outputs/checkpoints/qdcr_ruod_full/latest.pt`
- `outputs/checkpoints/qdcr_ruod_full/best.pt`
- `runs/baseline_uov2_full/scalars.tsv`
- `runs/qdcr_uov2_full/scalars.tsv`
- `runs/baseline_brackish_full/scalars.tsv`
- `runs/qdcr_brackish_full/scalars.tsv`
- `runs/baseline_ruod_full/scalars.tsv`
- `runs/qdcr_ruod_full/scalars.tsv`

## Repository path index

### Root files

- `README.md`
- `README-CN.md`
- `environment.linux.yml`

### Runtime entry files

- `scripts/train.py`: training entry point
- `scripts/eval.py`: evaluation entry point
- `scripts/eval_map.py`: evaluation + mAP reporting
- `scripts/prepare_dataset.py`: dataset connectivity inspection
- `scripts/generate_sample_config.py`: generate sample YAML configs
- `scripts/run_download_prepare.sh`: download datasets and generate configs
- `scripts/run_formal_matrix.sh`: run the formal experiment matrix
- `scripts/continue_overnight.sh`: continue overnight sample runs

### Config files

- `configs/qdcr_net.yaml`
- `configs/base.yaml`
- `configs/formal/README.md`
- `configs/formal/base_uov2_full.yaml`
- `configs/formal/qdcr_uov2_full.yaml`
- `configs/formal/base_brackish_full.yaml`
- `configs/formal/qdcr_brackish_full.yaml`
- `configs/formal/base_ruod_full.yaml`
- `configs/formal/qdcr_ruod_full.yaml`

Generated sample configs are written here by scripts:

- `configs/generated/qdcr_brackish_sample.yaml`
- `configs/generated/base_brackish_sample.yaml`
- `configs/generated/qdcr_ruod_sample.yaml`
- `configs/generated/base_ruod_sample.yaml`

### Source code files

- `src/__init__.py`
- `src/datasets/__init__.py`
- `src/datasets/underwater_detection.py`
- `src/engine/__init__.py`
- `src/engine/detection_ops.py`
- `src/engine/trainer.py`
- `src/losses/__init__.py`
- `src/losses/detection_loss.py`
- `src/models/__init__.py`
- `src/models/qdcr_net.py`
- `src/models/modules/__init__.py`
- `src/models/modules/cross_residual.py`
- `src/models/modules/quality_aware_fusion.py`
- `src/utils/__init__.py`
- `src/utils/config.py`
- `src/utils/logger.py`
- `src/utils/tracker.py`

### Data paths

- `data/datasets/README.md`
- `data/datasets/URPC2018/train/images`
- `data/datasets/URPC2018/train/labels`
- `data/datasets/URPC2018/val/images`
- `data/datasets/URPC2018/val/labels`
- `data/datasets/URPC2018_enhanced`
- `data/datasets/downloads/Brackish`
- `data/datasets/downloads/Brackish/train/images`
- `data/datasets/downloads/Brackish/valid/images`
- `data/datasets/downloads/RUOD`
- `data/datasets/downloads/RUOD/RUOD/train/images`
- `data/datasets/downloads/RUOD/RUOD/valid/images`
- `data/datasets/downloads/UOv2`
- `data/datasets/downloads/UOv2/train/images`
- `data/datasets/downloads/UOv2/valid/images`
- `data/raw/README.md`
- `data/raw/Brackish.zip`
- `data/raw/RUOD.zip`
- `data/raw/UOv2.zip`
- `data/logs/README.md`
- `data/results/README.md`

### Snapshot and output paths

- `outputs/checkpoints/README.md`
- `outputs/checkpoints/<experiment>/latest.pt`
- `outputs/checkpoints/<experiment>/best.pt`
- `outputs/checkpoints/<experiment>/predictions.json`
- `outputs/checkpoints/<experiment>/metrics.json`
- `outputs/automation/README.md`
- `outputs/automation/<task_name>.log`
- `outputs/automation/download_prepare.log`
- `outputs/automation/*_train.log`
- `outputs/automation/*_eval.log`
- `outputs/automation/ruod_unzip.log`
- `outputs/automation/ruod_qdcr_config.log`
- `outputs/automation/ruod_base_config.log`
- `outputs/visualizations/README.md`
- `runs/.gitkeep`
- `runs/<experiment>/scalars.tsv`

### Paper and manuscript files

- `paper/EXPERIMENT_WORKFLOW.md`
- `paper/MAIN_EXPERIMENT_MATRIX.md`
- `paper/main.tex`
- `paper/refs.bib`
- `paper/figures/README.md`
- `paper/sections/abstract.tex`
- `paper/sections/introduction.tex`
- `paper/sections/related_work.tex`
- `paper/sections/method.tex`
- `paper/sections/experiments.tex`
- `paper/sections/conclusion.tex`
- `paper/tables/README.md`
- `paper/tables/main_comparison.tex`

## Notes on path behavior

- The trainer reads dataset roots from the selected YAML config file.
- If no real dataset is found, `src/datasets/underwater_detection.py` falls back to synthetic samples.
- Checkpoint snapshots are written inside the `experiment.output_dir` path from the config.
- TensorBoard logs are written under `/ds1/runs/<project>/`, and each launch gets an independent run subdirectory containing `scalars.tsv` plus event files.
- Formal full-data runs use configs under `configs/formal/`.

## Recommended reading order

1. `scripts/train.py`
2. `scripts/eval_map.py`
3. `src/engine/trainer.py`
4. `src/datasets/underwater_detection.py`
5. `src/engine/detection_ops.py`
6. `src/models/qdcr_net.py`
