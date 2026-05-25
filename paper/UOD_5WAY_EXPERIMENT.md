# RUOD 5-Way Experiment Plan

This file freezes the single-dataset comparison package for `RUOD`.

## Scope

The current 5-way comparison uses these models only:

- `BaselineDetector`
- `QDCRNet (vanilla cross residual)`
- `QDCRNet (learnable-threshold cross residual)`
- `QDCRNet (sparse residual)`
- `YOLOv8`

Brackish-related experiments are excluded from this round.

## Config List

| Variant | Config | Output Dir | Run Dir |
| --- | --- | --- | --- |
| BaselineDetector | [base_ruod_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_tuned_full.yaml) | `outputs/checkpoints/baseline_ruod_tuned_full` | `runs/baseline_ruod_tuned_full` |
| QDCR vanilla | [qdcr_ruod_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_tuned_full.yaml) | `outputs/checkpoints/qdcr_ruod_tuned_full` | `runs/qdcr_ruod_tuned_full` |
| QDCR threshold | [qdcr_ruod_threshold_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_threshold_tuned_full.yaml) | `outputs/checkpoints/qdcr_ruod_threshold_tuned_full` | `runs/qdcr_ruod_threshold_tuned_full` |
| QDCR sparse | [qdcr_ruod_sparse_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_sparse_tuned_full.yaml) | `outputs/checkpoints/qdcr_ruod_sparse_tuned_full` | `runs/qdcr_ruod_sparse_tuned_full` |
| YOLOv8 | [yolov8_ruod_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/yolov8_ruod_full.yaml) | `outputs/checkpoints/yolov8_ruod_full` | `runs/yolov8_ruod_full` |

## Frozen Dataset Policy

- dataset: `RUOD`
- train split: `data/datasets/downloads/RUOD/RUOD/train/images`
- val split: `data/datasets/downloads/RUOD/RUOD/valid/images`
- classes: `4`
- image size:
  `640` for all five runs in this package
- batch size:
  `100` for all five runs in this package

## Frozen Training Parameters

### Internal models

Applies to:

- `BaselineDetector`
- `QDCR vanilla`
- `QDCR threshold`
- `QDCR sparse`

Shared settings:

- optimizer: `AdamW`
- learning rate: `0.0003`
- weight decay: `0.0005`
- epochs: `120`
- batch size: `100`
- feature dim: `64`
- num queries: `16`
- max objects: `16`
- early stopping:
  `enabled=true`, `monitor=loss`, `mode=min`, `patience=15`, `min_delta=0.0005`
- eval checkpoint: `best`
- eval conf thresh: `0.05`
- eval IoU thresh: `0.5`

Loss function for internal models:

- class: `DetectionLoss`
- classification term:
  focal-style classification loss
- box regression term:
  `SmoothL1Loss`
- IoU term:
  diagonal IoU penalty from matched boxes
- box weight: `5.0`
- IoU weight: `2.0`
- focal gamma: `2.0`
- background weight: `0.2`

### Variant-specific cross residual settings

`BaselineDetector`

- no dual branch
- no cross residual

`QDCR vanilla`

- cross residual mode: `vanilla`
- quality aware fusion: `true`
- cross residual stages: `stage2`, `stage3`
- small object neck: `true`

`QDCR threshold`

- cross residual mode: `learnable_threshold`
- threshold init: `0.1`
- threshold slope: `10.0`
- quality aware fusion: `true`
- cross residual stages: `stage2`, `stage3`
- small object neck: `true`

`QDCR sparse`

- cross residual mode: `sparse_residual`
- sparse mode: `soft_threshold`
- sparse lambda init: `0.1`
- top-k ratio placeholder: `0.5`
- quality aware fusion: `true`
- cross residual stages: `stage2`, `stage3`
- small object neck: `true`

### YOLOv8

- model: `yolov8`
- weights init: `yolov8n.pt`
- epochs: `100`
- batch size: `100`
- image size: `640`
- learning rate: `0.01`
- weight decay: `0.0005`
- early stopping patience: `20`
- eval split: `val`
- eval conf thresh: `0.25`
- eval IoU thresh: `0.5`
- loss family:
  Ultralytics built-in detection loss with `box`, `cls`, `dfl`

## Run Commands

### Run one-by-one

Internal models:

```bash
conda run -n yolo python scripts/train.py --config configs/formal/base_ruod_tuned_full.yaml
conda run -n yolo python scripts/train.py --config configs/formal/qdcr_ruod_tuned_full.yaml
conda run -n yolo python scripts/train.py --config configs/formal/qdcr_ruod_threshold_tuned_full.yaml
conda run -n yolo python scripts/train.py --config configs/formal/qdcr_ruod_sparse_tuned_full.yaml
```

YOLOv8:

```bash
conda run -n yolo python scripts/train_ultralytics.py --config configs/formal/yolov8_ruod_full.yaml
```

### Run as one batch

```bash
CONFIG_GLOB="configs/formal/base_ruod_tuned_full.yaml configs/formal/qdcr_ruod_tuned_full.yaml configs/formal/qdcr_ruod_threshold_tuned_full.yaml configs/formal/qdcr_ruod_sparse_tuned_full.yaml configs/formal/yolov8_ruod_full.yaml" scripts/run_formal_matrix.sh
```

## Required Saved Artifacts

Every experiment directory must contain:

- `experiment_config.yaml`
  complete frozen config snapshot
- `experiment_summary.json`
  key hyperparameter summary including optimizer, loss, lr, batch size, image size, thresholds, and variant settings
- `metrics.json`
  final eval metrics
- `predictions.json`
  internal models only
- `best.pt`
  best checkpoint

Run logging locations:

- internal models:
  `runs/<experiment_name>/`
- YOLOv8:
  `runs/<experiment_name>/` plus Ultralytics output under `outputs/checkpoints/yolov8_ruod_full`
- automation logs:
  `outputs/automation/<experiment_name>_train.log`
  `outputs/automation/<experiment_name>_eval.log`

## Expected Reporting Fields

The main result table should use:

- `Model`
- `Dataset`
- `Loss`
- `Acc`
- `Box IoU`
- `mAP50`
- `mAP50-95`
- `Precision`
- `Recall`
- `Params (M)`
- `GFLOPs`
- `FPS`

Recommended row order:

1. `BaselineDetector`
2. `YOLOv8`
3. `QDCR vanilla`
4. `QDCR threshold`
5. `QDCR sparse`

## Notes

- If `batch_size=100` exceeds actual memory capacity for any internal model, reduce only the failing run and record the deviation explicitly in `experiment_summary.json` and the final paper table notes.
- Keep the dataset split unchanged across all five runs.
- Do not mix this 5-way package with Brackish experiments.
