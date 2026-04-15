This directory stores formal full-data experiment configs for the paper.

## Current Scope

Implemented today:

- datasets: `UOv2`, `Brackish`, `RUOD`
- models: `BaselineDetector`, `QDCRNet`
- schedule: full train/val passes, early stopping enabled, `best.pt` used for reporting

The local `Brackish` dataset under `data/datasets/downloads/Brackish/` is already a Roboflow
YOLO export and contains `data.yaml` plus `README.roboflow.txt`. New formal configs may therefore
use the explicit dataset token `brackish_rf` while still pointing to the same local path.

## Planned Expansion

The next comparison wave should add:

- `Faster R-CNN`
- `YOLOv8`
- `RT-DETR`
- `Underwater-Enhance+Detector`

RUOD degradation evaluation configs are allowed to use:

`<model_family>_ruod_test{b,c,l}_eval.yaml`

These configs should point `experiment.output_dir` to an already trained RUOD checkpoint directory
and only swap `dataset.val_root` to the requested degradation split.

## Naming Convention

Use:

`<model_family>_<dataset>_full.yaml`

For the internal `BaselineDetector` and `QDCRNet`, tuned retraining configs may use:

`<model_family>_<dataset>_tuned_full.yaml`

If the goal is to bias checkpoint selection toward detection quality rather than raw loss, use:

`<model_family>_<dataset>_map_tuned_full.yaml`

Examples:

- `base_ruod_full.yaml`
- `qdcr_uov2_full.yaml`
- `base_ruod_tuned_full.yaml`
- `qdcr_brackish_rf_tuned_full.yaml`
- `base_ruod_map_tuned_full.yaml`
- `qdcr_uov2_map_tuned_full.yaml`
- `faster_rcnn_brackish_rf_full.yaml`
- `yolov8_ruod_full.yaml`
- `rtdetr_uov2_full.yaml`
- `underwater_enhance_det_brackish_rf_full.yaml`

Dataset token recommendations:

- `uov2`
- `ruod`
- `brackish`
- `brackish_rf`

Model family token recommendations:

- `base`
- `qdcr`
- `faster_rcnn`
- `yolov8`
- `rtdetr`
- `underwater_enhance_det`

## Authoring Rules

- keep `experiment.name`, `output_dir`, and checkpoint directory stem aligned
- keep training policy consistent across the main comparison unless a method requires a documented exception
- keep all formal configs runnable through `scripts/run_formal_matrix.sh`
- avoid changing datasets or image size mid-matrix without updating the paper matrix document
- prefer the `*_tuned_full.yaml` internal-model configs for any retraining after the detection-head refactor
- prefer the `*_map_tuned_full.yaml` variants when `map50` is no longer flat at `0` and you want best-checkpoint selection to follow detection quality
