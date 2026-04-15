# Main Experiment Matrix

This file is the paper-facing execution board for the current repository. It separates:

- `completed core runs` that already exist in the repo;
- `next main-comparison runs` that should be added next;
- `follow-up studies` required for a convincing paper package.

The main paper story should no longer stop at `BaselineDetector vs QDCRNet`. The minimum strong
comparison set is now:

- `BaselineDetector`
- `Faster R-CNN`
- `YOLOv8`
- `RT-DETR`
- `Underwater-Enhance+Detector`
- `QDCRNet`

Primary training datasets:

- `Brackish-Roboflow`
- `RUOD`
- `UOv2`

The current local `Brackish` directory is already a Roboflow export:

- [data/datasets/downloads/Brackish/data.yaml](/ds1/workspace/ai/QDCR-Net/data/datasets/downloads/Brackish/data.yaml)
- [data/datasets/downloads/Brackish/README.roboflow.txt](/ds1/workspace/ai/QDCR-Net/data/datasets/downloads/Brackish/README.roboflow.txt)

Additional evaluation blocks after the main table:

- cross-dataset generalization;
- RUOD degradation robustness;
- module ablations.

## Status Legend

- `completed`: metrics/checkpoints already exist in the repository
- `planned`: should be implemented and executed next
- `blocked`: depends on missing model integration or missing dataset path confirmation

## Stage A: Core Runs Already Completed

These are the runs currently present under `outputs/checkpoints/`.

| Dataset | Model | Config | Output Dir | Status |
| --- | --- | --- | --- | --- |
| UOv2 | BaselineDetector | [base_uov2_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_uov2_full.yaml) | `outputs/checkpoints/baseline_uov2_full` | completed |
| UOv2 | QDCRNet | [qdcr_uov2_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_uov2_full.yaml) | `outputs/checkpoints/qdcr_uov2_full` | completed |
| Brackish | BaselineDetector | [base_brackish_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_brackish_full.yaml) | `outputs/checkpoints/baseline_brackish_full` | completed |
| Brackish | QDCRNet | [qdcr_brackish_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_brackish_full.yaml) | `outputs/checkpoints/qdcr_brackish_full` | completed |
| Brackish-Roboflow | BaselineDetector | [base_brackish_rf_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_brackish_rf_full.yaml) | `outputs/checkpoints/baseline_brackish_rf_full` | planned |
| Brackish-Roboflow | QDCRNet | [qdcr_brackish_rf_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_brackish_rf_full.yaml) | `outputs/checkpoints/qdcr_brackish_rf_full` | planned |
| RUOD | BaselineDetector | [base_ruod_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_full.yaml) | `outputs/checkpoints/baseline_ruod_full` | completed |
| RUOD | QDCRNet | [qdcr_ruod_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_full.yaml) | `outputs/checkpoints/qdcr_ruod_full` | completed |

## Stage A2: Internal Model Retuning Runs

After the internal detection-head refactor, the old internal-model checkpoints are no longer the
right basis for new reporting. Retraining should move to these tuned configs first:

| Dataset | BaselineDetector | QDCRNet | Status |
| --- | --- | --- | --- |
| UOv2 | [base_uov2_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_uov2_tuned_full.yaml) | [qdcr_uov2_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_uov2_tuned_full.yaml) | ready to run |
| RUOD | [base_ruod_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_tuned_full.yaml) | [qdcr_ruod_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_tuned_full.yaml) | ready to run |
| Brackish-Roboflow | [base_brackish_rf_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_brackish_rf_tuned_full.yaml) | [qdcr_brackish_rf_tuned_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_brackish_rf_tuned_full.yaml) | ready to run |

The tuned internal-model policy is:

- `image_size`: `640`
- `feature_dim`: `64`
- `num_queries`: `16`
- `max_objects`: `16`
- `epochs`: `120`
- `lr`: `0.0003`
- `eval.conf_thresh`: `0.05`

An optional mAP-oriented variant is also prepared for the same six internal-model runs:

- use `*_map_tuned_full.yaml`
- switch early stopping monitor to `map50`
- switch early stopping mode to `max`
- relax score filtering to `eval.conf_thresh = 0.01`

## Stage B: Minimum Strong Main Comparison

This is the new paper-grade target matrix. Only the first and last columns are implemented today.
The middle four model families still need integration.

| Dataset | BaselineDetector | Faster R-CNN | YOLOv8 | RT-DETR | Underwater-Enhance+Detector | QDCRNet |
| --- | --- | --- | --- | --- | --- | --- |
| Brackish-Roboflow | planned | planned | planned | planned | planned | planned |
| RUOD | completed | planned | planned | planned | planned | completed |
| UOv2 | completed | planned | planned | planned | planned | completed |

Recommended execution order inside this matrix:

1. `BaselineDetector`
2. `QDCRNet`
3. `YOLOv8`
4. `Faster R-CNN`
5. `RT-DETR`
6. `Underwater-Enhance+Detector`

## Stage C: Formal Config Naming Plan

The repository should use the following naming convention for new formal configs:

`<model_family>_<dataset>_full.yaml`

Internal-model retraining configs may use:

`<model_family>_<dataset>_tuned_full.yaml`

Examples:

- `base_brackish_rf_full.yaml`
- `faster_rcnn_brackish_rf_full.yaml`
- `yolov8_brackish_rf_full.yaml`
- `rtdetr_brackish_rf_full.yaml`
- `underwater_enhance_det_brackish_rf_full.yaml`
- `qdcr_brackish_rf_full.yaml`
- `faster_rcnn_ruod_full.yaml`
- `yolov8_ruod_full.yaml`
- `rtdetr_ruod_full.yaml`
- `underwater_enhance_det_ruod_full.yaml`
- `faster_rcnn_uov2_full.yaml`
- `yolov8_uov2_full.yaml`
- `rtdetr_uov2_full.yaml`
- `underwater_enhance_det_uov2_full.yaml`

## Stage D: Follow-Up Experiments Required for Paper Credibility

### D1. Cross-Dataset Generalization

At least these runs should be added after the main table stabilizes:

| Train | Test | Models | Status |
| --- | --- | --- | --- |
| Brackish-Roboflow | RUOD | `BaselineDetector`, `YOLOv8`, `QDCRNet` | planned |
| RUOD | UOv2 | `BaselineDetector`, `YOLOv8`, `QDCRNet` | planned |
| UOv2 | Brackish-Roboflow | `BaselineDetector`, `YOLOv8`, `QDCRNet` | planned |

### D2. RUOD Degradation Robustness

The RUOD test split should be broken down by degradation subset.

| Dataset | Split | Models | Status |
| --- | --- | --- | --- |
| RUOD | `testb` | `BaselineDetector`, `YOLOv8`, `QDCRNet` | baseline/qdcr configs ready |
| RUOD | `testc` | `BaselineDetector`, `YOLOv8`, `QDCRNet` | baseline/qdcr configs ready |
| RUOD | `testl` | `BaselineDetector`, `YOLOv8`, `QDCRNet` | baseline/qdcr configs ready |

The repository now includes these ready-to-run RUOD degradation eval configs:

- [base_ruod_testb_eval.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_testb_eval.yaml)
- [base_ruod_testc_eval.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_testc_eval.yaml)
- [base_ruod_testl_eval.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_testl_eval.yaml)
- [qdcr_ruod_testb_eval.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_testb_eval.yaml)
- [qdcr_ruod_testc_eval.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_testc_eval.yaml)
- [qdcr_ruod_testl_eval.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_testl_eval.yaml)

### D3. Ablations

These are the minimum module-level ablations that align with the paper claims:

| Variant | Suggested Dataset | Status |
| --- | --- | --- |
| `w/o dual_branch` | RUOD | planned |
| `w/o quality_aware` | RUOD | planned |
| `w/o cross_residual` | RUOD | planned |
| `w/o small_object_neck` | RUOD | planned |
| `different loss weights` | RUOD | planned |
| `different input sizes` | Brackish-Roboflow or RUOD | planned |

## Frozen Training Policy

Until mAP becomes stable enough for fair model-to-model comparison, keep the default formal policy:

- `epochs`: `100`
- `batch_size`: `4`
- `max_batches_per_epoch`: `0`
- `eval.max_batches`: `0`
- `early stopping`: enabled
- `monitor`: validation `loss`
- `patience`: `10`
- `best checkpoint`: `best.pt`

Once detection quality becomes informative, consider switching early stopping and best-checkpoint
selection from `loss` to `map50`.
