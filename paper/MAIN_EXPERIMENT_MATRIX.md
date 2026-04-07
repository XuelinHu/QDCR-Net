# Main Experiment Matrix

This file freezes the first full experiment wave for the paper. It is intentionally limited to
the main comparison only:

- models: `BaselineDetector`, `QDCRNet`
- datasets: `UOv2`, `Brackish`, `RUOD`
- training mode: full-data training with early stopping
- execution status: `not started`

## Matrix

| Dataset | Model | Config | Output Dir | Status |
| --- | --- | --- | --- | --- |
| UOv2 | BaselineDetector | [base_uov2_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_uov2_full.yaml) | `outputs/checkpoints/baseline_uov2_full` | pending |
| UOv2 | QDCRNet | [qdcr_uov2_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_uov2_full.yaml) | `outputs/checkpoints/qdcr_uov2_full` | pending |
| Brackish | BaselineDetector | [base_brackish_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_brackish_full.yaml) | `outputs/checkpoints/baseline_brackish_full` | pending |
| Brackish | QDCRNet | [qdcr_brackish_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_brackish_full.yaml) | `outputs/checkpoints/qdcr_brackish_full` | pending |
| RUOD | BaselineDetector | [base_ruod_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/base_ruod_full.yaml) | `outputs/checkpoints/baseline_ruod_full` | pending |
| RUOD | QDCRNet | [qdcr_ruod_full.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_ruod_full.yaml) | `outputs/checkpoints/qdcr_ruod_full` | pending |

## Fixed Training Policy

- `epochs`: `100`
- `batch_size`: `4`
- `max_batches_per_epoch`: `0` for full-train pass
- `eval.max_batches`: `0` for full-val pass
- `early stopping`: enabled
- `monitor`: validation `loss`
- `patience`: `10`
- `best checkpoint`: `best.pt`

## Why This Matrix

This first-wave matrix is the minimum paper-comparison block that can answer the primary question:

`Does QDCRNet outperform the single-branch baseline on each selected dataset?`

It does not include:

- ablations;
- multi-seed repeats;
- robustness analysis;
- sensitivity analysis.

Those should only be launched after this matrix finishes and the mAP metrics become informative.
