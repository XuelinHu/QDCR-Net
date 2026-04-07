This directory stores the first-wave full-data experiment configs for the paper.

Scope of the frozen matrix:

- datasets: `UOv2`, `Brackish`, `RUOD`
- models: `BaselineDetector`, `QDCRNet`
- schedule: full train/val passes, early stopping enabled, `best.pt` used for reporting

These configs are intentionally more conservative than a final sweep so they remain runnable
under limited GPU resources.
