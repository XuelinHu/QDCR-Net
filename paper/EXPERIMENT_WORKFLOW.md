# QDCR-Net Paper Experiment Workflow

This document is the single source of truth for turning the current codebase into a paper-ready
experiment package. It is organized by the final writing goal of the paper rather than by scripts
or folders.

## 1. Final Paper Goal

The paper needs to prove three claims:

1. `QDCR-Net` improves underwater small-object detection accuracy compared with a lightweight
   baseline.
2. The gains come from the proposed components, not only from extra parameters or extra input
   branches.
3. The method remains efficient enough for practical deployment.

To support these claims, the final paper must contain:

- a main comparison table on multiple datasets;
- an ablation table tied to the proposed modules;
- a complexity table with `Params`, `GFLOPs`, and `FPS`;
- qualitative visualizations and failure cases;
- a short robustness or sensitivity analysis.

## 2. Current Code Status

The codebase already supports:

- PyTorch training and evaluation through [scripts/train.py](/ds1/workspace/ai/QDCR-Net/scripts/train.py),
  [scripts/eval.py](/ds1/workspace/ai/QDCR-Net/scripts/eval.py), and
  [scripts/eval_map.py](/ds1/workspace/ai/QDCR-Net/scripts/eval_map.py)
- YOLO-style dataset loading in
  [underwater_detection.py](/ds1/workspace/ai/QDCR-Net/src/datasets/underwater_detection.py)
- fixed-query detection training with matching, NMS, and mAP-style metrics in
  [trainer.py](/ds1/workspace/ai/QDCR-Net/src/engine/trainer.py) and
  [detection_ops.py](/ds1/workspace/ai/QDCR-Net/src/engine/detection_ops.py)
- baseline and QDCR model variants in
  [qdcr_net.py](/ds1/workspace/ai/QDCR-Net/src/models/qdcr_net.py)
- TensorBoard-compatible logging under `runs/`
- early stopping and `best.pt` checkpoint saving

The codebase does not yet provide a full paper-level experiment matrix. That part still needs to
be executed and recorded.

## 3. Dataset Status and Paper Alignment

### 3.1 Locally Prepared Datasets

The following datasets are already present locally and can be trained immediately:

- `UOv2`
- `Brackish`
- `RUOD`

Observed local splits:

- `UOv2`: train `5320`, valid `1520`, test `760`
- `Brackish`: train `11739`, valid `1467`, test `1468`
- `RUOD`: train `9800`, valid `4200`, test `300` split into `testb/testc/testl`

### 3.2 Frozen First-Wave Dataset Choice

The first full experiment wave is now frozen to the datasets already prepared locally:

- `UOv2`
- `Brackish`
- `RUOD`

This means the manuscript must be updated later so that the dataset section, result tables, and
released configs all match this dataset list.

## 4. Mapping Paper Sections to Required Experiments

### 4.1 Method Section -> What Must Be Proven

The method section currently claims:

- dual-branch feature extraction;
- cross-residual interaction;
- quality-aware dynamic fusion;
- small-object-oriented neck.

Each claimed module needs direct experimental evidence:

- `dual branch` -> compare against single-branch baseline;
- `cross residual` -> compare against dual branch without cross residual;
- `quality-aware fusion` -> compare against fixed fusion or direct add/concat fusion;
- `small-object neck` -> compare against a version without the neck;
- `full model` -> show final combination is better than partial variants.

### 4.2 Experiments Section -> Tables and Figures That Must Exist

The final experiment section should contain these blocks:

1. `Datasets`
   Report dataset sources, split sizes, class counts, and preprocessing format.

2. `Implementation Details`
   Report environment, training schedule, early stopping policy, optimizer, image size, batch size,
   and checkpoint selection rule.

3. `Main Comparison`
   Compare `Baseline` and `QDCR-Net` on every selected dataset using:
   - `mAP@0.5`
   - `mAP@0.5:0.95`
   - `Precision`
   - `Recall`
   - `Params`
   - `GFLOPs`
   - `FPS`

4. `Ablation`
   Compare intermediate variants aligned with the method claims.

5. `Visualization`
   Show successful detections, hard failures, and small-object cases.

6. `Robustness / Sensitivity`
   Show at least one compact analysis beyond the main comparison.

## 5. Experiment Matrix Required for a Paper-Ready Result

### 5.1 Mandatory Main Experiments

For every final paper dataset, run:

- `BaselineDetector`
- `QDCRNet`

This is the minimum main-comparison matrix.

For the currently frozen first-wave dataset list, this is:

- `3 datasets x 2 models = 6 full runs`

Suggested reporting checkpoint:

- choose `best.pt` by validation `loss` now;
- later consider switching the monitor to `map50` once the detector becomes stable.

### 5.2 Mandatory Ablations

At minimum, implement and run these variants:

1. `BaselineDetector`
2. `Dual branch without cross residual`
3. `Dual branch with cross residual only`
4. `Dual branch + cross residual + fixed fusion`
5. `Full QDCR-Net`

Optional but recommended:

6. `Enhanced-only detector`
7. `Full QDCR-Net without small-object neck`

Recommended dataset scope for ablation:

- run on `RUOD` first because it is larger and more diverse;
- then confirm the main conclusion on one additional dataset.

### 5.3 Mandatory Complexity Report

Every main result table should include:

- `Params`
- `GFLOPs`
- `FPS`

These are already computed by the current evaluation pipeline.

### 5.4 Mandatory Visualization

Prepare:

- 2 to 4 representative success cases;
- 2 to 4 hard cases or failure cases;
- at least one small-object-focused comparison figure.

## 6. Optional But Strongly Recommended Experiments

### 6.1 Repeatability

For the final main comparison, rerun with multiple seeds:

- recommended: `3 seeds`

Report:

- mean;
- standard deviation.

### 6.2 Sensitivity Analysis

Keep this compact. Do not run a large grid before the main experiments are stable.

Recommended parameters:

- learning rate;
- image size;
- number of queries.

### 6.3 Robustness Analysis

Recommended perturbations:

- blur;
- low contrast;
- color cast.

This can be done on one representative dataset if time is limited.

## 7. Current Completed Work

Already completed:

- local dataset preparation for `UOv2`, `Brackish`, and `RUOD`;
- sample runs for `Baseline` and `QDCR` on all three prepared datasets;
- evaluation outputs including `loss`, `acc`, `box_iou`, `precision`, `recall`, `map50`,
  `map50_95`, `Params`, `GFLOPs`, and `FPS`;
- early stopping support and `best.pt` checkpoint handling.

Current limitation of the sample runs:

- they are still too short to support paper claims;
- `map50` and `map50_95` are still `0`, which means the detector has not yet reached useful
  localization quality under strict detection evaluation;
- sample runs are suitable for pipeline verification, not for final reporting.

## 8. Immediate Actions Before Running Full Experiments

The following should happen before the first full training wave:

1. `Freeze the first comparison matrix`
   Start with:
   - `Baseline`
   - `QDCR`

2. `Freeze a training schedule`
   Suggested first formal schedule:
   - `epochs: 100`
   - `early stopping: enabled`
   - `patience: 10`
   - full train/val passes rather than sample `max_batches`

3. `Freeze reporting rules`
   Decide:
   - whether final reporting uses `best.pt`;
   - which validation metric chooses the best checkpoint.

4. `Create ablation-ready configs`
   Before large-scale training, expose model switches for:
   - no cross residual;
   - fixed fusion;
   - no small-object neck;
   - enhanced-only.

## 9. Recommended Execution Order

This is the least wasteful order:

1. Run the first full `Baseline vs QDCR` comparison on the frozen first-wave datasets.
2. Check whether `map50` leaves zero and whether the main trend is stable.
3. Only then launch ablations.
4. After the model is stable, run seed repeats and compact sensitivity analysis.
5. Finish visualization and final tables.

## 10. Deliverables Checklist

The paper experiment package is complete only when all of the following exist:

- final dataset list matches the manuscript;
- one main-comparison table per final paper dataset set;
- one ablation table tied to claimed modules;
- one complexity table;
- one visualization figure set;
- one short robustness or sensitivity subsection;
- reproducible configs and output folders for every reported number.

## 11. Recommended Next Step

The next concrete step is:

1. use the formal configs under `configs/formal/`;
2. run the frozen main matrix described in
   [MAIN_EXPERIMENT_MATRIX.md](/ds1/workspace/ai/QDCR-Net/paper/MAIN_EXPERIMENT_MATRIX.md);
3. only after that, promote successful settings into ablation and repeatability studies.
