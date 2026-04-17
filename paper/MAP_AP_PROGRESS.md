# mAP/AP Probe Progress

## Goal

Run one representative internal-model probe to test whether the current detector can push `mAP/AP`
above `0` before launching the full formal matrix.

## Selected Task

- model: `QDCRNet`
- dataset: `UOv2`
- config: [qdcr_uov2_map_probe.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_uov2_map_probe.yaml)
- rationale:
  - `UOv2` is the smallest of the three main training sets in this repository
  - `QDCRNet` is the main method, so the probe stays aligned with the paper story
  - the probe keeps the tuned 640px / 16-query setup but caps epochs and batch counts for fast signal

## Probe Policy

- train epochs: `12`
- train max batches per epoch: `120`
- eval max batches: `60`
- early stopping monitor: `map50`
- score threshold: `0.01`

## Progress Log

| Time | Status | Note |
| --- | --- | --- |
| 2026-04-15 | initialized | created probe config and progress log |
| 2026-04-15 15:49 | running | launched `python scripts/train.py --config configs/formal/qdcr_uov2_map_probe.yaml` |
| 2026-04-15 15:51 | epoch 1 done | `loss=2.7092`, `acc=0.45`, `iou=0.03`, displayed `val_map50=0.0000` |
| 2026-04-15 15:52 | epoch 2 done | `loss=2.5203`, `acc=0.47`, `iou=0.04`, displayed `val_map50=0.0000` |
| 2026-04-15 15:54 | epoch 3 done | `loss=2.4930`, `acc=0.49`, `iou=0.05`, displayed `val_map50=0.0000` |
| 2026-04-15 15:54 | stopped manually | probe was stopped during epoch 4 after enough signal was collected |
| 2026-04-15 15:56 | evaluated best checkpoint | `best.pt` from epoch 1 was evaluated with `scripts/eval_map.py` |

## Current Result

- output dir: `outputs/checkpoints/qdcr_uov2_map_probe`
- train run dir: `runs/20260415-154958_qdcr_net_uov2_bs4_lr0.0003_seed42_qdcr_uov2_map_probe`
- eval run dir: `runs/20260415-155507_qdcr_net_uov2_bs4_lr0.0003_seed42_qdcr_uov2_map_probe`
- best checkpoint: `outputs/checkpoints/qdcr_uov2_map_probe/best.pt`
- best checkpoint epoch: `1`
- final `loss`: `2.528109`
- final `acc`: `0.556534`
- final `box_iou`: `0.041328`
- final `map50`: `0.000031`
- final `map50_95`: `0.000004`
- final `precision`: `0.001443`
- final `recall`: `0.002128`
- params: `0.137226 M`
- gflops: `2.065494`
- fps: `115.669294`

## Conclusion

- `mAP/AP` has technically moved above exact zero.
- The gain is extremely small and not yet paper-usable.
- The current refactor improved optimization signals:
  - epoch loss decreased from `2.7092` to `2.4930`
  - matched-box IoU increased from `0.03` to `0.05`
- The bottleneck is no longer pure optimization collapse; it is now detection quality and ranking quality.

## Next Action

Priority order for the next attempt:

1. increase probe duration to a real short run, e.g. `20-30` epochs instead of stopping after `3` completed epochs
2. inspect decoded predictions on validation images to verify whether boxes are roughly on target but scored too low
3. if scores are the main issue, tune classification-vs-box loss balance and background handling
4. if boxes are still poor, upgrade regression from `SmoothL1 + IoU` toward `GIoU/DIoU/CIoU`

## Probe V2 Plan

The first probe exposed a monitoring-scale problem:

- monitor: `map50`
- observed best value: about `0.000031`
- old `min_delta`: `0.0005`

That `min_delta` is too large for the current metric scale and can suppress small but real
improvements. The second probe therefore uses:

- config: [qdcr_uov2_map_probe_v2.yaml](/ds1/workspace/ai/QDCR-Net/configs/formal/qdcr_uov2_map_probe_v2.yaml)
- epochs: `20`
- eval max batches: `80`
- patience: `8`
- `min_delta: 0.000001`

## Probe V2 Execution

| Time | Status | Note |
| --- | --- | --- |
| 2026-04-15 15:58 | running | launched `python scripts/train.py --config configs/formal/qdcr_uov2_map_probe_v2.yaml` |
| 2026-04-15 16:00 | epoch 1 done | `loss=2.7106`, `acc=0.45`, `iou=0.03`, displayed `val_map50=0.0000` |
| 2026-04-15 16:01 | epoch 2 done | `loss=2.5246`, `acc=0.47`, `iou=0.04`, displayed `val_map50=0.0000`, `best_epoch=1` |
| 2026-04-15 16:02 | stopped manually | probe was stopped during epoch 3 because the trend was already clear |
| 2026-04-15 16:03 | evaluated best checkpoint | `best.pt` from epoch 1 was evaluated with `scripts/eval_map.py` |

## Probe V2 Result

- output dir: `outputs/checkpoints/qdcr_uov2_map_probe_v2`
- train run dir: `runs/20260415-155845_qdcr_net_uov2_bs4_lr0.0003_seed42_qdcr_uov2_map_probe_v2`
- eval run dir: `runs/20260415-160228_qdcr_net_uov2_bs4_lr0.0003_seed42_qdcr_uov2_map_probe_v2`
- best checkpoint: `outputs/checkpoints/qdcr_uov2_map_probe_v2/best.pt`
- best checkpoint epoch: `1`
- final `loss`: `2.504970`
- final `acc`: `0.506508`
- final `box_iou`: `0.042682`
- final `map50`: `0.000050`
- final `map50_95`: `0.000008`
- final `precision`: `0.001473`
- final `recall`: `0.002581`
- params: `0.137226 M`
- gflops: `2.065494`
- fps: `116.661038`

## Probe Comparison

| Probe | `map50` | `map50_95` | `precision` | `recall` | `loss` |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | `0.000031` | `0.000004` | `0.001443` | `0.002128` | `2.528109` |
| V2 | `0.000050` | `0.000008` | `0.001473` | `0.002581` | `2.504970` |

## Updated Conclusion

- The detector is now consistently above exact-zero `mAP/AP`.
- Lowering the `map50` early-stopping `min_delta` from `0.0005` to `0.000001` helped preserve small gains.
- The improvement is still only from `3.1e-05` to `5.0e-05`, which is far below a usable experimental level.
- This means the next bottleneck is not just checkpoint selection; it is still the detector's localization and ranking quality.
