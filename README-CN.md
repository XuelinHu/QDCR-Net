# QDCR-Net 中文说明

QDCR-Net 是一个面向论文实验的水下目标检测项目骨架，当前仓库包含轻量化
PyTorch 训练/评估流程、QDCR-Net 双分支模型、单分支基线模型，以及实验配置、
数据目录、日志目录和论文写作目录。

英文版说明见 [README.md](README.md)。

## 编码与注释约定

- `src/` 和 `scripts/` 下的所有 Python 文件都已补充中文注释。
- Python 文件顶部统一使用 `# -*- coding: utf-8 -*-`。
- Shell 脚本中的中文注释也按 UTF-8 保存。
- 注释重点覆盖模块职责、训练流程、数据集回退逻辑、匹配评估逻辑和自动化脚本行为。

## 常用运行命令

```bash
conda env create -f environment.linux.yml
conda activate qdcr-net

conda run -n yolo python scripts/train.py --config configs/qdcr_net.yaml
conda run -n yolo python scripts/eval.py --config configs/qdcr_net.yaml
conda run -n yolo python scripts/eval_map.py --config configs/qdcr_net.yaml

bash scripts/run_download_prepare.sh
bash scripts/run_formal_matrix.sh
bash scripts/continue_overnight.sh
```

## 默认数据集路径

当前默认配置 `configs/qdcr_net.yaml` 和 `configs/base.yaml` 使用下面这些数据路径：

- `data/datasets/URPC2018/train/images`
- `data/datasets/URPC2018/train/labels`
- `data/datasets/URPC2018/val/images`
- `data/datasets/URPC2018/val/labels`
- `data/datasets/URPC2018_enhanced`

脚本涉及的下载包和解压目录路径：

- `data/raw/Brackish.zip`
- `data/raw/RUOD.zip`
- `data/raw/UOv2.zip`
- `data/datasets/downloads/Brackish`
- `data/datasets/downloads/RUOD`
- `data/datasets/downloads/UOv2`

`configs/formal/*.yaml` 中正式全量实验使用的数据路径：

- `data/datasets/downloads/UOv2/train/images`
- `data/datasets/downloads/UOv2/valid/images`
- `data/datasets/downloads/Brackish/train/images`
- `data/datasets/downloads/Brackish/valid/images`
- `data/datasets/downloads/RUOD/RUOD/train/images`
- `data/datasets/downloads/RUOD/RUOD/valid/images`

## 默认快照与输出路径

`qdcr_net.yaml` 默认实验名为 `qdcr_net_smoke`，对应输出路径：

- `outputs/checkpoints/qdcr_net_smoke/latest.pt`
- `outputs/checkpoints/qdcr_net_smoke/best.pt`
- `outputs/checkpoints/qdcr_net_smoke/predictions.json`
- `outputs/checkpoints/qdcr_net_smoke/metrics.json`
- `runs/qdcr_net_smoke/scalars.tsv`

`base.yaml` 默认实验名为 `baseline_detector`，对应输出路径：

- `outputs/checkpoints/baseline_detector/latest.pt`
- `outputs/checkpoints/baseline_detector/best.pt`
- `outputs/checkpoints/baseline_detector/predictions.json`
- `outputs/checkpoints/baseline_detector/metrics.json`
- `runs/baseline_detector/scalars.tsv`

自动化、可视化和实验记录路径：

- `outputs/automation/`
- `outputs/visualizations/`
- `data/logs/`
- `data/results/`

正式全量实验的快照路径：

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

通用快照/产物命名规则：

- `outputs/checkpoints/<experiment>/latest.pt`
- `outputs/checkpoints/<experiment>/best.pt`
- `outputs/checkpoints/<experiment>/predictions.json`
- `outputs/checkpoints/<experiment>/metrics.json`
- `runs/<experiment>/scalars.tsv`
- `outputs/automation/<task_name>.log`

## 仓库文件路径索引

### 根目录文件

- `README.md`
- `README-CN.md`
- `environment.linux.yml`

### 运行入口文件

- `scripts/train.py`：训练入口
- `scripts/eval.py`：评估入口
- `scripts/eval_map.py`：评估并打印 mAP 指标
- `scripts/prepare_dataset.py`：检查数据集索引与回退模式
- `scripts/generate_sample_config.py`：从数据集目录生成样例配置
- `scripts/run_download_prepare.sh`：下载公开数据集并生成配置
- `scripts/run_formal_matrix.sh`：顺序执行正式实验矩阵
- `scripts/continue_overnight.sh`：夜间续跑样例实验

### 配置文件路径

- `configs/qdcr_net.yaml`
- `configs/base.yaml`
- `configs/formal/README.md`
- `configs/formal/base_uov2_full.yaml`
- `configs/formal/qdcr_uov2_full.yaml`
- `configs/formal/base_brackish_full.yaml`
- `configs/formal/qdcr_brackish_full.yaml`
- `configs/formal/base_ruod_full.yaml`
- `configs/formal/qdcr_ruod_full.yaml`

脚本自动生成的样例配置输出到：

- `configs/generated/qdcr_brackish_sample.yaml`
- `configs/generated/base_brackish_sample.yaml`
- `configs/generated/qdcr_ruod_sample.yaml`
- `configs/generated/base_ruod_sample.yaml`

### 源码路径

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

### 数据目录路径

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

### 快照、日志与输出路径

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

### 论文与实验文档路径

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

## 路径使用说明

- 训练器会从所选 YAML 配置里读取 `dataset.train_root`、`dataset.val_root` 和 `experiment.output_dir`。
- 如果默认数据路径下没有真实数据，`src/datasets/underwater_detection.py` 会自动退回到合成样本。
- checkpoint 快照写入 `outputs/checkpoints/<experiment>/`。
- TensorBoard 日志统一写入当前仓库下的 `runs/`，每次启动会自动创建独立实验子目录，目录内同时保留 `scalars.tsv` 和 event 文件。
- 正式全量实验统一使用 `configs/formal/` 下的配置文件。

## 建议阅读顺序

1. `scripts/train.py`
2. `scripts/eval_map.py`
3. `src/engine/trainer.py`
4. `src/datasets/underwater_detection.py`
5. `src/engine/detection_ops.py`
6. `src/models/qdcr_net.py`
