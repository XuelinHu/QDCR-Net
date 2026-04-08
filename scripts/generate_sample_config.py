# -*- coding: utf-8 -*-
"""根据数据集目录自动生成样例实验配置。"""

import argparse
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """解析配置生成脚本参数。"""
    parser = argparse.ArgumentParser(description="Generate a sample experiment config from a dataset root.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=8)
    return parser.parse_args()


def find_dir(root: Path, split_names: list[str], leaf_names: set[str]) -> Path | None:
    """在数据集目录里搜索 train/val 对应的图片目录。"""
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        lower = path.as_posix().lower()
        if path.name.lower() not in leaf_names:
            continue
        if any(f"/{split}/" in lower or lower.endswith(f"/{split}") for split in split_names):
            return path
    return None


def main() -> None:
    """读取模板后按数据集结构填充实验配置。"""
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    template = yaml.safe_load(args.template.read_text(encoding="utf-8"))

    data_yaml = dataset_root / "data.yaml"
    names = template["dataset"].get("class_names", [])
    num_classes = template["dataset"].get("num_classes", len(names))
    if data_yaml.exists():
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = list(data.get("names", names))
        num_classes = int(data.get("nc", len(names)))

    train_images = find_dir(dataset_root, ["train"], {"images", "image"})
    val_images = find_dir(dataset_root, ["val", "valid", "validation"], {"images", "image"})
    if train_images is None:
        raise SystemExit(f"Could not find train/images under {dataset_root}")
    if val_images is None:
        # 没有单独验证集时退化为复用训练集，便于快速连通流程。
        val_images = train_images

    cfg = template
    cfg["experiment"]["name"] = args.experiment_name
    cfg["experiment"]["output_dir"] = f"outputs/checkpoints/{args.experiment_name}"
    cfg["experiment"]["runs_dir"] = f"runs/{args.experiment_name}"
    cfg["dataset"]["train_root"] = str(train_images.relative_to(ROOT))
    cfg["dataset"]["val_root"] = str(val_images.relative_to(ROOT))
    cfg["dataset"]["enhanced_root"] = str(train_images.relative_to(ROOT))
    cfg["dataset"]["num_classes"] = num_classes
    cfg["dataset"]["class_names"] = names
    cfg["dataset"]["synthetic_size"] = 0
    cfg["dataset"]["max_objects"] = args.max_objects
    cfg["dataset"]["image_size"] = args.image_size
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["max_batches_per_epoch"] = args.max_batches
    cfg["eval"]["max_batches"] = args.max_batches

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
