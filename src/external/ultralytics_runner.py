# -*- coding: utf-8 -*-
"""Ultralytics 模型适配：YOLOv8 / RT-DETR / 增强后检测。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import yaml
from ultralytics import RTDETR, YOLO

from src.utils import TensorBoardLogger, get_logger


def _dataset_root_from_image_root(image_root: str | Path) -> Path:
    image_path = Path(image_root)
    if image_path.name == "images":
        return image_path.parent.parent
    return image_path.parent


@dataclass
class UltralyticsRunner:
    """对 Ultralytics 训练/评估接口做项目内封装。"""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.logger = get_logger("qdcr.ultralytics")
        self.experiment_config = self.config.get("experiment", {})
        self.dataset_config = self.config.get("dataset", {})
        self.model_config = self.config.get("model", {})
        self.train_config = self.config.get("train", {})
        self.eval_config = self.config.get("eval", {})
        self.output_dir = Path(self.experiment_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = TensorBoardLogger(self.config).run_dir
        self.best_checkpoint_path = self.output_dir / "weights" / "best.pt"
        self.metrics_path = self.output_dir / "metrics.json"
        self.data_yaml_path = self._prepare_data_yaml()

    def train(self) -> dict[str, float]:
        model = self._build_model()
        self.logger.info("[ultralytics.train] model=%s data=%s output=%s", self.model_config["name"], self.data_yaml_path, self.output_dir)
        model.train(
            data=str(self.data_yaml_path),
            epochs=int(self.train_config.get("epochs", 100)),
            batch=int(self.train_config.get("batch_size", 4)),
            imgsz=int(self.dataset_config.get("image_size", 640)),
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True,
            device=str(self.experiment_config.get("device", "cuda")),
            workers=int(self.dataset_config.get("workers", 4)),
            patience=int(self.train_config.get("early_stopping", {}).get("patience", 10)),
            lr0=float(self.train_config.get("lr", 1e-3)),
        )
        return self.evaluate()

    def evaluate(self) -> dict[str, float]:
        model = self._build_model(weights_path=self._resolve_weights_path())
        split = str(self.eval_config.get("split", "val"))
        results = model.val(
            data=str(self.data_yaml_path),
            split=split,
            batch=int(self.train_config.get("batch_size", 4)),
            imgsz=int(self.dataset_config.get("image_size", 640)),
            device=str(self.experiment_config.get("device", "cuda")),
            conf=float(self.eval_config.get("conf_thresh", 0.25)),
            iou=float(self.eval_config.get("iou_thresh", 0.5)),
            project=str(self.output_dir.parent),
            name=f"{self.output_dir.name}_{split}_eval",
            exist_ok=True,
        )
        metrics = self._extract_metrics(results)
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def _build_model(self, weights_path: Path | None = None) -> Any:
        model_name = str(self.model_config.get("name", "yolov8"))
        weights = str(weights_path or self.model_config.get("weights", "yolov8n.pt"))
        if "rtdetr" in model_name.lower():
            return RTDETR(weights)
        return YOLO(weights)

    def _resolve_weights_path(self) -> Path | None:
        if self.best_checkpoint_path.exists():
            return self.best_checkpoint_path
        return None

    def _prepare_data_yaml(self) -> Path:
        if self.model_config.get("name") == "underwater_enhance_yolov8":
            return self._prepare_enhanced_dataset_yaml()

        dataset_root = _dataset_root_from_image_root(self.dataset_config["train_root"])
        data_yaml = dataset_root / "data.yaml"
        if data_yaml.exists():
            return data_yaml

        generated_path = self.output_dir / "data.yaml"
        payload = {
            "path": str(dataset_root.resolve()),
            "train": str(Path(self.dataset_config["train_root"]).resolve()),
            "val": str(Path(self.dataset_config["val_root"]).resolve()),
            "nc": int(self.dataset_config["num_classes"]),
            "names": self.dataset_config["class_names"],
        }
        generated_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return generated_path

    def _prepare_enhanced_dataset_yaml(self) -> Path:
        dataset_root = _dataset_root_from_image_root(self.dataset_config["train_root"])
        cache_root = self.output_dir / "enhanced_dataset"
        for split in ("train", "valid", "test"):
            src_split = dataset_root / split
            if not src_split.exists():
                continue
            image_dir = src_split / "images"
            label_dir = src_split / "labels"
            out_image_dir = cache_root / split / "images"
            out_label_dir = cache_root / split / "labels"
            out_image_dir.mkdir(parents=True, exist_ok=True)
            out_label_dir.mkdir(parents=True, exist_ok=True)
            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file():
                    continue
                out_image_path = out_image_dir / image_path.name
                if not out_image_path.exists():
                    image = cv2.imread(str(image_path))
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced_l = clahe.apply(l)
                    merged = cv2.merge((enhanced_l, a, b))
                    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                    cv2.imwrite(str(out_image_path), enhanced)
                label_path = label_dir / f"{image_path.stem}.txt"
                if label_path.exists():
                    target_label = out_label_dir / label_path.name
                    if not target_label.exists():
                        target_label.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")

        data_yaml_path = cache_root / "data.yaml"
        payload = {
            "path": str(cache_root.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images" if (cache_root / "test" / "images").exists() else "valid/images",
            "nc": int(self.dataset_config["num_classes"]),
            "names": self.dataset_config["class_names"],
        }
        data_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return data_yaml_path

    def _extract_metrics(self, results: Any) -> dict[str, float]:
        summary = getattr(results, "results_dict", {}) or {}
        box = getattr(results, "box", None)
        metrics = {
            "loss": float(summary.get("val/box_loss", 0.0) + summary.get("val/cls_loss", 0.0) + summary.get("val/dfl_loss", 0.0)),
            "acc": 0.0,
            "box_iou": 0.0,
            "map50": float(summary.get("metrics/mAP50(B)", 0.0)),
            "map50_95": float(summary.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(summary.get("metrics/precision(B)", 0.0)),
            "recall": float(summary.get("metrics/recall(B)", 0.0)),
            "params_m": 0.0,
            "gflops": 0.0,
            "fps": 0.0,
        }
        if box is not None:
            metrics["map50"] = float(getattr(box, "map50", metrics["map50"]))
            metrics["map50_95"] = float(getattr(box, "map", metrics["map50_95"]))
            metrics["precision"] = float(getattr(box, "mp", metrics["precision"]))
            metrics["recall"] = float(getattr(box, "mr", metrics["recall"]))
        return metrics

    def notify_target(self) -> Any:
        return SimpleNamespace(run_dir=self.run_dir, best_checkpoint_path=self.best_checkpoint_path)
