# -*- coding: utf-8 -*-
"""Torchvision Faster R-CNN 训练/评估适配。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.datasets import UnderwaterDetectionDataset
from src.engine.detection_ops import compute_detection_metrics
from src.utils import TensorBoardLogger, get_logger


class TorchvisionDetectionDataset(Dataset):
    """把当前数据集包装成 torchvision 检测模型需要的格式。"""

    def __init__(self, dataset: UnderwaterDetectionDataset) -> None:
        self.dataset = dataset
        self.image_size = dataset.image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.dataset[index]
        boxes = sample["gt_boxes"].clone()
        boxes_xyxy = torch.empty_like(boxes)
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * self.image_size
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * self.image_size
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * self.image_size
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * self.image_size
        target = {
            "boxes": boxes_xyxy,
            "labels": sample["gt_classes"] + 1,
            "image_id": torch.tensor([index], dtype=torch.long),
        }
        return sample["raw_image"], target


def _collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


@dataclass
class FasterRCNNRunner:
    """最小 Faster R-CNN 训练/评估控制器。"""

    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.logger = get_logger("qdcr.faster_rcnn")
        self.experiment_config = self.config.get("experiment", {})
        self.dataset_config = self.config.get("dataset", {})
        self.train_config = self.config.get("train", {})
        self.eval_config = self.config.get("eval", {})
        self.output_dir = Path(self.experiment_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "latest.pt"
        self.best_checkpoint_path = self.output_dir / "best.pt"
        self.metrics_path = self.output_dir / "metrics.json"
        self.device = torch.device(self.experiment_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.run_dir = TensorBoardLogger(self.config).run_dir

    def train(self) -> dict[str, float]:
        train_loader, val_loader = self._build_loaders()
        model = self._build_model().to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(self.train_config.get("lr", 0.005)), momentum=0.9, weight_decay=float(self.train_config.get("weight_decay", 0.0005)))
        best_loss = float("inf")
        best_metrics: dict[str, float] = {}
        epochs = int(self.train_config.get("epochs", 20))
        max_batches = int(self.train_config.get("max_batches_per_epoch", 0))
        for epoch in range(epochs):
            model.train()
            for batch_index, (images, targets) in enumerate(train_loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                images = [image.to(self.device) for image in images]
                targets = [{key: value.to(self.device) for key, value in target.items()} for target in targets]
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            metrics = self._evaluate_model(model, val_loader)
            torch.save({"model_state": model.state_dict()}, self.checkpoint_path)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_metrics = metrics
                torch.save({"model_state": model.state_dict()}, self.best_checkpoint_path)
            self.logger.info("[faster_rcnn.train] epoch=%s val_loss=%.4f map50=%.4f", epoch + 1, metrics["loss"], metrics["map50"])
        self.metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        return best_metrics

    def evaluate(self) -> dict[str, float]:
        _, val_loader = self._build_loaders()
        model = self._build_model().to(self.device)
        checkpoint = self.best_checkpoint_path if self.best_checkpoint_path.exists() else self.checkpoint_path
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(payload["model_state"])
        metrics = self._evaluate_model(model, val_loader)
        self.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def _build_model(self) -> nn.Module:
        num_classes = int(self.dataset_config.get("num_classes", 4)) + 1
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _build_loaders(self) -> tuple[DataLoader, DataLoader]:
        common_args = {
            "num_classes": int(self.dataset_config.get("num_classes", 4)),
            "image_size": int(self.dataset_config.get("image_size", 320)),
            "synthetic_size": int(self.dataset_config.get("synthetic_size", 0)),
            "max_objects": int(self.dataset_config.get("max_objects", 8)),
        }
        train_dataset = UnderwaterDetectionDataset(
            image_root=Path(self.dataset_config["train_root"]),
            annotation_root=Path(self.dataset_config["train_root"]).parent.parent / "labels",
            enhanced_root=Path(self.dataset_config.get("enhanced_root", self.dataset_config["train_root"])),
            **common_args,
        )
        val_dataset = UnderwaterDetectionDataset(
            image_root=Path(self.dataset_config["val_root"]),
            annotation_root=Path(self.dataset_config["val_root"]).parent.parent / "labels",
            enhanced_root=Path(self.dataset_config.get("enhanced_root", self.dataset_config["train_root"])),
            **common_args,
        )
        train_loader = DataLoader(TorchvisionDetectionDataset(train_dataset), batch_size=int(self.train_config.get("batch_size", 2)), shuffle=True, num_workers=0, collate_fn=_collate)
        val_loader = DataLoader(TorchvisionDetectionDataset(val_dataset), batch_size=int(self.train_config.get("batch_size", 2)), shuffle=False, num_workers=0, collate_fn=_collate)
        return train_loader, val_loader

    def _evaluate_model(self, model: nn.Module, val_loader: DataLoader) -> dict[str, float]:
        model.eval()
        predictions: list[dict[str, torch.Tensor]] = []
        ground_truths: list[dict[str, torch.Tensor]] = []
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [image.to(self.device) for image in images]
                eval_targets = [{key: value.to(self.device) for key, value in target.items()} for target in targets]
                model.train()
                loss_dict = model(images, eval_targets)
                total_loss += float(sum(loss_dict.values()).item())
                total_batches += 1
                model.eval()
                outputs = model(images)
                for output, target in zip(outputs, targets):
                    boxes = output["boxes"].detach().cpu().clone()
                    boxes[:, 0::2] /= int(self.dataset_config.get("image_size", 320))
                    boxes[:, 1::2] /= int(self.dataset_config.get("image_size", 320))
                    boxes_xywh = torch.empty_like(boxes)
                    boxes_xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
                    boxes_xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
                    boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
                    predictions.append(
                        {
                            "scores": output["scores"].detach().cpu(),
                            "classes": (output["labels"].detach().cpu() - 1).clamp(min=0),
                            "boxes": boxes_xywh,
                        }
                    )
                    gt_boxes_xyxy = target["boxes"].detach().cpu().clone()
                    gt_boxes_xyxy[:, 0::2] /= int(self.dataset_config.get("image_size", 320))
                    gt_boxes_xyxy[:, 1::2] /= int(self.dataset_config.get("image_size", 320))
                    gt_boxes_xywh = torch.empty_like(gt_boxes_xyxy)
                    gt_boxes_xywh[:, 0] = (gt_boxes_xyxy[:, 0] + gt_boxes_xyxy[:, 2]) / 2.0
                    gt_boxes_xywh[:, 1] = (gt_boxes_xyxy[:, 1] + gt_boxes_xyxy[:, 3]) / 2.0
                    gt_boxes_xywh[:, 2] = gt_boxes_xyxy[:, 2] - gt_boxes_xyxy[:, 0]
                    gt_boxes_xywh[:, 3] = gt_boxes_xyxy[:, 3] - gt_boxes_xyxy[:, 1]
                    ground_truths.append({"classes": target["labels"].detach().cpu() - 1, "boxes": gt_boxes_xywh})
        metrics = compute_detection_metrics(predictions, ground_truths, int(self.dataset_config.get("num_classes", 4)))
        metrics.update(
            {
                "loss": total_loss / max(total_batches, 1),
                "acc": 0.0,
                "box_iou": 0.0,
                "params_m": float(sum(p.numel() for p in model.parameters())) / 1_000_000.0,
                "gflops": 0.0,
                "fps": 0.0,
            }
        )
        return metrics

    def notify_target(self) -> Any:
        return SimpleNamespace(run_dir=self.run_dir, best_checkpoint_path=self.best_checkpoint_path)
