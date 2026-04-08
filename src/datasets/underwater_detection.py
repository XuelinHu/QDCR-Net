# -*- coding: utf-8 -*-
"""水下目标检测数据集定义。"""

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class UnderwaterDetectionDataset(Dataset):
    """支持真实 YOLO 标注与合成回退样本的数据集。"""

    image_root: Path
    annotation_root: Path | None = None
    enhanced_root: Path | None = None
    num_classes: int = 4
    image_size: int = 128
    synthetic_size: int = 32
    max_objects: int = 8
    _samples: list[dict[str, Any]] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        """初始化路径、图像变换，并优先索引真实数据。"""
        self.image_root = Path(self.image_root)
        self.annotation_root = Path(self.annotation_root) if self.annotation_root else None
        self.enhanced_root = Path(self.enhanced_root) if self.enhanced_root else None
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

        image_files = self._discover_images(self.image_root)
        if image_files:
            for image_path in image_files:
                sample = self._build_real_sample(image_path)
                if sample is not None:
                    self._samples.append(sample)

        if self._samples:
            return

        # 当本地没有可用真实数据时，自动构造合成样本以保证训练链路可跑通。
        self._samples = [self._build_synthetic_sample(index) for index in range(self.synthetic_size)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按索引返回一个样本，统一输出原图、增强图和标注。"""
        sample = self._samples[index]
        if sample["source"] == "real":
            raw_image = self._load_image_tensor(sample["image_path"])
            enhanced_image = (
                self._load_image_tensor(sample["enhanced_path"])
                if sample["enhanced_path"] is not None
                else self._auto_enhance(raw_image)
            )
        else:
            # 合成样本直接动态生成一对原图/增强图，避免额外落盘。
            raw_image, enhanced_image = self._build_synthetic_pair(sample["sample_id"], sample["annotations"])

        gt_boxes = torch.tensor(
            [annotation["bbox"] for annotation in sample["annotations"]],
            dtype=torch.float32,
        )
        gt_classes = torch.tensor(
            [annotation["class_id"] for annotation in sample["annotations"]],
            dtype=torch.long,
        )
        return {
            "sample_id": sample["sample_id"],
            "raw_image": raw_image,
            "enhanced_image": enhanced_image,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
            "source": sample["source"],
        }

    def _discover_images(self, root: Path) -> list[Path]:
        """递归发现数据目录里的常见图像文件。"""
        if not root.exists():
            return []

        image_files: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(sorted(root.rglob(pattern)))
        return image_files

    def _build_real_sample(self, image_path: Path) -> dict[str, Any] | None:
        """根据图像路径组装真实样本，如果无标注则跳过。"""
        annotations = self._load_labels_for_image(image_path)
        if not annotations:
            return None

        # 优先保留面积更大的目标，避免超出 max_objects 时随机截断。
        annotations.sort(key=lambda item: item["bbox"][2] * item["bbox"][3], reverse=True)
        return {
            "sample_id": image_path.relative_to(self.image_root).as_posix(),
            "annotations": annotations[: self.max_objects],
            "image_path": image_path,
            "enhanced_path": self._resolve_enhanced_path(image_path),
            "source": "real",
        }

    def _build_synthetic_sample(self, index: int) -> dict[str, Any]:
        """构造一个可重复生成的合成样本描述。"""
        object_count = 1 + (index % min(3, self.max_objects))
        annotations = []
        for offset in range(object_count):
            label = (index + offset) % self.num_classes
            annotations.append(
                {
                    "class_id": label,
                    "bbox": self._synthetic_bbox(label, offset),
                }
            )
        return {
            "sample_id": f"synthetic-{index}",
            "annotations": annotations,
            "source": "synthetic",
        }

    def _load_labels_for_image(self, image_path: Path) -> list[dict[str, Any]]:
        """读取 YOLO 标签文件，并过滤非法类别或越界框。"""
        label_path = self._resolve_label_path(image_path)
        if label_path is None or not label_path.exists():
            return []

        labels: list[dict[str, Any]] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                bbox = [float(value) for value in parts[1:5]]
            except ValueError:
                continue

            if 0 <= class_id < self.num_classes and all(0.0 <= value <= 1.0 for value in bbox):
                labels.append({"class_id": class_id, "bbox": bbox})

        return labels

    def _resolve_label_path(self, image_path: Path) -> Path | None:
        """尝试从多种常见目录结构推断标签路径。"""
        candidates: list[Path] = []
        if self.annotation_root is not None:
            try:
                candidates.append(
                    self.annotation_root / image_path.relative_to(self.image_root).with_suffix(".txt")
                )
            except ValueError:
                candidates.append(self.annotation_root / f"{image_path.stem}.txt")

        candidates.append(image_path.with_suffix(".txt"))

        parts = list(image_path.parts)
        for marker in ("images", "image", "imgs", "img"):
            if marker in parts:
                marker_index = parts.index(marker)
                # 常见的 YOLO 数据集会把 images 目录平移为 labels 目录。
                remapped = Path(*parts[:marker_index], "labels", *parts[marker_index + 1 :])
                candidates.append(remapped.with_suffix(".txt"))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def _resolve_enhanced_path(self, image_path: Path) -> Path | None:
        """根据增强图根目录推断对应增强图位置。"""
        if self.enhanced_root is None or not self.enhanced_root.exists():
            return None

        candidates: list[Path] = []
        try:
            candidates.append(self.enhanced_root / image_path.relative_to(self.image_root))
        except ValueError:
            candidates.append(self.enhanced_root / image_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_image_tensor(self, path: Path) -> torch.Tensor:
        """读取单张图像并转换为模型输入张量。"""
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def _auto_enhance(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """当缺少增强图时，使用简单亮度/对比度提升做兜底。"""
        return torch.clamp(image_tensor * 1.1 + 0.03, 0.0, 1.0)

    def _build_synthetic_pair(
        self,
        sample_id: str,
        annotations: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据样本 ID 稳定生成一对合成图像。"""
        seed = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        raw = torch.rand((3, self.image_size, self.image_size), generator=generator) * 0.12

        for annotation in annotations:
            x1, y1, x2, y2 = self._xywh_to_xyxy(annotation["bbox"])
            left = max(int(x1 * self.image_size), 0)
            top = max(int(y1 * self.image_size), 0)
            right = min(int(x2 * self.image_size), self.image_size)
            bottom = min(int(y2 * self.image_size), self.image_size)
            channel = int(annotation["class_id"]) % 3
            # 不同类别映射到不同通道，便于让模型看到稳定的类别差异。
            raw[channel, top:bottom, left:right] += 0.75

        raw = torch.clamp(raw, 0.0, 1.0)
        enhanced = torch.clamp(raw * 1.15 + 0.05, 0.0, 1.0)
        return raw, enhanced

    def _synthetic_bbox(self, label: int, slot: int) -> list[float]:
        """生成一个位于合理范围内的归一化边界框。"""
        width = max(0.14, 0.26 - 0.03 * slot)
        height = max(0.12, 0.22 - 0.02 * slot)
        center_x = min(0.18 + 0.18 * label + 0.08 * slot, 0.82)
        center_y = min(0.20 + 0.12 * label + 0.10 * slot, 0.82)
        return [center_x, center_y, width, height]

    def _xywh_to_xyxy(self, bbox: list[float]) -> tuple[float, float, float, float]:
        """把中心点格式边界框转换为左上角/右下角格式。"""
        center_x, center_y, width, height = bbox
        return (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
