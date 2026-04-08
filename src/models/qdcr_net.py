# -*- coding: utf-8 -*-
"""QDCR-Net 与基线检测器定义。"""

from pathlib import Path

import torch
from torch import nn

from .modules import CrossResidualBlock, QualityAwareFusion


class _ConvBranch(nn.Module):
    """共享卷积分支，把输入图像编码为紧凑的全局特征。"""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """输出展平后的单向量特征，供后续 query 头使用。"""
        return self.layers(image).flatten(1)


class _DetectionHead(nn.Module):
    """共享检测头，同时输出类别 logits 与归一化边界框。"""

    def __init__(self, feature_dim: int, num_detection_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_detection_classes),
        )
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, query_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """每个 query 都独立预测一个类别和一个边界框。"""
        return self.classifier(query_features), self.box_head(query_features)


class QDCRNet(nn.Module):
    """最小化实现的 QDCR 多 query 检测器。"""

    def __init__(self, num_classes: int, feature_dim: int = 32, num_queries: int = 8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_detection_classes = num_classes + 1

        self.raw_branch = _ConvBranch(out_channels=feature_dim)
        self.enhanced_branch = _ConvBranch(out_channels=feature_dim)
        self.cross_residual = CrossResidualBlock(in_channels=feature_dim, out_channels=feature_dim)
        self.fusion = QualityAwareFusion(channels=feature_dim)
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        self.head = _DetectionHead(feature_dim, self.num_detection_classes)

    def forward(self, raw_image: torch.Tensor, enhanced_image: torch.Tensor) -> dict[str, torch.Tensor]:
        # 两个分支分别提取特征，再通过交叉残差和质量感知模块融合。
        raw_feature = self.raw_branch(raw_image)
        enhanced_feature = self.enhanced_branch(enhanced_image)
        raw_feature, enhanced_feature = self.cross_residual(raw_feature, enhanced_feature)
        fused_feature = self.fusion(raw_feature, enhanced_feature)
        # 将全局融合特征复制到每个 query 上，形成固定数量的候选检测槽位。
        query_features = fused_feature.unsqueeze(1) + self.query_embed.weight.unsqueeze(0)
        logits, pred_boxes = self.head(query_features)
        return {
            "detections": logits,
            "fused_feature": fused_feature,
            "logits": logits,
            "pred_boxes": pred_boxes,
        }

    def save_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer | None = None,
        metadata: dict | None = None,
    ) -> None:
        """保存模型、优化器和附加元数据。"""
        payload = {
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metadata": metadata or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict:
        """加载 checkpoint，并在需要时恢复优化器状态。"""
        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload.get("metadata", {})


class BaselineDetector(nn.Module):
    """单分支基线检测器，接口与 QDCR-Net 保持一致。"""

    def __init__(self, num_classes: int, feature_dim: int = 32, num_queries: int = 8) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_detection_classes = num_classes + 1

        self.branch = _ConvBranch(out_channels=feature_dim)
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        self.head = _DetectionHead(feature_dim, self.num_detection_classes)

    def forward(self, raw_image: torch.Tensor, enhanced_image: torch.Tensor) -> dict[str, torch.Tensor]:
        # 基线模型不使用增强图，只保留相同的输出协议，便于统一训练和评估流程。
        del enhanced_image
        feature = self.branch(raw_image)
        query_features = feature.unsqueeze(1) + self.query_embed.weight.unsqueeze(0)
        logits, pred_boxes = self.head(query_features)
        return {
            "detections": logits,
            "fused_feature": feature,
            "logits": logits,
            "pred_boxes": pred_boxes,
        }

    def save_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer | None = None,
        metadata: dict | None = None,
    ) -> None:
        """保存基线模型 checkpoint。"""
        payload = {
            "model_state": self.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "metadata": metadata or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load_checkpoint(
        self,
        path: Path,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> dict:
        """加载基线模型 checkpoint。"""
        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload.get("metadata", {})
