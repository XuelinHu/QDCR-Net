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
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """同时输出空间特征图和全局特征，兼顾定位与全局语义。"""
        feature_map = self.encoder(image)
        spatial_feature = self.spatial_pool(feature_map)
        global_feature = self.global_pool(feature_map).flatten(1)
        return spatial_feature, global_feature


class _DetectionHead(nn.Module):
    """基于空间 token 与 query 的检测头，同时输出类别 logits 与归一化边界框。"""

    def __init__(self, feature_dim: int, num_detection_classes: int) -> None:
        super().__init__()
        self.token_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_detection_classes),
        )
        self.box_size_scale = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),
            nn.Sigmoid(),
        )
        self.box_center_offset = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        spatial_feature: torch.Tensor,
        global_feature: torch.Tensor,
        query_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """通过 query 在空间特征图上聚合局部证据，避免退化成固定模板框。"""
        batch_size, _, height, width = spatial_feature.shape
        tokens = self.token_proj(spatial_feature).flatten(2).transpose(1, 2)
        query = self.query_proj(query_embedding).unsqueeze(0).expand(batch_size, -1, -1)
        query = self.query_norm(query + global_feature.unsqueeze(1))
        attention = torch.matmul(query, tokens.transpose(1, 2)) / (tokens.size(-1) ** 0.5)
        attention = attention.softmax(dim=-1)
        query_features = torch.matmul(attention, tokens) + global_feature.unsqueeze(1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=height, device=spatial_feature.device, dtype=spatial_feature.dtype),
            torch.linspace(0.0, 1.0, steps=width, device=spatial_feature.device, dtype=spatial_feature.dtype),
            indexing="ij",
        )
        spatial_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(height * width, 2)
        coords = spatial_coords.unsqueeze(0).expand(batch_size, -1, -1)
        centers = torch.matmul(attention, coords)
        center_offset = self.box_center_offset(query_features) / max(height, width)
        centers = (centers + center_offset).clamp(0.0, 1.0)
        centered_coords = coords.unsqueeze(1) - centers.unsqueeze(2)
        variances = torch.sum(attention.unsqueeze(-1) * centered_coords.pow(2), dim=2).clamp(min=1e-6)
        base_sizes = 2.0 * torch.sqrt(variances)
        size_scale = 0.5 + self.box_size_scale(query_features)
        sizes = (base_sizes * size_scale).clamp(0.02, 1.0)
        pred_boxes = torch.cat([centers, sizes], dim=-1)
        return self.classifier(query_features), pred_boxes


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
        raw_map, raw_feature = self.raw_branch(raw_image)
        enhanced_map, enhanced_feature = self.enhanced_branch(enhanced_image)
        raw_feature, enhanced_feature = self.cross_residual(raw_feature, enhanced_feature)
        fused_feature = self.fusion(raw_feature, enhanced_feature)
        fusion_gate = torch.sigmoid(fused_feature).unsqueeze(-1).unsqueeze(-1)
        fused_map = fusion_gate * enhanced_map + (1.0 - fusion_gate) * raw_map
        logits, pred_boxes = self.head(fused_map, fused_feature, self.query_embed.weight)
        return {
            "detections": logits,
            "fused_feature": fused_feature,
            "fused_map": fused_map,
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
        spatial_feature, feature = self.branch(raw_image)
        logits, pred_boxes = self.head(spatial_feature, feature, self.query_embed.weight)
        return {
            "detections": logits,
            "fused_feature": feature,
            "fused_map": spatial_feature,
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
