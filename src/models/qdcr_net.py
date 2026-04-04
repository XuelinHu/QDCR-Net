from pathlib import Path

import torch
from torch import nn

from .modules import CrossResidualBlock, QualityAwareFusion


class _ConvBranch(nn.Module):
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
        return self.layers(image).flatten(1)


class _DetectionHead(nn.Module):
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
        return self.classifier(query_features), self.box_head(query_features)


class QDCRNet(nn.Module):
    """Minimal multi-query QDCR detector with detection heads."""

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
        raw_feature = self.raw_branch(raw_image)
        enhanced_feature = self.enhanced_branch(enhanced_image)
        raw_feature, enhanced_feature = self.cross_residual(raw_feature, enhanced_feature)
        fused_feature = self.fusion(raw_feature, enhanced_feature)
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
        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload.get("metadata", {})


class BaselineDetector(nn.Module):
    """Single-branch baseline detector sharing the same detection interface."""

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
        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["model_state"], strict=strict)
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload.get("metadata", {})
