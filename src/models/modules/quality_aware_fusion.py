import torch
from torch import nn


class QualityAwareFusion(nn.Module):
    """Gated feature fusion between raw and enhanced branches."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raw_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
        quality_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = self.gate(torch.cat([raw_feature, enhanced_feature], dim=1))
        fused = gate * enhanced_feature + (1.0 - gate) * raw_feature
        if quality_vector is not None:
            fused = fused + quality_vector
        return fused
