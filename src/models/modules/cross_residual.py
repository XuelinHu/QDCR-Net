import torch
from torch import nn


class CrossResidualBlock(nn.Module):
    """Lightweight bidirectional residual mixing for branch features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.raw_to_enhanced = nn.Linear(in_channels, out_channels)
        self.enhanced_to_raw = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        raw_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mixed_raw = self.activation(raw_feature + self.enhanced_to_raw(enhanced_feature))
        mixed_enhanced = self.activation(
            enhanced_feature + self.raw_to_enhanced(raw_feature)
        )
        return mixed_raw, mixed_enhanced
