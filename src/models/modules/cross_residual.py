# -*- coding: utf-8 -*-
"""跨分支残差交互模块。"""

import torch
from torch import nn


class CrossResidualBlock(nn.Module):
    """对原图分支和增强分支做轻量双向残差融合。"""

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
        """让两个分支互相补充信息，同时保留各自的残差语义。"""
        mixed_raw = self.activation(raw_feature + self.enhanced_to_raw(enhanced_feature))
        mixed_enhanced = self.activation(
            enhanced_feature + self.raw_to_enhanced(raw_feature)
        )
        return mixed_raw, mixed_enhanced
