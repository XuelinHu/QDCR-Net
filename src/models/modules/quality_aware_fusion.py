# -*- coding: utf-8 -*-
"""质量感知融合模块。"""

import torch
from torch import nn


class QualityAwareFusion(nn.Module):
    """使用门控权重融合原图分支与增强分支特征。"""

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
        """根据双分支特征生成融合门控，并可选叠加质量向量。"""
        gate = self.gate(torch.cat([raw_feature, enhanced_feature], dim=1))
        fused = gate * enhanced_feature + (1.0 - gate) * raw_feature
        if quality_vector is not None:
            fused = fused + quality_vector
        return fused
