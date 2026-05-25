# -*- coding: utf-8 -*-
"""跨分支残差交互模块。"""

import math

import torch
from torch import nn


class CrossResidualBlock(nn.Module):
    """对原图分支和增强分支做可配置的双向残差融合。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = "vanilla",
        threshold_init: float = 0.1,
        threshold_slope: float = 10.0,
        sparse_mode: str = "soft_threshold",
        sparse_lambda_init: float = 0.1,
        sparse_topk_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.raw_to_enhanced = nn.Linear(in_channels, out_channels)
        self.enhanced_to_raw = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.mode = mode.lower()
        self.sparse_mode = sparse_mode.lower()
        self.threshold_slope = threshold_slope
        self.sparse_topk_ratio = sparse_topk_ratio

        valid_modes = {"vanilla", "learnable_threshold", "sparse_residual"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported cross residual mode: {mode}")

        valid_sparse_modes = {"soft_threshold", "topk"}
        if self.sparse_mode not in valid_sparse_modes:
            raise ValueError(f"Unsupported sparse residual mode: {sparse_mode}")

        # Use unconstrained parameters and map them to positive thresholds.
        self.raw_threshold = nn.Parameter(torch.full((out_channels,), self._inverse_softplus(threshold_init)))
        self.enhanced_threshold = nn.Parameter(torch.full((out_channels,), self._inverse_softplus(threshold_init)))
        self.raw_sparse_lambda = nn.Parameter(
            torch.full((out_channels,), self._inverse_softplus(sparse_lambda_init))
        )
        self.enhanced_sparse_lambda = nn.Parameter(
            torch.full((out_channels,), self._inverse_softplus(sparse_lambda_init))
        )

    def forward(
        self,
        raw_feature: torch.Tensor,
        enhanced_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """让两个分支互相补充信息，同时保留各自的残差语义。"""
        residual_to_raw = self.enhanced_to_raw(enhanced_feature)
        residual_to_enhanced = self.raw_to_enhanced(raw_feature)

        residual_to_raw = self._filter_residual(
            residual_to_raw,
            self.raw_threshold,
            self.raw_sparse_lambda,
        )
        residual_to_enhanced = self._filter_residual(
            residual_to_enhanced,
            self.enhanced_threshold,
            self.enhanced_sparse_lambda,
        )

        mixed_raw = self.activation(raw_feature + residual_to_raw)
        mixed_enhanced = self.activation(enhanced_feature + residual_to_enhanced)
        return mixed_raw, mixed_enhanced

    def _filter_residual(
        self,
        residual: torch.Tensor,
        threshold_param: nn.Parameter,
        sparse_lambda_param: nn.Parameter,
    ) -> torch.Tensor:
        if self.mode == "vanilla":
            return residual
        if self.mode == "learnable_threshold":
            return self._apply_learnable_threshold(residual, threshold_param)
        return self._apply_sparse_residual(residual, sparse_lambda_param)

    def _apply_learnable_threshold(self, residual: torch.Tensor, threshold_param: nn.Parameter) -> torch.Tensor:
        threshold = torch.nn.functional.softplus(threshold_param).unsqueeze(0)
        gate = torch.sigmoid(self.threshold_slope * (residual.abs() - threshold))
        return residual * gate

    def _apply_sparse_residual(self, residual: torch.Tensor, sparse_lambda_param: nn.Parameter) -> torch.Tensor:
        if self.sparse_mode == "topk":
            return self._apply_topk_mask(residual)
        sparse_lambda = torch.nn.functional.softplus(sparse_lambda_param).unsqueeze(0)
        return torch.sign(residual) * torch.relu(residual.abs() - sparse_lambda)

    def _apply_topk_mask(self, residual: torch.Tensor) -> torch.Tensor:
        channels = residual.size(1)
        keep_count = max(1, min(channels, math.ceil(channels * self.sparse_topk_ratio)))
        topk_indices = residual.abs().topk(k=keep_count, dim=1).indices
        mask = torch.zeros_like(residual)
        mask.scatter_(1, topk_indices, 1.0)
        return residual * mask

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        if value <= 0:
            return -20.0
        return math.log(math.expm1(value))
