# -*- coding: utf-8 -*-
"""检测任务损失函数。"""

import torch
from torch import nn

from src.engine.detection_ops import pairwise_iou_xywh


class DetectionLoss(nn.Module):
    """分类损失与匹配框回归损失的组合。"""

    def __init__(
        self,
        box_weight: float = 5.0,
        iou_weight: float = 2.0,
        focal_gamma: float = 2.0,
        background_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.box_loss = nn.SmoothL1Loss(reduction="none")
        self.box_weight = box_weight
        self.iou_weight = iou_weight
        self.focal_gamma = focal_gamma
        self.background_weight = background_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        matched_classes: torch.Tensor,
        matched_boxes: torch.Tensor,
        matched_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """根据匹配结果计算总损失，并返回训练时常用的中间量。"""
        logits = predictions["logits"]
        pred_boxes = predictions["pred_boxes"]

        batch_size, num_queries, num_classes = logits.shape
        logits_flat = logits.reshape(batch_size * num_queries, num_classes)
        targets_flat = matched_classes.reshape(batch_size * num_queries)
        log_prob = torch.log_softmax(logits_flat, dim=-1)
        prob = log_prob.exp()
        target_log_prob = log_prob.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        background_class = num_classes - 1
        alpha = torch.ones_like(target_prob)
        alpha = torch.where(targets_flat == background_class, torch.full_like(alpha, self.background_weight), alpha)
        focal_factor = (1.0 - target_prob).pow(self.focal_gamma)
        classification_loss = -(alpha * focal_factor * target_log_prob).mean()

        valid_pred_boxes = pred_boxes[matched_mask]
        valid_target_boxes = matched_boxes[matched_mask]
        if valid_pred_boxes.numel() == 0:
            box_loss = pred_boxes.sum() * 0.0
            iou_loss = pred_boxes.sum() * 0.0
        else:
            raw_box_loss = self.box_loss(valid_pred_boxes, valid_target_boxes)
            box_loss = raw_box_loss.mean()
            iou_matrix = pairwise_iou_xywh(valid_pred_boxes, valid_target_boxes)
            iou_loss = 1.0 - torch.diag(iou_matrix).mean()

        loss = classification_loss + self.box_weight * box_loss + self.iou_weight * iou_loss
        predicted_class = logits.argmax(dim=-1)
        return {
            "loss": loss,
            "classification_loss": classification_loss.detach(),
            "box_loss": box_loss.detach(),
            "iou_loss": iou_loss.detach(),
            "predicted_class": predicted_class,
            "pred_boxes": pred_boxes.detach(),
        }
