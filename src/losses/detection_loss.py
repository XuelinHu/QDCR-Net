import torch
from torch import nn


class DetectionLoss(nn.Module):
    """Classification + matched-box regression loss."""

    def __init__(self, box_weight: float = 5.0) -> None:
        super().__init__()
        self.classification_loss = nn.CrossEntropyLoss()
        self.box_loss = nn.SmoothL1Loss(reduction="none")
        self.box_weight = box_weight

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        matched_classes: torch.Tensor,
        matched_boxes: torch.Tensor,
        matched_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = predictions["logits"]
        pred_boxes = predictions["pred_boxes"]

        batch_size, num_queries, num_classes = logits.shape
        classification_loss = self.classification_loss(
            logits.reshape(batch_size * num_queries, num_classes),
            matched_classes.reshape(batch_size * num_queries),
        )

        valid_mask_expanded = matched_mask.unsqueeze(-1).float()
        raw_box_loss = self.box_loss(pred_boxes, matched_boxes)
        box_loss = (raw_box_loss * valid_mask_expanded).sum() / valid_mask_expanded.sum().clamp(min=1.0)
        loss = classification_loss + self.box_weight * box_loss
        predicted_class = logits.argmax(dim=-1)
        return {
            "loss": loss,
            "classification_loss": classification_loss.detach(),
            "box_loss": box_loss.detach(),
            "predicted_class": predicted_class,
            "pred_boxes": pred_boxes.detach(),
        }
