# -*- coding: utf-8 -*-
"""检测任务中的匹配、解码和指标计算工具。"""

import torch
from torchvision.ops import nms


def pairwise_iou_xywh(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """计算两组 xywh 归一化框之间的两两 IoU。"""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.size(0), boxes_b.size(0)), dtype=boxes_a.dtype, device=boxes_a.device)

    boxes_b = boxes_b.to(device=boxes_a.device, dtype=boxes_a.dtype)

    boxes_a_xyxy = xywh_to_xyxy(boxes_a)
    boxes_b_xyxy = xywh_to_xyxy(boxes_b)

    left_top = torch.maximum(boxes_a_xyxy[:, None, :2], boxes_b_xyxy[None, :, :2])
    right_bottom = torch.minimum(boxes_a_xyxy[:, None, 2:], boxes_b_xyxy[None, :, 2:])
    wh = (right_bottom - left_top).clamp(min=0.0)
    intersection = wh[..., 0] * wh[..., 1]

    area_a = (boxes_a_xyxy[:, 2] - boxes_a_xyxy[:, 0]).clamp(min=0.0) * (
        boxes_a_xyxy[:, 3] - boxes_a_xyxy[:, 1]
    ).clamp(min=0.0)
    area_b = (boxes_b_xyxy[:, 2] - boxes_b_xyxy[:, 0]).clamp(min=0.0) * (
        boxes_b_xyxy[:, 3] - boxes_b_xyxy[:, 1]
    ).clamp(min=0.0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-6)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """把中心点格式边界框转换为角点格式。"""
    center_x = boxes[..., 0]
    center_y = boxes[..., 1]
    width = boxes[..., 2]
    height = boxes[..., 3]
    return torch.stack(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        dim=-1,
    )


def greedy_match(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    background_class: int,
    cls_weight: float = 1.0,
    l1_weight: float = 3.0,
    iou_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """基于分类、L1 与 IoU 混合代价做贪心匹配。"""
    num_queries = logits.size(0)
    matched_classes = torch.full(
        (num_queries,),
        background_class,
        dtype=torch.long,
        device=logits.device,
    )
    matched_boxes = torch.zeros((num_queries, 4), dtype=pred_boxes.dtype, device=pred_boxes.device)
    matched_mask = torch.zeros((num_queries,), dtype=torch.bool, device=logits.device)

    if gt_classes.numel() == 0:
        return matched_classes, matched_boxes, matched_mask

    probabilities = logits.softmax(dim=-1)
    iou_matrix = pairwise_iou_xywh(pred_boxes, gt_boxes)
    l1_matrix = torch.cdist(pred_boxes, gt_boxes, p=1)
    cls_cost = -probabilities[:, gt_classes]
    total_cost = cls_weight * cls_cost + l1_weight * l1_matrix + iou_weight * (1.0 - iou_matrix)

    # 展平成候选对后排序，实现一个简单但足够稳定的匹配过程。
    candidate_pairs: list[tuple[float, int, int]] = []
    for pred_index in range(num_queries):
        for gt_index in range(gt_classes.size(0)):
            candidate_pairs.append((float(total_cost[pred_index, gt_index].item()), pred_index, gt_index))
    candidate_pairs.sort(key=lambda item: item[0])

    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    for _, pred_index, gt_index in candidate_pairs:
        if pred_index in used_predictions or gt_index in used_ground_truth:
            continue
        used_predictions.add(pred_index)
        used_ground_truth.add(gt_index)
        matched_classes[pred_index] = gt_classes[gt_index]
        matched_boxes[pred_index] = gt_boxes[gt_index]
        matched_mask[pred_index] = True

    return matched_classes, matched_boxes, matched_mask


def decode_predictions(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    background_class: int,
    conf_thresh: float,
    iou_thresh: float,
) -> list[dict[str, torch.Tensor]]:
    """把模型输出解码为按图片组织的检测结果，并做按类 NMS。"""
    batch_detections: list[dict[str, torch.Tensor]] = []
    probabilities = logits.softmax(dim=-1)
    scores, classes = probabilities[..., :-1].max(dim=-1)

    for batch_index in range(logits.size(0)):
        score = scores[batch_index]
        cls = classes[batch_index]
        box = pred_boxes[batch_index]
        keep = score >= conf_thresh
        score = score[keep]
        cls = cls[keep]
        box = box[keep]
        if score.numel() == 0:
            batch_detections.append(
                {
                    "scores": torch.zeros((0,), device=logits.device),
                    "classes": torch.zeros((0,), dtype=torch.long, device=logits.device),
                    "boxes": torch.zeros((0, 4), device=logits.device),
                }
            )
            continue

        kept_indices = []
        box_xyxy = xywh_to_xyxy(box)
        for class_id in cls.unique():
            # 按类别分别执行 NMS，避免不同类别之间相互抑制。
            class_mask = cls == class_id
            class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
            selected = nms(box_xyxy[class_mask], score[class_mask], iou_thresh)
            kept_indices.append(class_indices[selected])
        final_indices = (
            torch.cat(kept_indices, dim=0)
            if kept_indices
            else torch.zeros((0,), dtype=torch.long, device=logits.device)
        )
        final_indices = final_indices[score[final_indices].argsort(descending=True)]
        batch_detections.append(
            {
                "scores": score[final_indices],
                "classes": cls[final_indices],
                "boxes": box[final_indices],
            }
        )

    return batch_detections


def compute_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    ground_truths: list[dict[str, torch.Tensor]],
    num_classes: int,
) -> dict[str, float]:
    """计算 mAP、precision、recall 等聚合指标。"""
    ap50_per_class = []
    ap5095_per_class = []
    precisions = []
    recalls = []
    for class_id in range(num_classes):
        ap50 = _average_precision(predictions, ground_truths, class_id, 0.5)
        ap_thresholds = [
            _average_precision(predictions, ground_truths, class_id, threshold)
            for threshold in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
        ]
        precision, recall = _precision_recall(predictions, ground_truths, class_id, 0.5)
        ap50_per_class.append(ap50)
        ap5095_per_class.append(sum(ap_thresholds) / len(ap_thresholds))
        precisions.append(precision)
        recalls.append(recall)

    return {
        "map50": sum(ap50_per_class) / max(len(ap50_per_class), 1),
        "map50_95": sum(ap5095_per_class) / max(len(ap5095_per_class), 1),
        "precision": sum(precisions) / max(len(precisions), 1),
        "recall": sum(recalls) / max(len(recalls), 1),
    }


def _average_precision(
    predictions: list[dict[str, torch.Tensor]],
    ground_truths: list[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> float:
    """针对单一类别和单一 IoU 阈值计算 AP。"""
    scored_predictions = []
    total_ground_truth = 0
    gt_lookup: dict[int, dict[str, torch.Tensor]] = {}

    for image_index, gt in enumerate(ground_truths):
        gt_mask = gt["classes"] == class_id
        gt_boxes = gt["boxes"][gt_mask]
        gt_lookup[image_index] = {
            "boxes": gt_boxes,
            "matched": torch.zeros((gt_boxes.size(0),), dtype=torch.bool),
        }
        total_ground_truth += int(gt_boxes.size(0))

    for image_index, prediction in enumerate(predictions):
        pred_mask = prediction["classes"] == class_id
        for score, box in zip(prediction["scores"][pred_mask], prediction["boxes"][pred_mask]):
            scored_predictions.append((float(score.item()), image_index, box))

    if total_ground_truth == 0:
        return 0.0

    scored_predictions.sort(key=lambda item: item[0], reverse=True)
    true_positive = []
    false_positive = []

    for _, image_index, pred_box in scored_predictions:
        gt_entry = gt_lookup[image_index]
        gt_boxes = gt_entry["boxes"]
        if gt_boxes.numel() == 0:
            true_positive.append(0.0)
            false_positive.append(1.0)
            continue

        ious = pairwise_iou_xywh(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_index = torch.max(ious, dim=0)
        if best_iou.item() >= iou_threshold and not gt_entry["matched"][best_index]:
            gt_entry["matched"][best_index] = True
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)

    tp_cumsum = torch.tensor(true_positive).cumsum(dim=0)
    fp_cumsum = torch.tensor(false_positive).cumsum(dim=0)
    recalls = tp_cumsum / max(total_ground_truth, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)

    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([1.0]), precisions, torch.tensor([0.0])])
    # 从后向前做 precision envelope，得到标准 PR 曲线面积。
    for index in range(precisions.numel() - 2, -1, -1):
        precisions[index] = torch.maximum(precisions[index], precisions[index + 1])

    recall_steps = recalls[1:] - recalls[:-1]
    return float(torch.sum(recall_steps * precisions[1:]).item())


def _precision_recall(
    predictions: list[dict[str, torch.Tensor]],
    ground_truths: list[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> tuple[float, float]:
    """在给定 IoU 阈值下计算单类别 precision 与 recall。"""
    scored_predictions = []
    total_ground_truth = 0
    gt_lookup: dict[int, dict[str, torch.Tensor]] = {}

    for image_index, gt in enumerate(ground_truths):
        gt_mask = gt["classes"] == class_id
        gt_boxes = gt["boxes"][gt_mask]
        gt_lookup[image_index] = {
            "boxes": gt_boxes,
            "matched": torch.zeros((gt_boxes.size(0),), dtype=torch.bool),
        }
        total_ground_truth += int(gt_boxes.size(0))

    for image_index, prediction in enumerate(predictions):
        pred_mask = prediction["classes"] == class_id
        for score, box in zip(prediction["scores"][pred_mask], prediction["boxes"][pred_mask]):
            scored_predictions.append((float(score.item()), image_index, box))

    scored_predictions.sort(key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    for _, image_index, pred_box in scored_predictions:
        gt_entry = gt_lookup[image_index]
        gt_boxes = gt_entry["boxes"]
        if gt_boxes.numel() == 0:
            fp += 1
            continue

        ious = pairwise_iou_xywh(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_index = torch.max(ious, dim=0)
        if best_iou.item() >= iou_threshold and not gt_entry["matched"][best_index]:
            gt_entry["matched"][best_index] = True
            tp += 1
        else:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_ground_truth, 1)
    return precision, recall
