"""Metric computation for object detection experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class MatchSummary:
    """Container for matching statistics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    ious: List[float]


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from xywh to xyxy."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    converted = boxes.clone()
    converted[:, 2] = boxes[:, 0] + boxes[:, 2]
    converted[:, 3] = boxes[:, 1] + boxes[:, 3]
    return converted


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU matrix."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


def greedy_match(
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float,
) -> MatchSummary:
    """Greedily match predictions to ground truth by label and IoU."""
    matched_gt = set()
    tp = 0
    fp = 0
    ious: List[float] = []

    if pred_boxes.numel() == 0:
        return MatchSummary(0, 0, int(gt_boxes.shape[0]), [])

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    for pred_index in range(pred_boxes.shape[0]):
        best_iou = 0.0
        best_gt = -1
        for gt_index in range(gt_boxes.shape[0]):
            if gt_index in matched_gt:
                continue
            if pred_labels[pred_index] != gt_labels[gt_index]:
                continue
            candidate_iou = float(iou_matrix[pred_index, gt_index].item())
            if candidate_iou >= iou_threshold and candidate_iou > best_iou:
                best_iou = candidate_iou
                best_gt = gt_index

        if best_gt >= 0:
            matched_gt.add(best_gt)
            tp += 1
            ious.append(best_iou)
        else:
            fp += 1

    fn = int(gt_boxes.shape[0] - len(matched_gt))
    return MatchSummary(tp, fp, fn, ious)


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP with monotonic precision envelope."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def compute_map(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    iou_thresholds: Iterable[float],
) -> Tuple[float, float]:
    """Compute mAP@0.5 and mAP@0.5:0.95."""
    class_ids = sorted(
        {
            int(label.item())
            for target in targets
            for label in target["labels"]
        }
    )
    ap_per_threshold: Dict[float, List[float]] = {threshold: [] for threshold in iou_thresholds}

    for threshold in iou_thresholds:
        for class_id in class_ids:
            scores: List[float] = []
            matches: List[int] = []
            total_gt = 0
            per_image_gt_used: Dict[int, set] = {}

            for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
                pred_mask = prediction["labels"] == class_id
                gt_mask = target["labels"] == class_id
                pred_boxes = prediction["boxes"][pred_mask]
                pred_scores = prediction["scores"][pred_mask]
                gt_boxes = target["boxes"][gt_mask]
                total_gt += int(gt_boxes.shape[0])
                per_image_gt_used[image_index] = set()

                if pred_boxes.numel() == 0:
                    continue

                sorted_indices = torch.argsort(pred_scores, descending=True)
                pred_boxes = pred_boxes[sorted_indices]
                pred_scores = pred_scores[sorted_indices]
                iou_matrix = box_iou(pred_boxes, gt_boxes) if gt_boxes.numel() > 0 else torch.zeros((pred_boxes.shape[0], 0))

                for pred_idx in range(pred_boxes.shape[0]):
                    scores.append(float(pred_scores[pred_idx].item()))
                    best_iou = 0.0
                    best_gt_idx = -1
                    for gt_idx in range(gt_boxes.shape[0]):
                        if gt_idx in per_image_gt_used[image_index]:
                            continue
                        candidate = float(iou_matrix[pred_idx, gt_idx].item())
                        if candidate >= threshold and candidate > best_iou:
                            best_iou = candidate
                            best_gt_idx = gt_idx
                    if best_gt_idx >= 0:
                        per_image_gt_used[image_index].add(best_gt_idx)
                        matches.append(1)
                    else:
                        matches.append(0)

            if total_gt == 0:
                continue
            if not scores:
                ap_per_threshold[threshold].append(0.0)
                continue

            order = np.argsort(-np.asarray(scores))
            true_positive = np.asarray(matches)[order]
            false_positive = 1 - true_positive
            cumulative_tp = np.cumsum(true_positive)
            cumulative_fp = np.cumsum(false_positive)
            recalls = cumulative_tp / max(total_gt, 1)
            precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-6)
            ap_per_threshold[threshold].append(_compute_ap(recalls=recalls, precisions=precisions))

    map50 = float(np.mean(ap_per_threshold.get(0.5, [0.0]))) if ap_per_threshold.get(0.5) else 0.0
    map50_95 = float(
        np.mean([ap for values in ap_per_threshold.values() for ap in values])
    ) if any(ap_per_threshold.values()) else 0.0
    return map50, map50_95


def compute_detection_metrics(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    iou_threshold: float,
) -> Dict[str, float]:
    """Compute precision, recall, F1 and mean IoU."""
    tp = 0
    fp = 0
    fn = 0
    ious: List[float] = []

    for prediction, target in zip(predictions, targets):
        match = greedy_match(
            pred_boxes=prediction["boxes"],
            pred_labels=prediction["labels"],
            gt_boxes=target["boxes"],
            gt_labels=target["labels"],
            iou_threshold=iou_threshold,
        )
        tp += match.true_positives
        fp += match.false_positives
        fn += match.false_negatives
        ious.extend(match.ious)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-6)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1_score),
        "IoU": mean_iou,
    }


def compute_count_metrics(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    """Compute image-level count error metrics."""
    pred_counts = np.asarray([prediction["boxes"].shape[0] for prediction in predictions], dtype=np.float64)
    gt_counts = np.asarray([target["boxes"].shape[0] for target in targets], dtype=np.float64)
    diff = pred_counts - gt_counts

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    positive = gt_counts > 0
    mape = float(np.mean(np.abs(diff[positive]) / gt_counts[positive]) * 100.0) if positive.any() else float("nan")
    bias = float(np.mean(diff))

    if pred_counts.size > 1 and np.std(pred_counts) > 0 and np.std(gt_counts) > 0:
        pearson_r = float(np.corrcoef(pred_counts, gt_counts)[0, 1])
    else:
        pearson_r = float("nan")

    ss_res = float(np.sum((gt_counts - pred_counts) ** 2))
    ss_tot = float(np.sum((gt_counts - np.mean(gt_counts)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Bias": bias,
        "Pearson_r": pearson_r,
        "R2": float(r2),
        "MAPE_valid_images": int(positive.sum()),
    }


def compute_computational_metrics(inference_times: Sequence[float], num_parameters: int) -> Dict[str, float]:
    """Compute runtime metrics."""
    mean_time = float(np.mean(inference_times)) if inference_times else math.inf
    fps = float(1.0 / mean_time) if inference_times and mean_time > 0 else 0.0
    return {
        "Inference_time": mean_time,
        "FPS": fps,
        "Parameters": float(num_parameters),
    }
