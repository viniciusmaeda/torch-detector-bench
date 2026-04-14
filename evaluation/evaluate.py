"""Evaluation pipeline for object detection models."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch

from evaluation.metrics import (
    compute_computational_metrics,
    compute_count_metrics,
    compute_detection_metrics,
    compute_map,
    xywh_to_xyxy,
)


def evaluate_model(model, dataloader, device: torch.device, config: Dict[str, Any], logger=None) -> Dict[str, float]:
    """Evaluate a detector on the test split and compute benchmark metrics."""
    all_predictions: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []
    inference_times: List[float] = []

    score_threshold = float(config["evaluation"]["score_threshold"])
    iou_threshold = float(config["evaluation"]["iou_threshold"])
    iou_thresholds = [round(value, 2) for value in torch.arange(0.5, 1.0, 0.05).tolist()]

    for images, targets in dataloader:
        start_time = time.perf_counter()
        predictions = model.predict_batch(images=images, device=device, score_threshold=score_threshold)
        elapsed = time.perf_counter() - start_time

        batch_size = max(len(images), 1)
        inference_times.extend([elapsed / batch_size] * batch_size)

        for prediction, target in zip(predictions, targets):
            all_predictions.append(
                {
                    "boxes": prediction.boxes,
                    "scores": prediction.scores,
                    "labels": prediction.labels,
                }
            )
            all_targets.append(
                {
                    "boxes": xywh_to_xyxy(target["boxes_xywh"]).cpu(),
                    "labels": target["labels"].cpu(),
                }
            )

    detection_metrics = compute_detection_metrics(
        predictions=all_predictions,
        targets=all_targets,
        iou_threshold=iou_threshold,
    )
    map50, map50_95 = compute_map(
        predictions=all_predictions,
        targets=all_targets,
        iou_thresholds=iou_thresholds,
    )
    count_metrics = compute_count_metrics(predictions=all_predictions, targets=all_targets)
    computational_metrics = compute_computational_metrics(
        inference_times=inference_times,
        num_parameters=model.num_parameters(),
    )

    metrics = {
        **detection_metrics,
        "mAP50": map50,
        "mAP50_95": map50_95,
        **count_metrics,
        **computational_metrics,
    }
    if logger is not None:
        logger.info("Evaluation metrics: %s", metrics)
    return metrics
