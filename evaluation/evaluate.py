"""Common, reproducible evaluation for every detector and CV fold."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from evaluation.coco_metrics import coco_evaluate
from evaluation.metrics import compute_computational_metrics, compute_count_metrics, compute_detection_metrics, xywh_to_xyxy
from utils.artifacts import save_figure, save_table, write_json


def evaluate_model(model, dataloader, device, config, logger=None, output_dir: Path | None = None):
    """Evaluate low-score detections for AP and fixed-threshold detections for counts."""
    settings = config["evaluation"]
    score = float(settings["score_threshold"])
    ap_score = float(settings.get("ap_score_threshold", 0.001))
    maximum = int(settings["max_detections"])
    if not 0 <= ap_score <= score <= 1 or maximum < 10:
        raise ValueError("Require 0 <= ap_score_threshold <= score_threshold <= 1 and max_detections >= 10.")
    category_ids = config["dataset"].get("category_ids", list(range(1, len(model.class_names))))
    categories = [{"id": c, "name": name} for c, name in zip(category_ids, model.class_names[1:])]
    ground_truth = {"info": {}, "images": [], "annotations": [], "categories": categories}
    detections, predictions, targets, times, image_rows, class_rows = [], [], [], [], [], []
    confusion = np.zeros((len(categories) + 1, len(categories) + 1), dtype=int)
    for batch_index, (images, batch_targets) in enumerate(dataloader):
        if batch_index == 0:
            for _ in range(int(settings.get("warmup_batches", 2))):
                model.predict_batch(images, device, ap_score)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        with torch.inference_mode():
            outputs = model.predict_batch(images, device, ap_score)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        if len(outputs) != len(images):
            raise ValueError("Detector returned a different number of predictions than images.")
        times.extend([elapsed / len(images)] * len(images))
        for image, target, output in zip(images, batch_targets, outputs):
            image_id = int(target["image_id"].item())
            order = torch.argsort(output.scores, descending=True)[:maximum]
            boxes, scores, labels = output.boxes[order].cpu(), output.scores[order].cpu(), output.labels[order].cpu()
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise FloatingPointError("Non-finite detector outputs.")
            if labels.numel() and (labels.min() < 1 or labels.max() > len(categories)):
                raise ValueError("Prediction label outside the global class mapping.")
            for box, confidence, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = [float(v) for v in box]
                detections.append({"image_id": image_id, "category_id": category_ids[int(label)-1],
                                   "bbox": [x0, y0, max(x1-x0, 0), max(y1-y0, 0)], "score": float(confidence)})
            fixed = scores >= score
            prediction = {"boxes": boxes[fixed], "scores": scores[fixed], "labels": labels[fixed]}
            keep = target["iscrowd"] == 0
            truth = {"boxes": xywh_to_xyxy(target["boxes_xywh"])[keep].cpu(), "labels": target["labels"][keep].cpu()}
            predictions.append(prediction)
            targets.append(truth)
            ground_truth["images"].append({"id": image_id, "width": image.width, "height": image.height})
            for box, label, area, crowd in zip(target["boxes_xywh"].tolist(), target["labels"].tolist(), target["area"].tolist(), target["iscrowd"].tolist()):
                ground_truth["annotations"].append({"id": len(ground_truth["annotations"])+1,
                    "image_id": image_id, "category_id": category_ids[label-1], "bbox": box, "area": area, "iscrowd": crowd})
            gt_count, pred_count = len(truth["boxes"]), len(prediction["boxes"])
            image_rows.append({"Image_ID": image_id, "File_Name": target["file_name"], "True_Count": gt_count,
                               "Predicted_Count": pred_count, "Error": pred_count-gt_count,
                               "Inference_time": elapsed/len(images)})
            for class_index, category in enumerate(categories, start=1):
                n_true = int((truth["labels"] == class_index).sum())
                n_pred = int((prediction["labels"] == class_index).sum())
                class_rows.append({"Image_ID": image_id, "Class": category["name"], "True_Count": n_true, "Predicted_Count": n_pred})
            confusion += detection_confusion(prediction, truth, len(categories), float(settings["iou_threshold"]))
    if not targets:
        raise ValueError("Cannot evaluate an empty fold.")
    ap, class_frame, curves = coco_evaluate(ground_truth, detections, category_ids, maximum)
    metrics = {**compute_detection_metrics(predictions, targets, float(settings["iou_threshold"])), **ap,
               **compute_count_metrics(predictions, targets),
               **compute_computational_metrics(times, model.num_parameters())}
    for class_index, name in enumerate(model.class_names[1:], start=1):
        cp = [{k: v[p["labels"] == class_index] for k, v in p.items()} for p in predictions]
        ct = [{k: v[t["labels"] == class_index] for k, v in t.items()} for t in targets]
        result = compute_detection_metrics(cp, ct, float(settings["iou_threshold"]))
        for metric in ("Precision", "Recall", "F1"):
            class_frame.loc[class_frame.Class == name, metric] = result[metric] if int(class_frame.loc[class_frame.Class == name, "Objects"].iloc[0]) else np.nan
    if output_dir is not None:
        write_json(output_dir / "predictions.coco.json", detections)
        write_json(output_dir / "ground_truth.coco.json", ground_truth)
        save_table(class_frame, output_dir / "per_class", "Held-out per-class detection metrics")
        pd.DataFrame(image_rows).to_csv(output_dir / "image_counts.csv", index=False)
        pd.DataFrame(class_rows).to_csv(output_dir / "class_counts.csv", index=False)
        curves.to_csv(output_dir / "precision_recall.csv", index=False)
        names = model.class_names[1:] + ["background"]
        pd.DataFrame(confusion, index=names, columns=names).to_csv(output_dir / "confusion_matrix.csv")
        plot_evaluation(pd.DataFrame(image_rows), class_frame, curves, confusion, names, output_dir)
    if logger:
        logger.info("Held-out evaluation: %s", metrics)
    return metrics


def detection_confusion(prediction, truth, n_classes, threshold):
    """Greedy confidence-ordered, class-agnostic IoU matching; rows are truth."""
    from evaluation.metrics import box_iou
    matrix = np.zeros((n_classes+1, n_classes+1), dtype=int)
    overlap = box_iou(prediction["boxes"], truth["boxes"])
    used = set()
    for index in range(len(prediction["boxes"])):
        available = [j for j in range(len(truth["boxes"])) if j not in used]
        best = max(available, key=lambda j: float(overlap[index, j])) if available else None
        pred = int(prediction["labels"][index])-1
        if best is not None and float(overlap[index, best]) >= threshold:
            matrix[int(truth["labels"][best])-1, pred] += 1
            used.add(best)
        else:
            matrix[n_classes, pred] += 1
    for index, label in enumerate(truth["labels"]):
        if index not in used:
            matrix[int(label)-1, n_classes] += 1
    return matrix


def plot_evaluation(counts, classes, curves, confusion, names, output_dir):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(counts.True_Count, counts.Predicted_Count, alpha=0.4, s=12)
    limit = max(counts.True_Count.max(), counts.Predicted_Count.max(), 1)
    axes[0].plot([0, limit], [0, limit], "k--")
    axes[0].set(xlabel="Ground-truth count", ylabel="Predicted count")
    means = (counts.True_Count + counts.Predicted_Count)/2
    bias = counts.Error.mean()
    axes[1].scatter(means, counts.Error, alpha=0.4, s=12)
    for value in (bias, bias-1.96*counts.Error.std(ddof=1), bias+1.96*counts.Error.std(ddof=1)):
        axes[1].axhline(value, linestyle="--", color="black")
    axes[1].set(xlabel="Mean of true and predicted counts", ylabel="Prediction − truth", title="Bland–Altman (descriptive limits)")
    save_figure(figure, output_dir / "figures" / "count_agreement")
    figure, axis = plt.subplots(figsize=(7, 5))
    for name, frame in curves.groupby("Class", sort=False):
        axis.plot(frame.Recall, frame.Precision50, alpha=0.6, linewidth=0.8)
    macro = curves.groupby("Recall").Precision50.mean()
    axis.plot(macro.index, macro, color="black", linewidth=2, label="Macro mean (supported classes)")
    axis.set(xlabel="Recall", ylabel="Interpolated precision", title="COCO precision–recall at IoU 0.50", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend()
    save_figure(figure, output_dir / "figures" / "precision_recall")
    figure, axis = plt.subplots(figsize=(15, 13))
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums>0)
    heat = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    axis.set(xlabel="Predicted class", ylabel="Ground-truth class", title="Detection confusion (row normalized)")
    figure.colorbar(heat, ax=axis)
    save_figure(figure, output_dir / "figures" / "confusion_matrix")
