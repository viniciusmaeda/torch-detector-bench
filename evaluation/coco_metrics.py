"""Official COCO AP and per-class precision-recall data."""

from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_evaluate(ground_truth, detections, category_ids, max_detections=100):
    """Return COCO 101-point AP; classes without ground truth stay undefined."""
    with redirect_stdout(StringIO()):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {"images": ground_truth["images"], "categories": ground_truth["categories"], "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = [i["id"] for i in ground_truth["images"]]
        evaluator.params.catIds = sorted(category_ids)
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    def average(values):
        valid = values[values >= 0]
        return float(valid.mean()) if valid.size else float("nan")
    metrics = {"mAP50": average(precision[0]), "mAP50_95": average(precision)}
    names = {c["id"]: c["name"] for c in ground_truth["categories"]}
    per_class, curves = [], []
    for index, category_id in enumerate(evaluator.params.catIds):
        anns = [a for a in ground_truth["annotations"] if a["category_id"] == category_id and not a.get("iscrowd", 0)]
        per_class.append({"Category_ID": category_id, "Class": names[category_id],
                          "Images": len({a["image_id"] for a in anns}), "Objects": len(anns),
                          "AP50": average(precision[0, :, index]), "AP50_95": average(precision[:, :, index])})
        for r, recall in enumerate(evaluator.params.recThrs):
            curves.append({"Class": names[category_id], "Recall": float(recall),
                           "Precision50": average(precision[0, r:r+1, index]),
                           "Precision50_95": average(precision[:, r, index])})
    return metrics, pd.DataFrame(per_class), pd.DataFrame(curves)
