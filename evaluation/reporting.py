"""Publication figures, fold-level summaries and exploratory paired statistics."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils.artifacts import save_figure, save_table, write_json

METRICS = ["mAP50_95", "mAP50", "Precision", "Recall", "F1", "IoU", "MAE", "RMSE",
           "MAPE", "Bias", "Pearson_r", "R2", "FPS", "Inference_time", "Parameters", "Training_seconds"]
LOWER_IS_BETTER = {"MAE", "RMSE", "MAPE", "Inference_time", "Training_seconds"}


def validate_reporting_config(config):
    settings = config["statistics"]
    metrics = settings["comparison_metrics"]
    if (not 0 < float(settings["alpha"]) < 1 or not set(metrics).issubset(METRICS)
            or settings["primary_metric"] not in metrics or len(set(metrics)) != len(metrics)):
        raise ValueError("Invalid statistical configuration.")
    required = settings.get("required_models", [])
    if not set(required).issubset({"yolo", "rtdetr", "fasterrcnn", "detr"}) or len(set(required)) != len(required):
        raise ValueError("statistics.required_models must contain distinct supported model keys.")


def corrected_paired_test(differences, test_train_ratio, alpha=0.05):
    """Nadeau–Bengio approximate correction for overlapping training sets.

SE = sqrt((1/n + mean(n_test/n_train)) * sample_variance(differences)).
The correction is approximate, especially for a single five-fold experiment.
"""
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("Paired inference requires at least two finite differences.")
    mean, sd = float(differences.mean()), float(differences.std(ddof=1))
    n = len(differences)
    se = float(np.sqrt((1/n + test_train_ratio) * sd**2))
    if np.isclose(sd, 0, rtol=0, atol=1e-12):
        identical = bool(np.isclose(mean, 0, rtol=0, atol=1e-12))
        return {"Mean_difference_A_minus_B": mean, "SD_difference": sd, "Corrected_SE": se,
                "CI_low": None, "CI_high": None,
                "t": None, "df": n-1, "p_raw": 1.0 if identical else None,
                "Paired_dz": None, "Status": "identical observed scores; CI undefined" if identical else "zero variance: inference undefined"}
    statistic = mean/se
    margin = float(stats.t.ppf(1-alpha/2, n-1)*se)
    return {"Mean_difference_A_minus_B": mean, "SD_difference": sd, "Corrected_SE": se,
            "CI_low": mean-margin, "CI_high": mean+margin, "t": statistic, "df": n-1,
            "p_raw": float(2*stats.t.sf(abs(statistic), n-1)), "Paired_dz": mean/sd,
            "Status": "exploratory approximate corrected t-test"}


def holm_adjust(pvalues):
    """Holm family-wise adjustment, with undefined tests retained as p=1."""
    values = np.asarray([1.0 if p is None or not np.isfinite(p) else p for p in pvalues])
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def generate_reports(root: Path, config: dict):
    """Use only atomically completed folds and refuse incompatible results."""
    validate_reporting_config(config)
    manifest = json.loads((root / "splits" / "manifest.json").read_text())
    expected = set(range(1, manifest["n_splits"]+1))
    rows, folders = [], {}
    for completion in sorted(root.glob("*/fold_*/completed.json")):
        folder = completion.parent
        row = json.loads((folder / "metrics.json").read_text())
        marker = json.loads(completion.read_text())
        if row["Manifest_SHA256"] != manifest["manifest_sha256"] or marker["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Cannot combine metrics from different manifests.")
        if marker["run_sha256"] != row["Run_SHA256"]:
            raise ValueError("Fold completion marker does not match its metrics.")
        val_ids = {i["image_id"] for i in manifest["images"] if i["fold"] == row["Fold"]}
        counts = pd.read_csv(folder / "evaluation" / "image_counts.csv")
        if counts.Image_ID.duplicated().any() or set(counts.Image_ID) != val_ids:
            raise ValueError("Held-out predictions do not match the saved fold allocation.")
        rows.append(row)
        folders[(row["Model"], row["Fold"])] = folder
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    write_methods(report, manifest, config)
    if not rows:
        write_json(report / "status.json", {"status": "no completed training folds", "expected_folds": len(expected)})
        return
    frame = pd.DataFrame(rows).sort_values(["Model", "Fold"])
    if frame.duplicated(["Model", "Fold"]).any():
        raise ValueError("Duplicate model/fold result.")
    if frame.Source_SHA256.nunique() != 1:
        raise ValueError("Cannot compare models evaluated by different source code versions.")
    if frame.Environment_SHA256.nunique() != 1:
        raise ValueError("Cannot combine model results from different software/hardware environments.")
    save_table(frame, report / "fold_metrics", "Independent final-epoch evaluation of each held-out fold")
    summary, manuscript, status = [], [], []
    complete_models = []
    for model, group in frame.groupby("Model", sort=False):
        if group.Run_SHA256.nunique() != 1:
            raise ValueError("A model's folds have different training configurations.")
        complete = set(group.Fold) == expected
        if complete:
            complete_models.append(model)
        status.append({"Model": model, "Completed_folds": len(group), "Expected_folds": len(expected), "Complete": complete})
        paper_row = {"Model": model, "Folds": f"{len(group)}/{len(expected)}", "Status": "complete" if complete else "partial"}
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean = float(values.mean()) if len(values) else np.nan
            sd = float(values.std(ddof=1)) if len(values)>1 else np.nan
            summary.append({"Model": model, "Metric": metric, "N_valid": len(values), "Expected_folds": len(expected),
                            "Complete": complete, "Mean": mean, "SD": sd, "Median": values.median(),
                            "Min": values.min(), "Max": values.max()})
            if metric in ("mAP50_95", "mAP50", "Precision", "Recall", "F1", "MAE", "RMSE", "FPS"):
                paper_row[metric] = f"{mean:.4f} ± {sd:.4f}" if np.isfinite(sd) else f"{mean:.4f} (SD unavailable)"
        manuscript.append(paper_row)
    save_table(pd.DataFrame(summary), report / "summary_statistics", "Descriptive fold statistics; sample SD (ddof=1)")
    save_table(pd.DataFrame(manuscript), report / "manuscript_table", "Final held-out fold scores: mean ± sample SD; partial runs explicitly marked")
    save_table(pd.DataFrame(status), report / "completion_status", "Benchmark completion status")
    write_json(report / "status.json", {"status": "results available", "models": status})
    generate_comparisons(frame, complete_models, report, config)
    plot_fold_metrics(frame, report)
    generate_class_reports(frame, folders, report)
    generate_learning_curves(frame, folders, report)
    for model in complete_models:
        generate_oof_report(model, frame, folders, manifest, report, config)
    print(pd.DataFrame(manuscript).to_string(index=False))


def generate_comparisons(frame, complete_models, report, config):
    settings = config["statistics"]
    alpha = float(settings["alpha"])
    metrics = settings["comparison_metrics"]
    if not 0 < alpha < 1 or not set(metrics).issubset(METRICS) or settings["primary_metric"] not in metrics:
        raise ValueError("Invalid statistical configuration.")
    required = set(settings.get("required_models", ["yolo", "rtdetr", "fasterrcnn", "detr"]))
    keys = set(frame.loc[frame.Model.isin(complete_models), "Model_Key"])
    if not required.issubset(keys):
        write_json(report / "comparison_status.json", {"status": "waiting for all pre-specified models to complete",
                                                       "missing_model_keys": sorted(required-keys)})
        return
    comparisons = []
    for a, b in combinations(complete_models, 2):
        left = frame[frame.Model == a].set_index("Fold")
        right = frame[frame.Model == b].set_index("Fold")
        paired = left.join(right, lsuffix="_A", rsuffix="_B", how="inner").sort_index()
        if not np.array_equal(paired.Train_Images_A, paired.Train_Images_B) or not np.array_equal(paired.Evaluation_Images_A, paired.Evaluation_Images_B):
            raise ValueError("Paired comparison requires identical fold sizes.")
        ratio = float((paired.Evaluation_Images_A/paired.Train_Images_A).mean())
        for metric in metrics:
            differences = (pd.to_numeric(paired[f"{metric}_A"], errors="coerce") - pd.to_numeric(paired[f"{metric}_B"], errors="coerce")).to_numpy()
            if not np.isfinite(differences).all():
                result = {"Status": "undefined metric in at least one fold", "p_raw": None}
            else:
                result = corrected_paired_test(differences, ratio, alpha)
            comparisons.append({"Model_A": a, "Model_B": b, "Metric": metric,
                                "Primary": metric == settings["primary_metric"], "N_pairs": len(paired),
                                "Test_train_ratio": ratio, "CI_level": 1-alpha,
                                "Better_direction": "lower" if metric in LOWER_IS_BETTER else "higher", **result})
    result = pd.DataFrame(comparisons)
    if result.empty:
        write_json(report / "comparison_status.json", {"status": "at least two complete models are required"})
        return
    adjusted = holm_adjust(result.p_raw.tolist())
    result["p_Holm"] = [p if pd.notna(raw) else np.nan for p, raw in zip(adjusted, result.p_raw)]
    result["Reject_H0_exploratory"] = result.p_Holm < alpha
    save_table(result, report / "paired_comparisons", "Exploratory Nadeau–Bengio corrected paired t-tests; Holm over all model-pair/metric tests; CIs are unadjusted")
    paper = []
    for row in result[result.Primary].to_dict("records"):
        value = row.get("Mean_difference_A_minus_B", np.nan)
        lo, hi = row.get("CI_low"), row.get("CI_high")
        paper.append({"Model_A": row["Model_A"], "Model_B": row["Model_B"], "Metric": row["Metric"],
                      "Difference_A_minus_B": f"{value:.4f}" if pd.notna(value) else "NA",
                      "Corrected_CI": f"[{lo:.4f}, {hi:.4f}]" if pd.notna(lo) and pd.notna(hi) else "NA",
                      "p_Holm": f"{row['p_Holm']:.4g}" if pd.notna(row['p_Holm']) else "NA"})
    save_table(pd.DataFrame(paper), report / "paired_manuscript_table",
               "Primary metric: exploratory paired differences, approximate unadjusted corrected CIs and Holm-adjusted p-values")
    write_json(report / "comparison_status.json", {"status": "exploratory comparisons available", "family_size": len(result),
                                                    "alpha": alpha, "primary_metric": settings["primary_metric"]})
    primary = result[result.Primary & result.CI_low.notna()].copy() if "CI_low" in result else pd.DataFrame()
    if not primary.empty:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(10, max(4, len(primary)*0.8)))
        for index, row in enumerate(primary.itertuples()):
            axis.plot([row.CI_low, row.CI_high], [index, index], color="tab:blue")
            axis.scatter(row.Mean_difference_A_minus_B, index, color="tab:blue")
        axis.set_yticks(range(len(primary)), [f"{a} − {b}" for a, b in zip(primary.Model_A, primary.Model_B)], fontsize=8)
        axis.axvline(0, color="black", linestyle="--")
        axis.set_xlabel(f"Difference in {settings['primary_metric']} (approximate corrected {(1-alpha)*100:g}% CI; unadjusted)")
        save_figure(figure, report / "figures" / "paired_primary_differences")


def plot_fold_metrics(frame, report):
    import matplotlib.pyplot as plt
    selected = ["mAP50_95", "mAP50", "F1", "MAE", "RMSE", "FPS"]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    models = frame.Model.unique()
    for axis, metric in zip(axes.flat, selected):
        series = [pd.to_numeric(frame.loc[frame.Model == m, metric], errors="coerce").dropna().to_numpy() for m in models]
        axis.boxplot(series, positions=range(len(models)), widths=0.45)
        for fold, group in frame.groupby("Fold"):
            values = [group.loc[group.Model == m, metric].iloc[0] if m in set(group.Model) else np.nan for m in models]
            axis.plot(range(len(models)), values, marker="o", linewidth=0.7, alpha=0.7, label=f"Fold {fold}")
        axis.set_xticks(range(len(models)), models, rotation=25, ha="right", fontsize=8)
        axis.set_title(metric)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("Final held-out scores; points/lines identify paired folds")
    figure.tight_layout()
    save_figure(figure, report / "figures" / "fold_metric_comparison")


def generate_class_reports(frame, folders, report):
    records = []
    for row in frame.itertuples():
        table = pd.read_csv(folders[(row.Model, row.Fold)] / "evaluation" / "per_class.csv")
        table["Model"], table["Fold"] = row.Model, row.Fold
        records.append(table)
    classes = pd.concat(records, ignore_index=True)
    save_table(classes, report / "per_class_folds", "Class support and final metrics by held-out fold")
    summary = classes.groupby(["Model", "Class"], sort=False).agg(
        AP50_95_mean=("AP50_95", "mean"), AP50_95_SD=("AP50_95", "std"),
        AP50_mean=("AP50", "mean"), F1_mean=("F1", "mean"), Evaluated_images=("Images", "sum"),
        Evaluated_objects=("Objects", "sum"), Folds_with_AP=("AP50_95", "count")).reset_index()
    save_table(summary, report / "per_class_summary", "Per-class fold means and sample SD; unsupported classes remain undefined")
    import matplotlib.pyplot as plt
    matrix = summary.pivot(index="Class", columns="Model", values="AP50_95_mean")
    figure, axis = plt.subplots(figsize=(max(7, len(matrix.columns)*2), 11))
    heat = axis.imshow(np.ma.masked_invalid(matrix.to_numpy()), vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20, ha="right", fontsize=8)
    axis.set_yticks(range(len(matrix)), matrix.index, fontsize=9)
    for r in range(len(matrix)):
        for c in range(len(matrix.columns)):
            value = matrix.iloc[r, c]
            axis.text(c, r, f"{value:.2f}" if pd.notna(value) else "NA", ha="center", va="center",
                      color="black" if pd.notna(value) and value > 0.6 else "white", fontsize=8)
    figure.colorbar(heat, ax=axis, label="Mean held-out AP50–95")
    save_figure(figure, report / "figures" / "per_class_ap")


def generate_learning_curves(frame, folders, report):
    normalized = []
    aliases = {"precision": "Precision", "recall": "Recall", "f1": "F1", "map50": "mAP50", "map50_95": "mAP50_95",
               "metrics/precision(B)": "Precision", "metrics/recall(B)": "Recall",
               "metrics/mAP50(B)": "mAP50", "metrics/mAP50-95(B)": "mAP50_95"}
    for row in frame.itertuples():
        folder = folders[(row.Model, row.Fold)]
        paths = list((folder / "training_curves").glob("*_history.csv"))
        path = paths[0] if paths else folder / "checkpoints" / "results.csv"
        if not path.exists():
            continue
        history = pd.read_csv(path)
        history.columns = history.columns.str.strip()
        record = pd.DataFrame({"Epoch": history.epoch, "Model": row.Model, "Fold": row.Fold})
        for column, name in aliases.items():
            if column in history:
                record[name] = history[column]
        for split in ("train", "val"):
            if f"{split}_loss" in history:
                record[f"{split.title()}_Loss"] = history[f"{split}_loss"]
            else:
                columns = [c for c in history if c.startswith(split+"/") and c.endswith("loss")]
                if columns:
                    record[f"{split.title()}_Loss"] = history[columns].sum(axis=1)
        if "F1" not in record and {"Precision", "Recall"}.issubset(record.columns):
            denominator = record.Precision+record.Recall
            record["F1"] = np.where(denominator>0, 2*record.Precision*record.Recall/denominator, 0)
        normalized.append(record)
    if not normalized:
        return
    history = pd.concat(normalized, ignore_index=True)
    history.to_csv(report / "learning_curves.csv", index=False)
    import matplotlib.pyplot as plt
    for model, group in history.groupby("Model", sort=False):
        figure, axes = plt.subplots(2, 3, figsize=(13, 7))
        for axis, metric in zip(axes.flat, ["Train_Loss", "Val_Loss", "Precision", "Recall", "mAP50", "mAP50_95"]):
            if metric not in group:
                axis.set_visible(False)
                continue
            for fold, trajectory in group.groupby("Fold"):
                axis.plot(trajectory.Epoch, trajectory[metric], alpha=0.25, linewidth=0.8)
            means = group.groupby("Epoch")[metric].mean()
            sd = group.groupby("Epoch")[metric].std().fillna(0)
            axis.plot(means.index, means, color="black", linewidth=1.8)
            axis.fill_between(means.index, means-sd, means+sd, alpha=0.15, color="black")
            axis.set(xlabel="Epoch", title=metric)
            axis.grid(alpha=0.2)
        figure.suptitle(f"{model}: native monitoring, fold mean ± SD (not final benchmark)")
        figure.tight_layout()
        key = folders[(model, int(group.Fold.iloc[0]))].parent.name
        save_figure(figure, report / "figures" / f"learning_{key}")


def generate_oof_report(model, frame, folders, manifest, report, config):
    """Pool predictions only for descriptive analysis, separate from fold statistics."""
    from evaluation.coco_metrics import coco_evaluate
    from evaluation.evaluate import plot_evaluation
    counts, classes, detections = [], [], []
    gt = {"info": {}, "images": [], "categories": [], "annotations": []}
    confusion = None
    model_frame = frame[frame.Model == model]
    for row in model_frame.itertuples():
        folder = folders[(model, row.Fold)] / "evaluation"
        c = pd.read_csv(folder / "image_counts.csv")
        c["Fold"] = row.Fold
        counts.append(c)
        classes.append(pd.read_csv(folder / "class_counts.csv"))
        detections.extend(json.loads((folder / "predictions.coco.json").read_text()))
        part = json.loads((folder / "ground_truth.coco.json").read_text())
        gt["images"].extend(part["images"])
        gt["categories"] = part["categories"]
        for ann in part["annotations"]:
            ann["id"] = len(gt["annotations"])+1
            gt["annotations"].append(ann)
        matrix = pd.read_csv(folder / "confusion_matrix.csv", index_col=0).to_numpy()
        confusion = matrix if confusion is None else confusion+matrix
    counts = pd.concat(counts, ignore_index=True)
    if counts.Image_ID.duplicated().any() or set(counts.Image_ID) != {i["image_id"] for i in manifest["images"]}:
        raise ValueError("OOF analysis requires exactly one held-out prediction per image.")
    key = folders[(model, int(model_frame.Fold.iloc[0]))].parent.name
    output = report / "out_of_fold" / key
    write_json(output / "predictions.coco.json", detections)
    write_json(output / "ground_truth.coco.json", gt)
    counts.to_csv(output / "image_counts.csv", index=False)
    pd.concat(classes, ignore_index=True).to_csv(output / "class_counts.csv", index=False)
    metrics, per_class, curves = coco_evaluate(gt, detections, manifest["category_ids"], int(config["evaluation"]["max_detections"]))
    errors = counts.Error.to_numpy()
    metrics.update(MAE=float(np.abs(errors).mean()), RMSE=float(np.sqrt(np.mean(errors**2))), Bias=float(errors.mean()))
    write_json(output / "descriptive_metrics.json", {"Model": model, "Interpretation": "pooled OOF descriptive scores, not mean fold scores", **metrics})
    save_table(per_class, output / "per_class", "Pooled out-of-fold descriptive class AP")
    curves.to_csv(output / "precision_recall.csv", index=False)
    names = manifest["class_names"][1:]+["background"]
    pd.DataFrame(confusion, index=names, columns=names).to_csv(output / "confusion_matrix.csv")
    plot_evaluation(counts, per_class, curves, confusion, names, output)


def write_methods(report, manifest, config):
    settings = config["evaluation"]
    text = f"""# Scientific reporting protocol

Dataset: {len(manifest['images'])} images; {len(manifest['category_ids'])} annotated classes.
Five-fold design configured as {manifest['n_splits']} folds, seed {manifest['seed']}; all images participate.
Manifest SHA256: `{manifest['manifest_sha256']}`.
Allocation: capacity-constrained rare-label-first iterative multilabel stratification,
following Sechidis, Tsoumakas and Vlahavas (2011). Presence per image is the stratification target;
box counts are audited separately. Explicit group CSV: {bool(manifest['group_sha256'])}.
Filename equality is not used to infer groups. Images from the same source/video or augmented
variants must be assigned explicit groups when present. The default assumes independent images.

Each fold starts from fresh pretrained weights and a fresh optimizer/scheduler.
Each training command restarts every requested fold; prior results for selected model
families and the previous reports are archived and excluded from current comparisons.
Training epochs are specified before the experiment. Evaluation uses the FINAL epoch checkpoint,
never the best checkpoint selected by the held-out fold; Ultralytics uses final-epoch EMA weights.
Native learning curves may monitor held-out data but do not select stopping time or weights.
Hyperparameter/model selection after inspecting CV scores can introduce selection bias; this
experiment does not provide an independent final test set or nested hyperparameter selection.

Final AP uses the official pycocotools COCO evaluator, IoU 0.50:0.05:0.95,
101 recall points, all object sizes, maximum {settings['max_detections']} detections/image,
minimum retained score {settings['ap_score_threshold']}. AP50 is the IoU 0.50 slice.
Precision/recall/F1 are micro metrics with fixed score {settings['score_threshold']} and
IoU {settings['iou_threshold']}; boxes are matched in descending confidence order.
Counts are total objects per image at this same fixed threshold; per-class counts are also exported.
MAPE excludes images whose true count is zero; valid support is reported.
Undefined AP/correlation/R2 values stay NA, never zero. No crowd annotations are supported
in the common Ultralytics training export. Parameter counts include all trained-architecture
parameters (including frozen ones), before optional inference-time BatchNorm fusion.
Training_seconds includes monitoring, checkpoint/curve export and training-phase galleries.
FPS is batch-amortized predict latency including preprocessing/postprocessing, excluding disk
loading, with CUDA synchronization and {settings['warmup_batches']} warm-up calls.
Compare FPS only on the same hardware and consider the configured batch sizes.

Tables summarize FINAL held-out scores, not epochs: mean, sample SD (ddof=1), median, range,
and valid-fold counts. Boxplots display five actual points with fold pairing.
Learning curves are descriptive native framework monitoring; native Ultralytics P/R/AP differ
from the common final evaluator and must not be used as final cross-model benchmark scores.
OOF pooled metrics/PR curves/confusion/count-agreement plots are descriptive and are separately
labeled; pooled AP is not the mean of fold AP because predictions from different models are mixed.
Bland–Altman limits are descriptive mean error ± 1.96 sample SD, without a normality claim.

Primary metric: {config['statistics']['primary_metric']}; pre-specified comparison metrics:
{', '.join(config['statistics']['comparison_metrics'])}.
Paired comparisons require all configured required_models to have every fold completed, with the
same manifest and evaluation source. Nadeau–Bengio approximate corrected paired t-test:
SE = sqrt((1/k + mean(n_test/n_train)) * sample variance of paired differences), df = k−1.
Report raw difference A−B, paired dz (descriptive), approximate {(1-config['statistics']['alpha'])*100:g}%
UNADJUSTED difference CI and two-sided p. Holm adjusts across ALL reported model-pair/metric
tests together. These CIs are not simultaneous family-wise intervals.
Degenerate nonzero differences with zero variance have undefined inference.
With only five folds, power is low and normality cannot be established reliably: all tests are
EXPLORATORY approximations, not proof of superiority. Overlapping training sets are not
independent replicates. Epochs and individual bounding boxes are never statistical replicates.
More independent datasets or pre-specified repeated CV would strengthen inference.

References:
- https://github.com/trent-b/iterative-stratification (Sechidis et al. reference and original algorithm)
- https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
"""
    (report / "methods_and_statistics.md").write_text(text, encoding="utf-8")
