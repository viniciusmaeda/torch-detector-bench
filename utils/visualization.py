"""Visualization and reporting helpers."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle


def print_summary_table(results_df: pd.DataFrame) -> None:
    """Print the condensed benchmark summary table."""
    summary = results_df[
        ["Model", "Architecture_Type", "mAP50", "MAE", "RMSE", "Pearson_r", "FPS"]
    ].copy()
    summary = summary.rename(columns={"Architecture_Type": "Architecture", "Pearson_r": "r"})
    print(summary.to_string(index=False))


def save_training_history(history: list[dict], output_path: Path) -> pd.DataFrame:
    """Persist epoch-level training history to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_path, index=False)
    return history_df


def plot_training_history(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    """Create a YOLO-like training summary plot for PyTorch detectors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = history_df["epoch"]
    plots = [
        ("Loss", [("train_loss", "Train Loss"), ("val_loss", "Val Loss")]),
    ]

    component_pairs: list[tuple[str, list[tuple[str, str]]]] = []
    for column in sorted(history_df.columns):
        if not column.startswith("train_") or column == "train_loss":
            continue
        base_name = column.removeprefix("train_")
        val_column = f"val_{base_name}"
        if val_column in history_df.columns:
            component_pairs.append(
                (
                    base_name.replace("_", " ").title(),
                    [(column, f"Train {base_name}"), (val_column, f"Val {base_name}")],
                )
            )

    plots.extend(component_pairs)
    plots.extend(
        [
            ("Precision", [("precision", "Precision")]),
            ("Recall", [("recall", "Recall")]),
            ("F1", [("f1", "F1")]),
            ("mAP50", [("map50", "mAP50")]),
            ("mAP50-95", [("map50_95", "mAP50-95")]),
        ]
    )

    num_columns = 3
    num_rows = ceil(len(plots) / num_columns)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(14, 4 * num_rows))
    axes = axes.flatten()

    for axis, (panel_title, series) in zip(axes, plots):
        for column, label in series:
            if column in history_df:
                axis.plot(epochs, history_df[column], marker="o", linewidth=2, label=label)
        axis.set_title(panel_title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        if len(series) > 1:
            axis.legend()

    for axis in axes[len(plots):]:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_prediction_gallery(
    model,
    dataloader,
    device,
    output_path: Path,
    score_threshold: float,
    class_names: list[str],
    max_images: int = 4,
) -> None:
    """Save a small qualitative prediction gallery for a trained detector."""
    if dataloader is None or max_images <= 0:
        return

    batch = next(iter(dataloader), None)
    if batch is None:
        return

    images, _targets = batch
    images = images[:max_images]
    predictions = model.predict_batch(
        images=images,
        device=device,
        score_threshold=score_threshold,
    )

    figure, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    if len(images) == 1:
        axes = [axes]

    for axis, image, prediction in zip(axes, images, predictions):
        axis.imshow(image)
        axis.axis("off")
        for box, score, label in zip(prediction.boxes, prediction.scores, prediction.labels):
            x0, y0, x1, y1 = [float(value) for value in box.tolist()]
            width = max(x1 - x0, 1.0)
            height = max(y1 - y0, 1.0)
            axis.add_patch(
                Rectangle(
                    (x0, y0),
                    width,
                    height,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2,
                )
            )
            class_index = int(label)
            class_name = class_names[class_index] if 0 <= class_index < len(class_names) else str(class_index)
            axis.text(
                x0,
                max(y0 - 4.0, 0.0),
                f"{class_name}: {float(score):.2f}",
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.7, "pad": 2},
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
