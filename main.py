"""Single entry point for training and benchmarking object detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import yaml

from datasets.dataset_loader import build_dataloaders
from evaluation.evaluate import evaluate_model
from models.detr_model import DETRDetector
from models.faster_rcnn_model import FasterRCNNDetector
from models.rtdetr_model import RTDETRDetector
from models.yolo_model import YOLODetector
from training.train import train_model
from utils.config import get_training_config
from utils.logger import get_logger, log_config, set_seed
from utils.optimizer_factory import get_optimizer_hyperparameters, get_scheduler_hyperparameters
from utils.paths import get_model_results_dir, get_results_root
from utils.visualization import print_summary_table


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare PyTorch object detectors.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolo", "rtdetr", "fasterrcnn", "detr"],
        help="Model architecture to train and evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the experiment configuration file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_model(model_name: str, config: Dict[str, Any], device: torch.device):
    """Instantiate the selected detector."""
    num_classes = int(config["dataset"]["num_classes"])
    class_names = config["dataset"]["class_names"]

    if model_name == "yolo":
        return YOLODetector(config=config, class_names=class_names, device=device)
    if model_name == "rtdetr":
        return RTDETRDetector(config=config, class_names=class_names, device=device)
    if model_name == "fasterrcnn":
        return FasterRCNNDetector(
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            config=config,
        )
    if model_name == "detr":
        return DETRDetector(
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            config=config,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def save_results(row: Dict[str, Any], metrics_path: Path) -> pd.DataFrame:
    """Append experiment results to the metrics CSV file."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    if metrics_path.exists():
        current_df = pd.read_csv(metrics_path)
        combined_df = pd.concat([current_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_csv(metrics_path, index=False)
    return combined_df


def main() -> None:
    """Run the full train and evaluation pipeline."""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = project_root / args.config
    config = load_config(str(config_path))

    if not config["dataset"].get("class_names") or int(config["dataset"].get("num_classes", 0)) <= 1:
        raise ValueError(
            "Dataset classes are not configured in config/config.yaml. "
            "Run `python datasets/sync_dataset_classes.py` before main.py."
        )

    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_results_dir = get_model_results_dir(
        project_root=project_root,
        config=config,
        model_name=args.model,
    )

    logger = get_logger(
        log_dir=model_results_dir / "training_logs",
        name=f"{args.model}_benchmark",
    )
    log_config(logger=logger, config=config)
    logger.info("Using device: %s", device)
    logger.info("Resolved training config: %s", get_training_config(config, args.model))
    logger.info("Resolved optimizer: %s", get_optimizer_hyperparameters(config, args.model))
    logger.info("Resolved scheduler: %s", get_scheduler_hyperparameters(config, args.model))

    dataloaders = build_dataloaders(
        config=config,
        project_root=project_root,
        model_name=args.model,
    )
    model = build_model(model_name=args.model, config=config, device=device)

    optimizer = model.build_optimizer()
    train_model(
        model=model,
        dataloader=dataloaders["train"],
        optimizer=optimizer,
        device=device,
        val_dataloader=dataloaders["val"],
        config=config,
        logger=logger,
    )

    metrics = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        config=config,
        logger=logger,
    )

    metrics_row = {
        "Model": model.name,
        "Architecture_Type": model.architecture_type,
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "Bias": metrics["Bias"],
        "Pearson_r": metrics["Pearson_r"],
        "R2": metrics["R2"],
        "FPS": metrics["FPS"],
        "Inference_time": metrics["Inference_time"],
        "Parameters": metrics["Parameters"],
    }

    metrics_csv = get_results_root(project_root=project_root, config=config) / "metrics.csv"
    metrics_df = save_results(row=metrics_row, metrics_path=metrics_csv)
    print_summary_table(metrics_df.tail(1))
    logger.info("Saved metrics to %s", metrics_csv)


if __name__ == "__main__":
    main()
