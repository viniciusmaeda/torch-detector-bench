"""Single entry point for training and benchmarking object detectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
import yaml

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
        choices=["yolo", "rtdetr", "fasterrcnn", "detr", "all"],
        help="Model architecture to train and evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the experiment configuration file.",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true", help="Validate dataset and export common folds without training.")
    actions.add_argument("--reports-only", action="store_true", help="Regenerate publication reports from completed folds.")
    parser.add_argument("--fold", nargs="+", type=int, help="Start fresh training on selected folds (1-based); default: all folds.")
    args = parser.parse_args()
    if not args.model and not (args.prepare_only or args.reports_only):
        parser.error("--model is required for training.")
    return args


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_model(model_name: str, config: Dict[str, Any], device: torch.device):
    """Instantiate the selected detector."""
    num_classes = int(config["dataset"]["num_classes"])
    class_names = config["dataset"]["class_names"]

    if model_name == "yolo":
        from models.yolo_model import YOLODetector
        return YOLODetector(config=config, class_names=class_names, device=device)
    if model_name == "rtdetr":
        from models.rtdetr_model import RTDETRDetector
        return RTDETRDetector(config=config, class_names=class_names, device=device)
    if model_name == "fasterrcnn":
        from models.faster_rcnn_model import FasterRCNNDetector
        return FasterRCNNDetector(
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            config=config,
        )
    if model_name == "detr":
        from models.detr_model import DETRDetector
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


def run_holdout(args, config, project_root) -> None:
    """Run the full train and evaluation pipeline."""
    from datasets.dataset_loader import build_dataloaders
    from evaluation.evaluate import evaluate_model
    from training.train import train_model

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


def environment_metadata(device):
    import importlib.metadata
    import platform
    import subprocess
    packages = {}
    for name in ("torch", "torchvision", "ultralytics", "transformers", "numpy", "pandas", "scipy", "pycocotools", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda, "git_commit": commit.stdout.strip() or None}


def archive_previous_runs(root, model_names, current_dirs, source_hash, environment_hash):
    """Start clean runs while preserving previous scientific artifacts outside reports."""
    from datetime import datetime, timezone
    import json
    import shutil
    from utils.artifacts import write_json

    selected = {path for path in current_dirs if path.exists()}
    for path in root.glob("*/run.json"):
        previous = json.loads(path.read_text())
        resolved = previous["resolved"]
        if resolved["model_key"] in model_names:
            selected.add(path.parent)
        elif (resolved["source_sha256"] != source_hash
              or resolved.get("environment_sha256") != environment_hash):
            raise ValueError(
                "Other models have results from different code/environment. "
                "Run --model all or use a new experiment.name before comparing them."
            )
    reports = root / "reports"
    if not selected and not reports.exists():
        return
    archive = root / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive.mkdir(parents=True, exist_ok=False)
    moves = []
    for path in sorted(selected):
        destination = archive / path.name
        path.rename(destination)
        moves.append({"original_path": str(path), "archived_path": str(destination)})
    if reports.exists():
        reports.rename(archive / "reports")
        # Preparation just regenerated these dataset figures; keep them available.
        figures = reports / "figures"
        figures.mkdir(parents=True)
        for name in ("class_distribution", "fold_class_support"):
            for suffix in (".png", ".pdf", ".svg"):
                source = archive / "reports" / "figures" / (name+suffix)
                if source.exists():
                    shutil.copy2(source, figures / source.name)
    write_json(archive / "archive.json", {"models": moves, "reason": "fresh training execution"})
    print(f"Previous results archived at {archive}")


def run_cross_validation(args, config, project_root):
    from copy import deepcopy
    from time import perf_counter
    from datetime import datetime, timezone
    import gc
    from datasets.cross_validation import prepare_cross_validation
    from evaluation.reporting import generate_reports, validate_reporting_config
    from utils.artifacts import fingerprint, write_json
    from utils.paths import get_model_base_dir

    validate_reporting_config(config)
    manifest = prepare_cross_validation(config, project_root)
    root = get_results_root(project_root, config)
    write_json(root / "resolved_config.json", config)
    print(f"Prepared {len(manifest['images'])} images, {len(manifest['category_ids'])} annotated classes, {manifest['n_splits']} folds.")
    print(f"Manifest: {root / 'splits' / 'manifest.json'}")
    if args.prepare_only:
        return
    if args.reports_only:
        generate_reports(root, config)
        print(f"Reports: {root / 'reports'}")
        return
    from datasets.dataset_loader import build_dataloaders
    from evaluation.evaluate import evaluate_model
    from training.train import train_model
    from utils.visualization import save_prediction_gallery

    folds = args.fold or list(range(1, manifest["n_splits"]+1))
    if len(set(folds)) != len(folds) or any(f < 1 or f > manifest["n_splits"] for f in folds):
        raise ValueError("--fold must contain distinct fold numbers within the configured range.")
    models = ["yolo", "rtdetr", "fasterrcnn", "detr"] if args.model == "all" else [args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = environment_metadata(device)
    environment_hash = fingerprint({k: v for k, v in environment.items() if k != "git_commit"})
    sources = {str(p.relative_to(project_root)): p.read_text() for directory in ("datasets", "evaluation", "models", "training", "utils")
               for p in (project_root / directory).glob("*.py")}
    sources["main.py"] = (project_root / "main.py").read_text()
    source_hash = fingerprint(sources)
    write_json(root / "source" / f"{source_hash}.json", sources)
    runs = []
    for model_name in models:
        resolved = {"model_key": model_name, "model": config.get("models", {}).get(model_name, {}),
                    "training": get_training_config(config, model_name), "manifest_sha256": manifest["manifest_sha256"],
                    "source_sha256": source_hash, "environment_sha256": environment_hash}
        training_settings = resolved["training"]
        if int(training_settings["epochs"]) <= 0 or int(training_settings["batch_size"]) <= 0 or int(training_settings["image_size"]) <= 0:
            raise ValueError("Training epochs, batch_size and image_size must be positive.")
        run_hash = fingerprint(resolved)
        model_root = get_model_base_dir(project_root, config, model_name)
        runs.append((model_name, resolved, run_hash, model_root))
    # Archive every selected model before training any of them, so --model all
    # cannot mix this execution with older results during intermediate reporting.
    archive_previous_runs(root, models, [run[3] for run in runs], source_hash, environment_hash)
    write_json(root / "reports" / "status.json", {"status": "fresh training execution started", "models": models, "folds": folds})
    for model_name, resolved, run_hash, model_root in runs:
        metadata_path = model_root / "run.json"
        write_json(metadata_path, {"run_sha256": run_hash, "resolved": resolved, "environment": environment,
                   "started_utc": datetime.now(timezone.utc).isoformat()})
        for fold in folds:
            fold_config = deepcopy(config)
            heldout_ids = [i["image_id"] for i in manifest["images"] if i["fold"] == fold]
            train_ids = [i["image_id"] for i in manifest["images"] if i["fold"] != fold]
            fold_config["_cv"] = {"fold": fold, "seed": int(config["experiment"]["seed"])+fold-1,
                                   "train_ids": train_ids, "val_ids": heldout_ids,
                                   "data_yaml": str((root / "splits" / f"fold_{fold:02d}" / "data.yaml").resolve())}
            fold_root = get_model_results_dir(project_root, fold_config, model_name)
            completed_path = fold_root / "completed.json"
            set_seed(fold_config["_cv"]["seed"])
            logger = get_logger(fold_root / "training_logs", f"{model_name}_fold_{fold:02d}")
            log_config(logger, fold_config)
            write_json(fold_root / "config.json", fold_config)
            write_json(fold_root / "environment.json", environment)
            logger.info("Fold %s: %s training / %s held-out images; final-epoch checkpoint", fold, len(train_ids), len(heldout_ids))
            dataloaders = build_dataloaders(fold_config, project_root, model_name)
            model = build_model(model_name, fold_config, device)
            optimizer = model.build_optimizer()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = perf_counter()
            checkpoint = train_model(model, dataloaders["train"], optimizer, device,
                                     val_dataloader=dataloaders["val"], config=fold_config, logger=logger)
            training_seconds = perf_counter()-start
            metrics = evaluate_model(model, dataloaders["val"], device, fold_config, logger, fold_root / "evaluation")
            if model.is_ultralytics:
                save_prediction_gallery(model, dataloaders["val"], device,
                                        fold_root / "evaluation" / "figures" / "qualitative_predictions",
                                        float(config["evaluation"]["score_threshold"]), model.class_names,
                                        int(get_training_config(config, model_name).get("qualitative_images", 6)))
            row = {"Model": model.name, "Model_Key": model_name, "Architecture_Type": model.architecture_type,
                   "Fold": fold, "Seed": fold_config["_cv"]["seed"], "Train_Images": len(train_ids),
                   "Evaluation_Images": len(heldout_ids), "Epochs": int(get_training_config(config, model_name)["epochs"]),
                   "Manifest_SHA256": manifest["manifest_sha256"], "Run_SHA256": run_hash,
                   "Source_SHA256": source_hash, "Checkpoint": str(checkpoint), "Training_seconds": training_seconds,
                   "Environment_SHA256": environment_hash,
                   "Peak_GPU_memory_MB": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None,
                   **metrics}
            write_json(fold_root / "metrics.json", row)
            write_json(completed_path, {"run_sha256": run_hash, "manifest_sha256": manifest["manifest_sha256"],
                                       "completed_utc": datetime.now(timezone.utc).isoformat()})
            del model, optimizer, dataloaders
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        generate_reports(root, config)
    print(f"Publication reports: {root / 'reports'}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config = load_config(str(project_root / args.config))
    if config.get("cross_validation", {}).get("enabled", False):
        run_cross_validation(args, config, project_root)
    else:
        if args.prepare_only or args.reports_only or args.fold or args.model == "all":
            raise ValueError("These options require cross_validation.enabled: true.")
        run_holdout(args, config, project_root)


if __name__ == "__main__":
    main()
