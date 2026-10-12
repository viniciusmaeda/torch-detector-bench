"""Common Ultralytics training policy for independent CV folds."""

from pathlib import Path

from datasets.dataset_loader import build_yolo_data_config
from utils.config import get_training_value
from utils.paths import CHECKPOINTS_DIRNAME, get_model_results_dir


def fit_ultralytics(detector, config, device, logger, constructor):
    data_path = build_yolo_data_config(config, detector.project_root, detector.config_key)
    root = get_model_results_dir(detector.project_root, config, detector.config_key)
    training = {
        "data": str(data_path), "model": detector.model_name,
        "epochs": int(get_training_value(config, detector.config_key, "epochs")),
        "imgsz": int(get_training_value(config, detector.config_key, "image_size")),
        "batch": int(get_training_value(config, detector.config_key, "batch_size")),
        "workers": int(get_training_value(config, detector.config_key, "num_workers")),
        "lr0": float(get_training_value(config, detector.config_key, "learning_rate")),
        "weight_decay": float(get_training_value(config, detector.config_key, "weight_decay", 0.0)),
        "momentum": float(get_training_value(config, detector.config_key, "momentum", 0.9)),
        "project": str(root), "name": CHECKPOINTS_DIRNAME,
        "device": (device.index or 0) if device.type == "cuda" else str(device),
        "optimizer": detector.optimizer_name, "pretrained": True, "exist_ok": True,
        "seed": int(config.get("_cv", {}).get("seed", config["experiment"]["seed"])),
        "deterministic": True, "plots": True,
    }
    if "_cv" in config:
        training.update(patience=0, resume=False, save=True)
    detector.model.train(**training)
    trainer = detector.model.trainer
    if hasattr(trainer, "model"):
        # Count the trained architecture before prediction fuses BatchNorm layers.
        detector.parameter_count = sum(p.numel() for p in trainer.model.parameters())
    checkpoint = Path(trainer.save_dir) / "weights" / ("last.pt" if "_cv" in config else "best.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not produce the required checkpoint: {checkpoint}")
    if "_cv" in config:
        import pandas as pd
        history = pd.read_csv(Path(trainer.save_dir) / "results.csv")
        history.columns = history.columns.str.strip()
        if len(history) != training["epochs"]:
            raise RuntimeError("Training stopped before the pre-specified final epoch.")
    detector.evaluation_checkpoint_path = checkpoint
    detector.best_checkpoint_path = checkpoint  # Backwards-compatible attribute.
    detector.model = constructor(str(checkpoint))
    logger.info("%s evaluation checkpoint: %s", detector.name, checkpoint)
    return checkpoint
