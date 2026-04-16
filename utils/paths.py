"""Helpers for organizing experiment artifacts inside the results directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from utils.config import get_model_key

MODEL_RESULTS_DIRS = {
    "detr": "Detr",
    "fasterrcnn": "FasterRCNN",
    "cascadercnn": "CascadeRCNN",
}
CHECKPOINTS_DIRNAME = "checkpoints"


def get_results_root(project_root: Path, config: Dict[str, Any]) -> Path:
    """Return the root directory that stores experiment outputs."""
    return project_root / str(config["experiment"]["output_dir"])


def format_yolo_results_dir(model_name: str) -> str:
    """Format a YOLO variant into the expected top-level results directory name."""
    stem = Path(model_name).stem
    if stem.lower().startswith("yolo"):
        return f"YOLO{stem[4:]}"
    return stem


def format_rtdetr_results_dir(model_name: str) -> str:
    """Format an RT-DETR variant into the expected top-level results directory name."""
    stem = Path(model_name).stem
    if stem.lower().startswith("rtdetr"):
        suffix = stem[6:]
        return f"RTDETR{suffix}"
    return stem


def format_fasterrcnn_results_dir(model_name: str) -> str:
    """Format a Faster R-CNN variant into the expected top-level results directory name."""
    stem = Path(model_name).stem
    if stem.lower().startswith("fasterrcnn_"):
        suffix = stem[len("fasterrcnn_"):].replace("_", "-")
        return f"FasterRCNN-{suffix}"
    return MODEL_RESULTS_DIRS["fasterrcnn"]


def format_cascadercnn_results_dir(model_name: str) -> str:
    """Format a Cascade R-CNN variant into the expected top-level results directory name."""
    stem = Path(model_name).stem
    if stem.lower().startswith("cascade_rcnn_"):
        suffix = stem[len("cascade_rcnn_"):].replace("_", "-")
        return f"CascadeRCNN-{suffix}"
    return MODEL_RESULTS_DIRS["cascadercnn"]


def get_model_results_dir(project_root: Path, config: Dict[str, Any], model_name: str) -> Path:
    """Return the model-specific directory inside results/."""
    model_key = get_model_key(model_name)
    results_root = get_results_root(project_root=project_root, config=config)
    if model_key == "yolo":
        yolo_model_name = str(config.get("models", {}).get("yolo", {}).get("model_name", "yolo"))
        return results_root / format_yolo_results_dir(yolo_model_name)
    if model_key == "rtdetr":
        rtdetr_model_name = str(config.get("models", {}).get("rtdetr", {}).get("model_name", "rtdetr"))
        return results_root / format_rtdetr_results_dir(rtdetr_model_name)
    if model_key == "fasterrcnn":
        fasterrcnn_model_name = str(
            config.get("models", {}).get("fasterrcnn", {}).get("model_name", "fasterrcnn")
        )
        return results_root / format_fasterrcnn_results_dir(fasterrcnn_model_name)
    if model_key == "cascadercnn":
        cascadercnn_model_name = str(
            config.get("models", {}).get("cascadercnn", {}).get("model_name", "cascadercnn")
        )
        return results_root / format_cascadercnn_results_dir(cascadercnn_model_name)
    return results_root / MODEL_RESULTS_DIRS[model_key]


def get_model_checkpoints_dir(project_root: Path, config: Dict[str, Any], model_name: str) -> Path:
    """Return the fixed checkpoints directory for a given model."""
    return get_model_results_dir(
        project_root=project_root,
        config=config,
        model_name=model_name,
    ) / CHECKPOINTS_DIRNAME
