"""Ultralytics YOLO wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import torch
from PIL import Image
from ultralytics import YOLO

from models.common import PredictionResult
from models.ultralytics_training import fit_ultralytics
from utils.config import get_training_value
from utils.optimizer_factory import get_optimizer_hyperparameters

SUPPORTED_YOLO_MODELS: Final[tuple[str, ...]] = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
)
DEFAULT_YOLO_MODEL: Final[str] = "yolov8n.pt"


class YOLODetector:
    """Wrapper around Ultralytics YOLO models."""

    config_key = "yolo"
    architecture_type = "One-stage detector"
    is_ultralytics = True

    def __init__(self, config: Dict[str, Any], class_names: List[str], device: torch.device) -> None:
        self.config = config
        self.class_names = class_names
        self.device = device
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_name = self._resolve_model_name(config)
        self.optimizer_name = get_optimizer_hyperparameters(config, self.config_key)["name"]
        self.name = f"YOLO ({self.model_name})"
        self.model = YOLO(self.model_name)
        self.best_checkpoint_path: Optional[Path] = None

    @staticmethod
    def _resolve_model_name(config: Dict[str, Any]) -> str:
        """Read and validate the configured YOLO model name."""
        model_name = str(
            config.get("models", {}).get("yolo", {}).get("model_name", DEFAULT_YOLO_MODEL)
        ).strip()
        if model_name not in SUPPORTED_YOLO_MODELS:
            supported = ", ".join(SUPPORTED_YOLO_MODELS)
            raise ValueError(
                f"Unsupported YOLO model '{model_name}'. Supported values: {supported}"
            )
        return model_name

    @staticmethod
    def _device_argument() -> int | str:
        """Return the device format expected by Ultralytics."""
        return 0 if torch.cuda.is_available() else "cpu"

    def build_optimizer(self) -> None:
        """YOLO manages its optimizer internally."""
        return None

    def build_scheduler(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        """YOLO manages its scheduler internally."""
        return None

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        device: torch.device,
        config: Dict[str, Any],
        logger,
    ) -> Optional[Path]:
        """Train YOLO using the Ultralytics training loop."""
        return fit_ultralytics(self, config, device, logger, YOLO)

    def predict_batch(
        self,
        images: List[Image.Image],
        device: torch.device,
        score_threshold: float,
    ) -> List[PredictionResult]:
        """Run inference on a batch of PIL images."""
        results = self.model.predict(
            source=images,
            conf=score_threshold,
            imgsz=int(get_training_value(self.config, self.config_key, "image_size")),
            max_det=int(self.config["evaluation"]["max_detections"]),
            device=(device.index or 0) if device.type == "cuda" else str(device),
            verbose=False,
        )
        predictions: List[PredictionResult] = []
        for result in results:
            if result.boxes is None:
                predictions.append(
                    PredictionResult(
                        boxes=torch.zeros((0, 4)),
                        scores=torch.zeros((0,)),
                        labels=torch.zeros((0,), dtype=torch.int64),
                    )
                )
                continue

            predictions.append(
                PredictionResult(
                    boxes=result.boxes.xyxy.detach().cpu(),
                    scores=result.boxes.conf.detach().cpu(),
                    labels=result.boxes.cls.detach().cpu().to(torch.int64) + 1,
                )
            )
        return predictions

    def num_parameters(self) -> int:
        """Count trainable parameters of the PyTorch YOLO model."""
        if hasattr(self, "parameter_count"):
            return self.parameter_count
        return sum(parameter.numel() for parameter in self.model.model.parameters())
