"""Ultralytics YOLO wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import torch
from PIL import Image
from ultralytics import YOLO

from datasets.dataset_loader import build_yolo_data_config
from models.common import PredictionResult
from utils.config import get_training_value
from utils.optimizer_factory import get_optimizer_hyperparameters
from utils.paths import CHECKPOINTS_DIRNAME, get_model_results_dir

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
    uses_external_training_loop = True

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
        yolo_data_path = build_yolo_data_config(config=config, project_root=self.project_root)
        model_results_dir = get_model_results_dir(
            project_root=self.project_root,
            config=config,
            model_name=self.config_key,
        )
        results = self.model.train(
            data=str(yolo_data_path),
            model=self.model_name,
            epochs=int(get_training_value(config, self.config_key, "epochs")),
            imgsz=int(get_training_value(config, self.config_key, "image_size")),
            batch=int(get_training_value(config, self.config_key, "batch_size")),
            lr0=float(get_training_value(config, self.config_key, "learning_rate")),
            project=str(model_results_dir),
            name=CHECKPOINTS_DIRNAME,
            device=self._device_argument(),
            optimizer=self.optimizer_name,
            pretrained=True,
            exist_ok=True,
        )
        best_weights = Path(results.save_dir) / "weights" / "best.pt"
        if best_weights.exists():
            self.best_checkpoint_path = best_weights
            self.model = YOLO(str(best_weights))
        logger.info("YOLO base model: %s", self.model_name)
        logger.info("YOLO optimizer: %s", self.optimizer_name)
        logger.info("YOLO best checkpoint: %s", self.best_checkpoint_path)
        return self.best_checkpoint_path

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
            device=self._device_argument(),
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
        return sum(parameter.numel() for parameter in self.model.model.parameters() if parameter.requires_grad)
