"""Torchvision Faster R-CNN wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from models.common import PredictionResult
from utils.config import get_training_value
from utils.optimizer_factory import (
    build_torch_optimizer,
    build_torch_scheduler,
    get_optimizer_hyperparameters,
    get_scheduler_hyperparameters,
)
from utils.paths import get_model_checkpoints_dir


class FasterRCNNDetector:
    """Wrapper around torchvision Faster R-CNN."""

    name = "Faster R-CNN"
    config_key = "fasterrcnn"
    architecture_type = "Two-stage detector"
    is_ultralytics = False

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        device: torch.device,
        config: Dict[str, Any],
    ) -> None:
        self.class_names = class_names
        self.device = device
        self.config = config
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model.to(device)
        checkpoints_dir = get_model_checkpoints_dir(
            project_root=Path(__file__).resolve().parents[1],
            config=config,
            model_name=self.config_key,
        )
        self.best_checkpoint_path = (
            checkpoints_dir
            / "weights"
            / "best.pt"
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        """Create the optimizer from the shared experiment config."""
        optimizer_config = get_optimizer_hyperparameters(self.config, self.config_key)
        return build_torch_optimizer(
            parameters=self.model.parameters(),
            optimizer_name=optimizer_config["name"],
            learning_rate=float(get_training_value(self.config, self.config_key, "learning_rate")),
            weight_decay=optimizer_config["weight_decay"],
            momentum=optimizer_config["momentum"],
        )

    def build_scheduler(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """Create an optional scheduler from the shared experiment config."""
        if optimizer is None:
            return None
        scheduler_config = get_scheduler_hyperparameters(self.config, self.config_key)
        return build_torch_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_config["name"],
            milestones=scheduler_config["milestones"],
            gamma=scheduler_config["gamma"],
        )

    def prepare_batch(self, images: List[Image.Image], targets: List[Dict[str, Any]], device: torch.device):
        """Convert a batch into torchvision detection inputs."""
        processed_images: List[torch.Tensor] = []
        processed_targets: List[Dict[str, torch.Tensor]] = []

        image_size = int(get_training_value(self.config, self.config_key, "image_size"))
        for image, target in zip(images, targets):
            orig_w, orig_h = image.size
            resized_image = F.resize(image, [image_size, image_size])
            image_tensor = F.to_tensor(resized_image)
            image_tensor = F.normalize(
                image_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            boxes = target["boxes_xywh"].clone()
            if boxes.numel() > 0:
                boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                scale_x = image_size / orig_w
                scale_y = image_size / orig_h
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
            else:
                boxes = boxes.reshape(0, 4)

            processed_targets.append(
                {
                    "boxes": boxes.to(device),
                    "labels": target["labels"].to(device),
                    "image_id": target["image_id"].to(device),
                    "area": target["area"].to(device),
                    "iscrowd": target["iscrowd"].to(device),
                }
            )
            processed_images.append(image_tensor.to(device))

        return processed_images, processed_targets

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        device: torch.device,
        config: Dict[str, Any],
        logger,
    ) -> Optional[Path]:
        """Train the model using the shared pipeline."""
        return None

    def predict_batch(
        self,
        images: List[Image.Image],
        device: torch.device,
        score_threshold: float,
    ) -> List[PredictionResult]:
        """Run inference on a batch."""
        dummy_targets = [
            {
                "boxes_xywh": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
            for idx in range(len(images))
        ]
        processed_images, _ = self.prepare_batch(images=images, targets=dummy_targets, device=device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(processed_images)

        predictions: List[PredictionResult] = []
        image_size = int(get_training_value(self.config, self.config_key, "image_size"))
        for image, output in zip(images, outputs):
            keep = output["scores"] >= score_threshold
            boxes = output["boxes"][keep].detach().cpu()
            orig_w, orig_h = image.size
            if boxes.numel() > 0:
                boxes[:, [0, 2]] *= orig_w / image_size
                boxes[:, [1, 3]] *= orig_h / image_size
            predictions.append(
                PredictionResult(
                    boxes=boxes,
                    scores=output["scores"][keep].detach().cpu(),
                    labels=output["labels"][keep].detach().cpu(),
                )
            )
        return predictions

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
