"""HuggingFace DETR wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

from models.common import PredictionResult
from utils.config import get_training_value
from utils.optimizer_factory import (
    build_torch_optimizer,
    build_torch_scheduler,
    get_optimizer_hyperparameters,
    get_scheduler_hyperparameters,
)
from utils.paths import get_model_checkpoints_dir


class DETRDetector:
    """Wrapper around HuggingFace DETR."""

    name = "DETR"
    config_key = "detr"
    architecture_type = "Transformer-based detector"
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
        id2label = {index: label for index, label in enumerate(class_names[1:])}
        label2id = {label: index for index, label in id2label.items()}
        image_size = int(get_training_value(config, self.config_key, "image_size"))
        self.image_processor = DetrImageProcessor.from_pretrained(
            "facebook/detr-resnet-50", size={"shortest_edge": image_size, "longest_edge": image_size}
        )
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes - 1,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )
        self._configure_backbone_trainability()
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
        base_learning_rate = float(get_training_value(self.config, self.config_key, "learning_rate"))
        backbone_learning_rate = float(
            get_training_value(self.config, self.config_key, "backbone_learning_rate", base_learning_rate)
        )
        backbone_parameters: list[torch.nn.Parameter] = []
        other_parameters: list[torch.nn.Parameter] = []

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._is_backbone_parameter(name):
                backbone_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        parameter_groups: list[dict[str, Any]] = []
        if other_parameters:
            parameter_groups.append({"params": other_parameters, "lr": base_learning_rate})
        if backbone_parameters:
            parameter_groups.append({"params": backbone_parameters, "lr": backbone_learning_rate})

        return build_torch_optimizer(
            parameters=parameter_groups,
            optimizer_name=optimizer_config["name"],
            learning_rate=base_learning_rate,
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

    @staticmethod
    def _is_backbone_parameter(parameter_name: str) -> bool:
        """Identify backbone parameters in the Hugging Face DETR model."""
        return "backbone" in parameter_name

    def _configure_backbone_trainability(self) -> None:
        """Optionally freeze the backbone during fine-tuning."""
        freeze_backbone = bool(get_training_value(self.config, self.config_key, "freeze_backbone", False))
        if not freeze_backbone:
            return
        for name, parameter in self.model.named_parameters():
            if self._is_backbone_parameter(name):
                parameter.requires_grad = False

    def prepare_batch(self, images: List[Image.Image], targets: List[Dict[str, Any]], device: torch.device):
        """Convert a batch into DETR processor inputs."""
        annotations: List[Dict[str, Any]] = []
        for target in targets:
            coco_annotations = []
            for box, label, area, iscrowd in zip(
                target["boxes_xywh"].tolist(),
                target["labels"].tolist(),
                target["area"].tolist(),
                target["iscrowd"].tolist(),
            ):
                coco_annotations.append(
                    {
                        "bbox": box,
                        "category_id": int(label) - 1,
                        "area": float(area),
                        "iscrowd": int(iscrowd),
                    }
                )
            annotations.append(
                {
                    "image_id": int(target["image_id"].item()),
                    "annotations": coco_annotations,
                }
            )

        encoded = self.image_processor(
            images=images,
            annotations=annotations,
            return_tensors="pt",
        )
        pixel_values = encoded["pixel_values"].to(device)
        pixel_mask = encoded["pixel_mask"].to(device)
        labels = [{key: value.to(device) for key, value in label.items()} for label in encoded["labels"]]
        return pixel_values, pixel_mask, labels

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
        self.model.eval()
        encoded = self.image_processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        pixel_mask = encoded["pixel_mask"].to(device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.tensor([[image.height, image.width] for image in images], device=device)
        processed = self.image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )

        predictions: List[PredictionResult] = []
        for output in processed:
            predictions.append(
                PredictionResult(
                    boxes=output["boxes"].detach().cpu(),
                    scores=output["scores"].detach().cpu(),
                    labels=(output["labels"] + 1).detach().cpu(),
                )
            )
        return predictions

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(parameter.numel() for parameter in self.model.parameters())
