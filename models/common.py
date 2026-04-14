"""Common interfaces and helper functions for detector wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch


@dataclass
class PredictionResult:
    """Unified prediction structure for downstream metric computation."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class DetectorProtocol(Protocol):
    """Shared interface expected by the training and evaluation pipeline."""

    name: str
    config_key: str
    architecture_type: str
    is_ultralytics: bool

    def build_optimizer(self) -> Optional[torch.optim.Optimizer]:
        ...

    def build_scheduler(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        ...

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        device: torch.device,
        config: Dict[str, Any],
        logger,
    ) -> Optional[Path]:
        ...

    def predict_batch(
        self,
        images: List[Any],
        device: torch.device,
        score_threshold: float,
    ) -> List[PredictionResult]:
        ...

    def num_parameters(self) -> int:
        ...
