"""Optimizer and scheduler configuration helpers for the supported detectors."""

from __future__ import annotations

from typing import Final, Iterable

import torch

from utils.config import get_training_value

SUPPORTED_OPTIMIZERS: Final[tuple[str, ...]] = (
    "SGD",
    "Adam",
    "AdamW",
    "RMSProp",
)
SUPPORTED_SCHEDULERS: Final[tuple[str, ...]] = (
    "none",
    "MultiStepLR",
)
DEFAULT_OPTIMIZER: Final[str] = "AdamW"
DEFAULT_SCHEDULER: Final[str] = "none"


def normalize_optimizer_name(optimizer_name: str | None) -> str:
    """Normalize and validate an optimizer name from the config."""
    normalized = (optimizer_name or DEFAULT_OPTIMIZER).strip()
    aliases = {
        "rmsprop": "RMSProp",
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
    }
    normalized = aliases.get(normalized.lower(), normalized)
    if normalized not in SUPPORTED_OPTIMIZERS:
        supported = ", ".join(SUPPORTED_OPTIMIZERS)
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported values: {supported}")
    return normalized


def get_optimizer_name(config: dict, model_name: str) -> str:
    """Read the optimizer choice from the model training config."""
    return normalize_optimizer_name(
        get_training_value(config, model_name, "optimizer", DEFAULT_OPTIMIZER)
    )


def normalize_scheduler_name(scheduler_name: str | None) -> str:
    """Normalize and validate a scheduler name from the config."""
    normalized = (scheduler_name or DEFAULT_SCHEDULER).strip()
    aliases = {
        "none": "none",
        "multisteplr": "MultiStepLR",
    }
    normalized = aliases.get(normalized.lower(), normalized)
    if normalized not in SUPPORTED_SCHEDULERS:
        supported = ", ".join(SUPPORTED_SCHEDULERS)
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Supported values: {supported}")
    return normalized


def get_scheduler_name(config: dict, model_name: str) -> str:
    """Read the scheduler choice from the model training config."""
    return normalize_scheduler_name(
        get_training_value(config, model_name, "scheduler", DEFAULT_SCHEDULER)
    )


def get_optimizer_hyperparameters(config: dict, model_name: str) -> dict:
    """Return resolved optimizer hyperparameters for a model."""
    return {
        "name": get_optimizer_name(config, model_name),
        "weight_decay": float(get_training_value(config, model_name, "weight_decay", 0.0)),
        "momentum": float(get_training_value(config, model_name, "momentum", 0.9)),
    }


def get_scheduler_hyperparameters(config: dict, model_name: str) -> dict:
    """Return resolved scheduler hyperparameters for a model."""
    milestones = get_training_value(config, model_name, "scheduler_milestones", [])
    return {
        "name": get_scheduler_name(config, model_name),
        "milestones": [int(milestone) for milestone in milestones],
        "gamma": float(get_training_value(config, model_name, "scheduler_gamma", 0.1)),
    }


def build_torch_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """Build a PyTorch optimizer from the shared config."""
    name = normalize_optimizer_name(optimizer_name)
    if name == "SGD":
        return torch.optim.SGD(
            params=parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if name == "Adam":
        return torch.optim.Adam(
            params=parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "AdamW":
        return torch.optim.AdamW(
            params=parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "RMSProp":
        return torch.optim.RMSprop(
            params=parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def build_torch_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    milestones: list[int],
    gamma: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a PyTorch learning-rate scheduler from the shared config."""
    name = normalize_scheduler_name(scheduler_name)
    if name == "none":
        return None
    if name == "MultiStepLR":
        if not milestones:
            raise ValueError("scheduler_milestones must be provided when using MultiStepLR")
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    raise ValueError(f"Unsupported scheduler '{scheduler_name}'")
