"""Configuration helpers for shared and model-specific settings."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
            continue
        merged[key] = deepcopy(value)
    return merged


def get_model_key(model_name: str) -> str:
    """Map CLI model names and display names to config keys."""
    normalized = model_name.strip().lower().replace(" ", "").replace("-", "")
    aliases = {
        "yolo": "yolo",
        "rtdetr": "rtdetr",
        "fasterrcnn": "fasterrcnn",
        "detr": "detr",
        "cascadercnn": "cascadercnn",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model name '{model_name}'")
    return aliases[normalized]


def get_model_section(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Return the raw model section from the config."""
    return config.get("models", {}).get(get_model_key(model_name), {})


def get_training_config(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Return training config with model-specific overrides applied."""
    merged = deepcopy(config["training"])
    overrides = get_model_section(config, model_name).get("training", {})
    return deep_merge_dicts(merged, overrides)


def get_training_value(config: Dict[str, Any], model_name: str, key: str, default: Any = None) -> Any:
    """Read a resolved training value for a given model."""
    training_config = get_training_config(config, model_name)
    if default is None:
        return training_config[key]
    return training_config.get(key, default)
