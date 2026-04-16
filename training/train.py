"""Training loop implementations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import torch
from tqdm.auto import tqdm

from evaluation.evaluate import evaluate_model
from utils.config import get_training_value
from utils.paths import get_model_checkpoints_dir, get_model_results_dir
from utils.visualization import plot_training_history, save_prediction_gallery, save_training_history


def _extract_loss_components(loss_output) -> dict[str, float]:
    """Normalize framework-specific loss outputs into scalar logging values."""
    if loss_output is None:
        return {}

    if isinstance(loss_output, dict):
        return {
            key: float(value.detach().item())
            for key, value in loss_output.items()
            if torch.is_tensor(value) and value.numel() == 1
        }

    if hasattr(loss_output, "loss_dict"):
        return {
            key: float(value.detach().item())
            for key, value in loss_output.loss_dict.items()
            if torch.is_tensor(value) and value.numel() == 1
        }

    return {}


def _run_torch_epoch(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    logger=None,
    epoch: int = 0,
    total_epochs: int = 0,
    log_interval: int = 10,
) -> float:
    """Run one epoch for torchvision and DETR models."""
    model.model.train()
    total_loss = 0.0
    skipped_batches = 0
    component_totals: dict[str, float] = {}
    grad_clip_norm = float(get_training_value(config=model.config, model_name=model.config_key, key="grad_clip_norm", default=0.1))

    progress_bar = tqdm(
        enumerate(dataloader, start=1),
        total=len(dataloader),
        desc=f"Train {model.name} [{epoch}/{total_epochs}]",
        leave=False,
    )
    for batch_index, (images, targets) in progress_bar:
        optimizer.zero_grad()

        if model.config_key == "fasterrcnn":
            processed_images, processed_targets = model.prepare_batch(images, targets, device)
            loss_output = model.model(processed_images, processed_targets)
            loss = sum(loss_value for loss_value in loss_output.values())
        elif model.config_key == "detr":
            pixel_values, pixel_mask, labels = model.prepare_batch(images, targets, device)
            loss_output = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = loss_output.loss
        else:
            raise ValueError(f"Unsupported torch training loop for {model.name}")

        if not torch.isfinite(loss):
            skipped_batches += 1
            if logger is not None:
                logger.warning(
                    "Skipping non-finite loss for %s at epoch [%s/%s] batch [%s/%s]",
                    model.name,
                    epoch,
                    total_epochs,
                    batch_index,
                    len(dataloader),
                )
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=grad_clip_norm)
        if not torch.isfinite(grad_norm):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            if logger is not None:
                logger.warning(
                    "Skipping optimizer step due to non-finite gradient norm for %s at epoch [%s/%s] batch [%s/%s]",
                    model.name,
                    epoch,
                    total_epochs,
                    batch_index,
                    len(dataloader),
                )
            continue
        optimizer.step()
        batch_loss = float(loss.item())
        total_loss += batch_loss
        for key, value in _extract_loss_components(loss_output).items():
            component_totals[key] = component_totals.get(key, 0.0) + value

        running_loss = total_loss / batch_index
        progress_bar.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{running_loss:.4f}")

        if logger is not None and (
            batch_index == 1
            or batch_index % log_interval == 0
            or batch_index == len(dataloader)
        ):
            logger.info(
                "Epoch [%s/%s] train batch [%s/%s] loss=%.4f avg_loss=%.4f",
                epoch,
                total_epochs,
                batch_index,
                len(dataloader),
                batch_loss,
                running_loss,
            )

    processed_batches = max(len(dataloader) - skipped_batches, 1)
    if logger is not None and skipped_batches:
        logger.warning(
            "Skipped %s training batches for %s in epoch [%s/%s] due to non-finite values",
            skipped_batches,
            model.name,
            epoch,
            total_epochs,
        )
    return {
        "loss": total_loss / processed_batches,
        "components": {
            key: value / processed_batches
            for key, value in component_totals.items()
        },
    }


def _validate_torch_epoch(
    model,
    dataloader,
    device: torch.device,
    logger=None,
    epoch: int = 0,
    total_epochs: int = 0,
    log_interval: int = 10,
) -> float:
    """Compute validation loss."""
    if model.config_key == "fasterrcnn":
        # Torchvision detection models return losses only in train mode.
        model.model.train()
    else:
        model.model.eval()
    total_loss = 0.0
    component_totals: dict[str, float] = {}

    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader, start=1),
            total=len(dataloader),
            desc=f"Val {model.name} [{epoch}/{total_epochs}]",
            leave=False,
        )
        for batch_index, (images, targets) in progress_bar:
            if model.config_key == "fasterrcnn":
                processed_images, processed_targets = model.prepare_batch(images, targets, device)
                loss_output = model.model(processed_images, processed_targets)
                loss = sum(loss_value for loss_value in loss_output.values())
            elif model.config_key == "detr":
                pixel_values, pixel_mask, labels = model.prepare_batch(images, targets, device)
                loss_output = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = loss_output.loss
            else:
                raise ValueError(f"Unsupported torch validation loop for {model.name}")

            batch_loss = float(loss.item())
            total_loss += batch_loss
            for key, value in _extract_loss_components(loss_output).items():
                component_totals[key] = component_totals.get(key, 0.0) + value

            running_loss = total_loss / batch_index
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", avg=f"{running_loss:.4f}")

            if logger is not None and (
                batch_index == 1
                or batch_index % log_interval == 0
                or batch_index == len(dataloader)
            ):
                logger.info(
                    "Epoch [%s/%s] val batch [%s/%s] loss=%.4f avg_loss=%.4f",
                    epoch,
                    total_epochs,
                    batch_index,
                    len(dataloader),
                    batch_loss,
                    running_loss,
                )

    model.model.train()
    processed_batches = max(len(dataloader), 1)
    return {
        "loss": total_loss / processed_batches,
        "components": {
            key: value / processed_batches
            for key, value in component_totals.items()
        },
    }


def _save_checkpoint(model, checkpoint_path: Path, optimizer: Optional[torch.optim.Optimizer], epoch: int, loss: float) -> None:
    """Persist a training checkpoint."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "loss": loss,
        },
        checkpoint_path,
    )


def train_model(
    model,
    dataloader,
    optimizer,
    device: torch.device,
    val_dataloader=None,
    config: Optional[Dict[str, Any]] = None,
    logger=None,
):
    """Train the selected detector and save the best checkpoint."""
    if getattr(model, "uses_external_training_loop", False):
        return model.fit(
            train_dataloader=dataloader,
            val_dataloader=val_dataloader,
            device=device,
            config=config,
            logger=logger,
        )

    if config is None:
        raise ValueError("Configuration is required for training.")

    project_root = Path(__file__).resolve().parents[1]
    model_results_dir = get_model_results_dir(
        project_root=project_root,
        config=config,
        model_name=model.config_key,
    )
    checkpoint_dir = get_model_checkpoints_dir(
        project_root=project_root,
        config=config,
        model_name=model.config_key,
    )
    weights_dir = checkpoint_dir / "weights"
    best_checkpoint_path = weights_dir / "best.pt"
    last_checkpoint_path = weights_dir / "last.pt"
    results_csv_path = checkpoint_dir / "results.csv"
    history_dir = model_results_dir / "training_curves"
    history_csv_path = history_dir / f"{model.config_key}_history.csv"
    history_plot_path = history_dir / f"{model.config_key}_results.png"

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    epochs = int(get_training_value(config, model.config_key, "epochs"))
    log_interval = int(get_training_value(config, model.config_key, "log_interval", 10))
    scheduler = model.build_scheduler(optimizer) if optimizer is not None else None
    for epoch in range(epochs):
        epoch_start = perf_counter()
        if logger is not None:
            logger.info("Starting epoch [%s/%s] for %s", epoch + 1, epochs, model.name)

        train_summary = _run_torch_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            logger=logger,
            epoch=epoch + 1,
            total_epochs=epochs,
            log_interval=log_interval,
        )
        val_summary = _validate_torch_epoch(
            model=model,
            dataloader=val_dataloader,
            device=device,
            logger=logger,
            epoch=epoch + 1,
            total_epochs=epochs,
            log_interval=log_interval,
        )
        train_loss = float(train_summary["loss"])
        val_loss = float(val_summary["loss"])
        if scheduler is not None:
            scheduler.step()
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            device=device,
            config=config,
            logger=None,
        )
        epoch_history = {
            "epoch": float(epoch + 1),
            "time": perf_counter() - epoch_start,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": val_metrics["Precision"],
            "recall": val_metrics["Recall"],
            "f1": val_metrics["F1"],
            "map50": val_metrics["mAP50"],
            "map50_95": val_metrics["mAP50_95"],
        }
        for key, value in train_summary["components"].items():
            epoch_history[f"train_{key}"] = value
        for key, value in val_summary["components"].items():
            epoch_history[f"val_{key}"] = value
        if scheduler is not None:
            epoch_history["learning_rate"] = float(scheduler.get_last_lr()[0])
        history.append(epoch_history)

        _save_checkpoint(model, last_checkpoint_path, optimizer, epoch, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, best_checkpoint_path, optimizer, epoch, val_loss)
            if logger is not None:
                logger.info("New best checkpoint saved to %s", best_checkpoint_path)

        save_training_history(
            history=[
                {
                    "epoch": row["epoch"],
                    "time": row["time"],
                    "train/loss": row["train_loss"],
                    "val/loss": row["val_loss"],
                    "metrics/precision(B)": row["precision"],
                    "metrics/recall(B)": row["recall"],
                    "metrics/F1(B)": row["f1"],
                    "metrics/mAP50(B)": row["map50"],
                    "metrics/mAP50-95(B)": row["map50_95"],
                    **{
                        key.replace("train_", "train/"): value
                        for key, value in row.items()
                        if key.startswith("train_") and key != "train_loss"
                    },
                    **{
                        key.replace("val_", "val/"): value
                        for key, value in row.items()
                        if key.startswith("val_") and key != "val_loss"
                    },
                    **(
                        {"lr/pg0": row["learning_rate"]}
                        if "learning_rate" in row
                        else {}
                    ),
                }
                for row in history
            ],
            output_path=results_csv_path,
        )

        if logger is not None:
            logger.info(
                "Epoch [%s/%s] train_loss=%.4f val_loss=%.4f precision=%.4f recall=%.4f mAP50=%.4f mAP50_95=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                val_metrics["Precision"],
                val_metrics["Recall"],
                val_metrics["mAP50"],
                val_metrics["mAP50_95"],
            )
            if train_summary["components"] or val_summary["components"]:
                logger.info(
                    "Epoch [%s/%s] train_components=%s val_components=%s",
                    epoch + 1,
                    epochs,
                    train_summary["components"],
                    val_summary["components"],
                )
            logger.info("Updated epoch results CSV at %s", results_csv_path)

    history_df = save_training_history(history=history, output_path=history_csv_path)
    plot_training_history(
        history_df=history_df,
        output_path=history_plot_path,
        title=f"{model.name} Training Summary",
    )
    if logger is not None:
        logger.info("Saved training history to %s", history_csv_path)
        logger.info("Saved training plot to %s", history_plot_path)

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])
    qualitative_path = history_dir / f"{model.config_key}_qualitative_predictions.png"
    save_prediction_gallery(
        model=model,
        dataloader=val_dataloader,
        device=device,
        output_path=qualitative_path,
        score_threshold=float(config["evaluation"]["score_threshold"]),
        class_names=model.class_names,
        max_images=int(get_training_value(config, model.config_key, "qualitative_images", 4)),
    )
    if logger is not None:
        logger.info("Saved qualitative predictions to %s", qualitative_path)
    return best_checkpoint_path
