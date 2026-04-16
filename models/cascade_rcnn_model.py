"""MMDetection Cascade R-CNN wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import numpy as np
import torch
from PIL import Image

from models.common import PredictionResult
from utils.config import get_training_value
from utils.optimizer_factory import (
    get_optimizer_hyperparameters,
    get_scheduler_hyperparameters,
)
from utils.paths import get_model_checkpoints_dir

SUPPORTED_CASCADERCNN_MODELS: Final[tuple[str, ...]] = (
    "cascade_rcnn_r50_fpn",
)
DEFAULT_CASCADERCNN_MODEL: Final[str] = "cascade_rcnn_r50_fpn"


class CascadeRCNNDetector:
    """Wrapper around MMDetection Cascade R-CNN."""

    config_key = "cascadercnn"
    architecture_type = "Two-stage detector"
    is_ultralytics = False
    uses_external_training_loop = True

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        device: torch.device,
        config: Dict[str, Any],
    ) -> None:
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.config = config
        self.project_root = Path(__file__).resolve().parents[1]
        self.model_name = self._resolve_model_name(config)
        self.name = f"Cascade R-CNN ({self.model_name})"
        self.checkpoints_dir = get_model_checkpoints_dir(
            project_root=self.project_root,
            config=config,
            model_name=self.config_key,
        )
        self.config_path = self.checkpoints_dir / "cascade_rcnn_generated.py"
        self.best_checkpoint_path: Optional[Path] = None
        self.model = None
        self.model_cfg_dict = self._build_mmdet_config()
        self.model_cfg = self._as_config(self.model_cfg_dict)
        self._dump_generated_config()
        self.model = self._build_inference_model(checkpoint_path=None)

    @staticmethod
    def _resolve_model_name(config: Dict[str, Any]) -> str:
        """Read and validate the configured Cascade R-CNN variant."""
        model_name = str(
            config.get("models", {}).get("cascadercnn", {}).get("model_name", DEFAULT_CASCADERCNN_MODEL)
        ).strip()
        if model_name not in SUPPORTED_CASCADERCNN_MODELS:
            supported = ", ".join(SUPPORTED_CASCADERCNN_MODELS)
            raise ValueError(
                f"Unsupported Cascade R-CNN model '{model_name}'. Supported values: {supported}"
            )
        return model_name

    @staticmethod
    def _device_argument() -> str:
        """Return the device format expected by MMDetection."""
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _normalize_mmdet_optimizer_name(optimizer_name: str) -> str:
        """Map shared optimizer names to MMEngine registry names."""
        aliases = {
            "RMSProp": "RMSprop",
        }
        return aliases.get(optimizer_name, optimizer_name)

    @staticmethod
    def _import_mmdet_dependencies():
        """Import MMDetection dependencies lazily with a clear error message."""
        try:
            from mmcv.transforms import Compose
            from mmengine.config import Config
            from mmengine.runner import Runner
            from mmdet.apis import inference_detector, init_detector
            from mmdet.utils import register_all_modules
        except ImportError as exc:
            raise ImportError(
                "Cascade R-CNN requires MMDetection. Install it with "
                "`pip install -U openmim`, `mim install mmengine`, "
                "`mim install \"mmcv>=2.0.0\"`, and `mim install mmdet`."
            ) from exc
        return Compose, Config, Runner, inference_detector, init_detector, register_all_modules

    def build_optimizer(self) -> None:
        """MMDetection manages the optimizer internally."""
        return None

    def build_scheduler(
        self,
        optimizer: Optional[torch.optim.Optimizer],
    ) -> None:
        """MMDetection manages the scheduler internally."""
        return None

    def _dump_generated_config(self) -> None:
        """Persist the generated MMDetection config for reproducibility."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.model_cfg.dump(self.config_path)

    def _as_config(self, config_dict: dict[str, Any]):
        """Convert a Python dictionary into an MMEngine Config object."""
        _, Config, _, _, _, _ = self._import_mmdet_dependencies()
        return Config(config_dict)

    def _build_dataset_config(self, split: str, pipeline: list[dict[str, Any]], test_mode: bool) -> dict[str, Any]:
        """Build a COCO dataset config for a split."""
        dataset_root = (self.project_root / self.config["dataset"]["root"]).resolve()
        images_key = f"{split}_images"
        annotations_key = f"{split}_annotations"
        class_names = tuple(self.class_names[1:])
        num_workers = int(get_training_value(self.config, self.config_key, "num_workers", 2))
        persistent_workers = num_workers > 0
        batch_size = (
            int(get_training_value(self.config, self.config_key, "eval_batch_size", 1))
            if test_mode
            else int(get_training_value(self.config, self.config_key, "batch_size"))
        )
        dataset_cfg = {
            "type": "CocoDataset",
            "metainfo": {"classes": class_names},
            "data_root": str(dataset_root),
            "ann_file": str(self.config["dataset"][annotations_key]),
            "data_prefix": {"img": f"{self.config['dataset'][images_key].rstrip('/')}/"},
            "pipeline": pipeline,
            "backend_args": None,
            "test_mode": test_mode,
        }
        if not test_mode:
            dataset_cfg["filter_cfg"] = {"filter_empty_gt": True, "min_size": 32}

        dataloader_cfg = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "sampler": {
                "type": "DefaultSampler",
                "shuffle": not test_mode,
            },
            "dataset": dataset_cfg,
        }
        if not test_mode:
            dataloader_cfg["batch_sampler"] = {"type": "AspectRatioBatchSampler"}
        else:
            dataloader_cfg["drop_last"] = False
        return dataloader_cfg

    def _build_param_scheduler(self, epochs: int) -> list[dict[str, Any]]:
        """Build MMEngine scheduler config from the shared training settings."""
        scheduler_config = get_scheduler_hyperparameters(self.config, self.config_key)
        scheduler_name = scheduler_config["name"]
        warmup_iters = int(get_training_value(self.config, self.config_key, "warmup_iters", 0))
        warmup_factor = float(get_training_value(self.config, self.config_key, "warmup_factor", 0.001))
        schedulers: list[dict[str, Any]] = []

        if warmup_iters > 0:
            schedulers.append(
                {
                    "type": "LinearLR",
                    "start_factor": warmup_factor,
                    "by_epoch": False,
                    "begin": 0,
                    "end": warmup_iters,
                }
            )

        if scheduler_name == "MultiStepLR":
            schedulers.append(
                {
                    "type": "MultiStepLR",
                    "begin": 0,
                    "end": epochs,
                    "by_epoch": True,
                    "milestones": scheduler_config["milestones"],
                    "gamma": scheduler_config["gamma"],
                }
            )

        return schedulers

    def _build_mmdet_config(self) -> dict[str, Any]:
        """Create a self-contained MMDetection config for Cascade R-CNN."""
        dataset_root = (self.project_root / self.config["dataset"]["root"]).resolve()
        short_edge = int(get_training_value(self.config, self.config_key, "image_size"))
        max_edge = int(get_training_value(self.config, self.config_key, "max_image_size", 1333))
        epochs = int(get_training_value(self.config, self.config_key, "epochs"))
        num_classes = self.num_classes - 1
        optimizer_config = get_optimizer_hyperparameters(self.config, self.config_key)
        optimizer_name = self._normalize_mmdet_optimizer_name(optimizer_config["name"])
        learning_rate = float(get_training_value(self.config, self.config_key, "learning_rate"))
        log_interval = int(get_training_value(self.config, self.config_key, "log_interval", 10))
        checkpoint_interval = int(get_training_value(self.config, self.config_key, "checkpoint_interval", 1))
        max_keep_ckpts = int(get_training_value(self.config, self.config_key, "max_keep_checkpoints", 2))
        load_from = self.config.get("models", {}).get("cascadercnn", {}).get("checkpoint")
        score_threshold = float(self.config["evaluation"]["score_threshold"])
        max_detections = int(self.config["evaluation"]["max_detections"])

        train_pipeline = [
            {"type": "LoadImageFromFile", "backend_args": None},
            {"type": "LoadAnnotations", "with_bbox": True},
            {"type": "Resize", "scale": (max_edge, short_edge), "keep_ratio": True},
            {"type": "RandomFlip", "prob": 0.5},
            {"type": "PackDetInputs"},
        ]
        eval_pipeline = [
            {"type": "LoadImageFromFile", "backend_args": None},
            {"type": "Resize", "scale": (max_edge, short_edge), "keep_ratio": True},
            {"type": "LoadAnnotations", "with_bbox": True},
            {
                "type": "PackDetInputs",
                "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
            },
        ]

        optimizer = {
            "type": optimizer_name,
            "lr": learning_rate,
            "weight_decay": optimizer_config["weight_decay"],
        }
        if optimizer_name in {"SGD", "RMSprop"}:
            optimizer["momentum"] = optimizer_config["momentum"]

        val_ann_file = str(dataset_root / self.config["dataset"]["val_annotations"])
        test_ann_file = str(dataset_root / self.config["dataset"]["test_annotations"])

        return {
            "default_scope": "mmdet",
            "work_dir": str(self.checkpoints_dir),
            "launcher": "none",
            "resume": False,
            "load_from": str(load_from) if load_from else None,
            "env_cfg": {
                "cudnn_benchmark": False,
                "mp_cfg": {"mp_start_method": "fork", "opencv_num_threads": 0},
                "dist_cfg": {"backend": "nccl"},
            },
            "randomness": {"seed": int(self.config["experiment"]["seed"])},
            "log_level": "INFO",
            "log_processor": {"type": "LogProcessor", "window_size": 50, "by_epoch": True},
            "vis_backends": [{"type": "LocalVisBackend"}],
            "visualizer": {
                "type": "DetLocalVisualizer",
                "vis_backends": [{"type": "LocalVisBackend"}],
                "name": "visualizer",
            },
            "default_hooks": {
                "timer": {"type": "IterTimerHook"},
                "logger": {"type": "LoggerHook", "interval": log_interval},
                "param_scheduler": {"type": "ParamSchedulerHook"},
                "checkpoint": {
                    "type": "CheckpointHook",
                    "interval": checkpoint_interval,
                    "save_best": "coco/bbox_mAP",
                    "rule": "greater",
                    "max_keep_ckpts": max_keep_ckpts,
                },
                "sampler_seed": {"type": "DistSamplerSeedHook"},
                "visualization": {"type": "DetVisualizationHook"},
            },
            "model": {
                "type": "CascadeRCNN",
                "data_preprocessor": {
                    "type": "DetDataPreprocessor",
                    "mean": [123.675, 116.28, 103.53],
                    "std": [58.395, 57.12, 57.375],
                    "bgr_to_rgb": True,
                    "pad_size_divisor": 32,
                },
                "backbone": {
                    "type": "ResNet",
                    "depth": 50,
                    "num_stages": 4,
                    "out_indices": (0, 1, 2, 3),
                    "frozen_stages": 1,
                    "norm_cfg": {"type": "BN", "requires_grad": True},
                    "norm_eval": True,
                    "style": "pytorch",
                    "init_cfg": {"type": "Pretrained", "checkpoint": "torchvision://resnet50"},
                },
                "neck": {
                    "type": "FPN",
                    "in_channels": [256, 512, 1024, 2048],
                    "out_channels": 256,
                    "num_outs": 5,
                },
                "rpn_head": {
                    "type": "RPNHead",
                    "in_channels": 256,
                    "feat_channels": 256,
                    "anchor_generator": {
                        "type": "AnchorGenerator",
                        "scales": [8],
                        "ratios": [0.5, 1.0, 2.0],
                        "strides": [4, 8, 16, 32, 64],
                    },
                    "bbox_coder": {
                        "type": "DeltaXYWHBBoxCoder",
                        "target_means": [0.0, 0.0, 0.0, 0.0],
                        "target_stds": [1.0, 1.0, 1.0, 1.0],
                    },
                    "loss_cls": {
                        "type": "CrossEntropyLoss",
                        "use_sigmoid": True,
                        "loss_weight": 1.0,
                    },
                    "loss_bbox": {
                        "type": "SmoothL1Loss",
                        "beta": 1.0 / 9.0,
                        "loss_weight": 1.0,
                    },
                },
                "roi_head": {
                    "type": "CascadeRoIHead",
                    "num_stages": 3,
                    "stage_loss_weights": [1.0, 0.5, 0.25],
                    "bbox_roi_extractor": {
                        "type": "SingleRoIExtractor",
                        "roi_layer": {"type": "RoIAlign", "output_size": 7, "sampling_ratio": 0},
                        "out_channels": 256,
                        "featmap_strides": [4, 8, 16, 32],
                    },
                    "bbox_head": [
                        {
                            "type": "Shared2FCBBoxHead",
                            "in_channels": 256,
                            "fc_out_channels": 1024,
                            "roi_feat_size": 7,
                            "num_classes": num_classes,
                            "bbox_coder": {
                                "type": "DeltaXYWHBBoxCoder",
                                "target_means": [0.0, 0.0, 0.0, 0.0],
                                "target_stds": [0.1, 0.1, 0.2, 0.2],
                            },
                            "reg_class_agnostic": True,
                            "loss_cls": {
                                "type": "CrossEntropyLoss",
                                "use_sigmoid": False,
                                "loss_weight": 1.0,
                            },
                            "loss_bbox": {"type": "SmoothL1Loss", "beta": 1.0, "loss_weight": 1.0},
                        },
                        {
                            "type": "Shared2FCBBoxHead",
                            "in_channels": 256,
                            "fc_out_channels": 1024,
                            "roi_feat_size": 7,
                            "num_classes": num_classes,
                            "bbox_coder": {
                                "type": "DeltaXYWHBBoxCoder",
                                "target_means": [0.0, 0.0, 0.0, 0.0],
                                "target_stds": [0.05, 0.05, 0.1, 0.1],
                            },
                            "reg_class_agnostic": True,
                            "loss_cls": {
                                "type": "CrossEntropyLoss",
                                "use_sigmoid": False,
                                "loss_weight": 1.0,
                            },
                            "loss_bbox": {"type": "SmoothL1Loss", "beta": 1.0, "loss_weight": 1.0},
                        },
                        {
                            "type": "Shared2FCBBoxHead",
                            "in_channels": 256,
                            "fc_out_channels": 1024,
                            "roi_feat_size": 7,
                            "num_classes": num_classes,
                            "bbox_coder": {
                                "type": "DeltaXYWHBBoxCoder",
                                "target_means": [0.0, 0.0, 0.0, 0.0],
                                "target_stds": [0.033, 0.033, 0.067, 0.067],
                            },
                            "reg_class_agnostic": True,
                            "loss_cls": {
                                "type": "CrossEntropyLoss",
                                "use_sigmoid": False,
                                "loss_weight": 1.0,
                            },
                            "loss_bbox": {"type": "SmoothL1Loss", "beta": 1.0, "loss_weight": 1.0},
                        },
                    ],
                },
                "train_cfg": {
                    "rpn": {
                        "assigner": {
                            "type": "MaxIoUAssigner",
                            "pos_iou_thr": 0.7,
                            "neg_iou_thr": 0.3,
                            "min_pos_iou": 0.3,
                            "match_low_quality": True,
                            "ignore_iof_thr": -1,
                        },
                        "sampler": {
                            "type": "RandomSampler",
                            "num": 256,
                            "pos_fraction": 0.5,
                            "neg_pos_ub": -1,
                            "add_gt_as_proposals": False,
                        },
                        "allowed_border": 0,
                        "pos_weight": -1,
                        "debug": False,
                    },
                    "rpn_proposal": {
                        "nms_pre": 2000,
                        "max_per_img": 2000,
                        "nms": {"type": "nms", "iou_threshold": 0.7},
                        "min_bbox_size": 0,
                    },
                    "rcnn": [
                        {
                            "assigner": {
                                "type": "MaxIoUAssigner",
                                "pos_iou_thr": 0.5,
                                "neg_iou_thr": 0.5,
                                "min_pos_iou": 0.5,
                                "match_low_quality": False,
                                "ignore_iof_thr": -1,
                            },
                            "sampler": {
                                "type": "RandomSampler",
                                "num": 512,
                                "pos_fraction": 0.25,
                                "neg_pos_ub": -1,
                                "add_gt_as_proposals": True,
                            },
                            "pos_weight": -1,
                            "debug": False,
                        },
                        {
                            "assigner": {
                                "type": "MaxIoUAssigner",
                                "pos_iou_thr": 0.6,
                                "neg_iou_thr": 0.6,
                                "min_pos_iou": 0.6,
                                "match_low_quality": False,
                                "ignore_iof_thr": -1,
                            },
                            "sampler": {
                                "type": "RandomSampler",
                                "num": 512,
                                "pos_fraction": 0.25,
                                "neg_pos_ub": -1,
                                "add_gt_as_proposals": True,
                            },
                            "pos_weight": -1,
                            "debug": False,
                        },
                        {
                            "assigner": {
                                "type": "MaxIoUAssigner",
                                "pos_iou_thr": 0.7,
                                "neg_iou_thr": 0.7,
                                "min_pos_iou": 0.7,
                                "match_low_quality": False,
                                "ignore_iof_thr": -1,
                            },
                            "sampler": {
                                "type": "RandomSampler",
                                "num": 512,
                                "pos_fraction": 0.25,
                                "neg_pos_ub": -1,
                                "add_gt_as_proposals": True,
                            },
                            "pos_weight": -1,
                            "debug": False,
                        },
                    ],
                },
                "test_cfg": {
                    "rpn": {
                        "nms_pre": 1000,
                        "max_per_img": 1000,
                        "nms": {"type": "nms", "iou_threshold": 0.7},
                        "min_bbox_size": 0,
                    },
                    "rcnn": {
                        "score_thr": score_threshold,
                        "nms": {"type": "nms", "iou_threshold": 0.5},
                        "max_per_img": max_detections,
                    },
                },
            },
            "train_dataloader": self._build_dataset_config(
                split="train",
                pipeline=train_pipeline,
                test_mode=False,
            ),
            "val_dataloader": self._build_dataset_config(
                split="val",
                pipeline=eval_pipeline,
                test_mode=True,
            ),
            "test_dataloader": self._build_dataset_config(
                split="test",
                pipeline=eval_pipeline,
                test_mode=True,
            ),
            "val_evaluator": {
                "type": "CocoMetric",
                "ann_file": val_ann_file,
                "metric": "bbox",
                "format_only": False,
                "backend_args": None,
            },
            "test_evaluator": {
                "type": "CocoMetric",
                "ann_file": test_ann_file,
                "metric": "bbox",
                "format_only": False,
                "backend_args": None,
            },
            "train_cfg": {"type": "EpochBasedTrainLoop", "max_epochs": epochs, "val_interval": 1},
            "val_cfg": {"type": "ValLoop"},
            "test_cfg": {"type": "TestLoop"},
            "optim_wrapper": {
                "type": "OptimWrapper",
                "optimizer": optimizer,
            },
            "param_scheduler": self._build_param_scheduler(epochs=epochs),
            "auto_scale_lr": {"enable": False, "base_batch_size": 16},
        }

    def _build_inference_model(self, checkpoint_path: Optional[Path]):
        """Create or reload the MMDetection model used for evaluation."""
        _, _, _, _, init_detector, register_all_modules = self._import_mmdet_dependencies()
        register_all_modules()
        return init_detector(
            self.model_cfg,
            checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
            device=self._device_argument(),
        )

    def _build_inference_pipeline(self):
        """Build an ndarray-based pipeline for predict_batch."""
        Compose, _, _, _, _, register_all_modules = self._import_mmdet_dependencies()
        register_all_modules()
        short_edge = int(get_training_value(self.config, self.config_key, "image_size"))
        max_edge = int(get_training_value(self.config, self.config_key, "max_image_size", 1333))
        return Compose(
            [
                {"type": "mmdet.LoadImageFromNDArray"},
                {"type": "Resize", "scale": (max_edge, short_edge), "keep_ratio": True},
                {
                    "type": "PackDetInputs",
                    "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
                },
            ]
        )

    def _resolve_best_checkpoint_path(self) -> Optional[Path]:
        """Locate the best checkpoint emitted by MMDetection."""
        candidates = sorted(
            self.checkpoints_dir.glob("best*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]

        latest_checkpoint = self.checkpoints_dir / "latest.pth"
        if latest_checkpoint.exists():
            return latest_checkpoint

        epoch_checkpoints = sorted(
            self.checkpoints_dir.glob("epoch_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if epoch_checkpoints:
            return epoch_checkpoints[0]
        return None

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        device: torch.device,
        config: Dict[str, Any],
        logger,
    ) -> Optional[Path]:
        """Train Cascade R-CNN using the MMEngine runner."""
        _, Config, Runner, _, _, register_all_modules = self._import_mmdet_dependencies()
        register_all_modules()
        cfg = Config(self.model_cfg.to_dict())
        runner = Runner.from_cfg(cfg)
        runner.train()

        best_checkpoint_path = self._resolve_best_checkpoint_path()
        if best_checkpoint_path is None:
            raise FileNotFoundError(
                f"MMDetection training finished but no checkpoint was found in {self.checkpoints_dir}"
            )

        self.best_checkpoint_path = best_checkpoint_path
        self.model = self._build_inference_model(checkpoint_path=best_checkpoint_path)
        if logger is not None:
            logger.info("Cascade R-CNN MMDetection config: %s", self.config_path)
            logger.info("Cascade R-CNN best checkpoint: %s", best_checkpoint_path)
        return best_checkpoint_path

    def predict_batch(
        self,
        images: List[Image.Image],
        device: torch.device,
        score_threshold: float,
    ) -> List[PredictionResult]:
        """Run inference on a batch of PIL images."""
        _, _, _, inference_detector, _, _ = self._import_mmdet_dependencies()
        if self.model is None:
            self.model = self._build_inference_model(checkpoint_path=self.best_checkpoint_path)

        bgr_images = [np.asarray(image)[:, :, ::-1].copy() for image in images]
        inference_pipeline = self._build_inference_pipeline()
        results = inference_detector(self.model, bgr_images, test_pipeline=inference_pipeline)
        if not isinstance(results, list):
            results = [results]

        predictions: List[PredictionResult] = []
        for result in results:
            pred_instances = result.pred_instances
            keep = pred_instances.scores >= score_threshold
            predictions.append(
                PredictionResult(
                    boxes=pred_instances.bboxes[keep].detach().cpu(),
                    scores=pred_instances.scores[keep].detach().cpu(),
                    labels=pred_instances.labels[keep].detach().cpu().to(torch.int64) + 1,
                )
            )
        return predictions

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        if self.model is None:
            self.model = self._build_inference_model(checkpoint_path=self.best_checkpoint_path)
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
