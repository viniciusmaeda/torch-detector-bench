"""COCO dataset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from utils.config import get_training_config
from utils.paths import get_model_results_dir


class COCODetectionDataset(Dataset):
    """Lightweight COCO dataset that returns PIL images and target dictionaries."""

    def __init__(
        self,
        images_dir: Path,
        annotations_file: Path,
        image_size: int,
        category_id_to_contiguous: Dict[int, int] | None = None,
        image_ids: List[int] | None = None,
    ) -> None:
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.image_size = image_size
        self.category_id_to_contiguous = category_id_to_contiguous or {}
        self.coco = COCO(str(annotations_file))
        self.image_ids = list(sorted(self.coco.imgs.keys())) if image_ids is None else list(image_ids)
        if not set(self.image_ids).issubset(self.coco.imgs):
            raise ValueError("Unknown COCO image ID in fold.")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict[str, Any]]:
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        boxes_xywh: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in annotations:
            x_min, y_min, width, height = ann["bbox"]
            x_max = min(float(x_min + width), float(image_width))
            y_max = min(float(y_min + height), float(image_height))
            x_min = max(float(x_min), 0.0)
            y_min = max(float(y_min), 0.0)
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue
            boxes_xywh.append([x_min, y_min, width, height])
            category_id = int(ann["category_id"])
            contiguous_label = self.category_id_to_contiguous.get(category_id, category_id)
            labels.append(contiguous_label)
            areas.append(width * height)
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target: Dict[str, Any] = {
            "boxes_xywh": torch.as_tensor(boxes_xywh, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
            "size": torch.tensor([image.height, image.width], dtype=torch.int64),
            "image_path": str(image_path),
            "file_name": image_info["file_name"],
            "annotations": annotations,
        }
        return image, target


def collate_fn(batch: List[Tuple[Image.Image, Dict[str, Any]]]):
    """Custom collate for variable-length detection targets."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(config: Dict[str, Any], project_root: Path, model_name: str) -> Dict[str, DataLoader]:
    """Create train, validation and test dataloaders."""
    dataset_config = config["dataset"]
    training_config = get_training_config(config, model_name)

    dataset_root = (project_root / dataset_config["root"]).resolve()
    image_size = int(training_config["image_size"])
    num_workers = int(training_config["num_workers"])
    batch_size = int(training_config["batch_size"])
    if "_cv" in config:
        mapping = {category_id: index + 1 for index, category_id in enumerate(dataset_config["category_ids"])}
        loaders = {}
        for split in ("train", "val"):
            dataset = COCODetectionDataset(
                images_dir=dataset_root / dataset_config["images"],
                annotations_file=dataset_root / dataset_config["annotations"],
                image_size=image_size,
                category_id_to_contiguous=mapping,
                image_ids=config["_cv"][f"{split}_ids"],
            )
            generator = torch.Generator().manual_seed(int(config["_cv"]["seed"]))
            loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                                       num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                       collate_fn=collate_fn, generator=generator, worker_init_fn=seed_worker)
        return loaders
    category_id_to_contiguous = build_category_id_mapping(
        annotation_file=dataset_root / dataset_config["train_annotations"]
    )

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = COCODetectionDataset(
            images_dir=dataset_root / dataset_config[f"{split}_images"],
            annotations_file=dataset_root / dataset_config[f"{split}_annotations"],
            image_size=image_size,
            category_id_to_contiguous=category_id_to_contiguous,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
        )
    return loaders


def build_yolo_data_config(config: Dict[str, Any], project_root: Path, model_name: str = "yolo") -> Path:
    """Create an Ultralytics data YAML file from the COCO dataset layout."""
    if "_cv" in config:
        return Path(config["_cv"]["data_yaml"])
    dataset_root = (project_root / config["dataset"]["root"]).resolve()
    output_dir = get_model_results_dir(project_root=project_root, config=config, model_name=model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo_data_path = output_dir / "yolo_data.yaml"

    split_to_label_dir = {"train": "train", "val": "valid", "test": "test"}
    for split, label_dir in split_to_label_dir.items():
        convert_coco_to_yolo_labels(
            images_dir=dataset_root / config["dataset"][f"{split}_images"],
            annotation_file=dataset_root / config["dataset"][f"{split}_annotations"],
            labels_dir=dataset_root / "labels" / label_dir,
        )

    class_names = config["dataset"]["class_names"][1:]
    data = {
        "path": str(dataset_root),
        "train": config["dataset"]["train_images"],
        "val": config["dataset"]["val_images"],
        "test": config["dataset"]["test_images"],
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    import yaml

    with open(yolo_data_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return yolo_data_path


def load_coco_categories(annotation_file: Path) -> Dict[int, str]:
    """Read category identifiers from a COCO annotation file."""
    with open(annotation_file, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    return {category["id"]: category["name"] for category in data["categories"]}


def build_category_id_mapping(annotation_file: Path) -> Dict[int, int]:
    """Create a contiguous class-id mapping from COCO category ids."""
    categories = load_coco_categories(annotation_file=annotation_file)
    sorted_category_ids = sorted(categories.keys())
    return {category_id: index + 1 for index, category_id in enumerate(sorted_category_ids)}


def infer_dataset_classes(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Infer class names and class count from the COCO train annotations."""
    dataset_root = (project_root / config["dataset"]["root"]).resolve()
    if config.get("cross_validation", {}).get("enabled", False):
        from datasets.cross_validation import load_pool
        load_pool(config, project_root)
        return config
    annotation_file = dataset_root / config["dataset"]["train_annotations"]
    categories = load_coco_categories(annotation_file=annotation_file)
    sorted_category_ids = sorted(categories.keys())

    config["dataset"]["class_names"] = ["__background__"] + [
        categories[category_id] for category_id in sorted_category_ids
    ]
    config["dataset"]["num_classes"] = len(config["dataset"]["class_names"])
    return config


def seed_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the DataLoader's reproducible worker seed."""
    import random
    import numpy as np
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def convert_coco_to_yolo_labels(images_dir: Path, annotation_file: Path, labels_dir: Path) -> None:
    """Convert COCO annotations into YOLO text labels for Ultralytics training."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    with open(annotation_file, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    image_lookup = {image["id"]: image for image in data["images"]}
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for annotation in data["annotations"]:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    category_ids = sorted(category["id"] for category in data["categories"])
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}

    for image_id, image_info in image_lookup.items():
        label_path = labels_dir / f"{Path(image_info['file_name']).stem}.txt"
        width = float(image_info["width"])
        height = float(image_info["height"])
        lines: List[str] = []

        for annotation in annotations_by_image.get(image_id, []):
            x_min, y_min, box_width, box_height = annotation["bbox"]
            x_center = (x_min + (box_width / 2.0)) / width
            y_center = (y_min + (box_height / 2.0)) / height
            norm_width = box_width / width
            norm_height = box_height / height
            class_id = category_to_index[int(annotation["category_id"])]
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        label_path.write_text("\n".join(lines), encoding="utf-8")
