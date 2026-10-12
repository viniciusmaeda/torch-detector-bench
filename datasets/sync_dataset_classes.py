"""Synchronize COCO class metadata into config/config.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)



def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    """Persist YAML data to disk."""
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)



def extract_categories(annotation_file: Path) -> tuple[list[int], list[str]]:
    """Extract ordered COCO category ids and names."""
    with open(annotation_file, "r", encoding="utf-8") as stream:
        data = json.load(stream)

    categories = sorted(data["categories"], key=lambda item: int(item["id"]))
    category_ids = [int(category["id"]) for category in categories]
    class_names = [str(category["name"]) for category in categories]
    return category_ids, class_names



def main() -> None:
    """Update config/config.yaml with class metadata from train annotations."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / 'config' / 'config.yaml'
    config = load_yaml(config_path)

    if config.get('cross_validation', {}).get('enabled', False):
        from datasets.cross_validation import load_pool
        load_pool(config, project_root)
        save_yaml(config_path, config)
        print(f'Updated {config_path}: {config["dataset"]["num_classes"] - 1} annotated classes plus background')
        return

    dataset_root = (project_root / config['dataset']['root']).resolve()
    train_annotations = dataset_root / config['dataset']['train_annotations']

    if not train_annotations.exists():
        raise FileNotFoundError(f'Train annotation file not found: {train_annotations}')

    _, class_names = extract_categories(train_annotations)

    config['dataset']['num_classes'] = len(class_names) + 1
    config['dataset']['class_names'] = ['__background__'] + class_names
    config['dataset'].pop('category_ids', None)
    config['dataset'].pop('category_id_to_contiguous', None)

    save_yaml(config_path, config)

    print(f'Updated {config_path}')
    print(f'num_classes: {config["dataset"]["num_classes"]}')
    print('class_names:')
    for class_name in config['dataset']['class_names']:
        print(f'  - {class_name}')


if __name__ == '__main__':
    main()
