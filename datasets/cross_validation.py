"""Single-pool, reproducible multilabel cross-validation preparation.

Capacity-constrained iterative stratification follows the rare-label-first
allocation of Sechidis, Tsoumakas and Vlahavas (2011). Optional explicit groups
are indivisible; their sizes can make exactly equal folds impossible.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils.artifacts import fingerprint, save_figure, save_table, write_json
from utils.paths import get_results_root


def iterative_folds(labels: np.ndarray, n_splits: int, seed: int, weights=None) -> np.ndarray:
    """Allocate rare labels first, resolving ties using the experiment seed."""
    labels = np.asarray(labels, dtype=float)
    if labels.ndim != 2 or len(labels) < n_splits or n_splits < 2:
        raise ValueError("Need at least n_splits images/groups and a 2-D label matrix.")
    rng = np.random.default_rng(seed)
    weights = np.ones(len(labels), dtype=int) if weights is None else np.asarray(weights, dtype=int)
    if len(weights) != len(labels) or np.any(weights <= 0):
        raise ValueError("Each stratification unit must have a positive image count.")
    total = int(weights.sum())
    capacity = np.full(n_splits, total // n_splits, dtype=float)
    capacity[:total % n_splits] += 1
    rng.shuffle(capacity)
    desired = np.outer(capacity / total, labels.sum(axis=0))
    remaining = np.ones(len(labels), dtype=bool)
    assignments = np.full(len(labels), -1, dtype=int)
    while remaining.any():
        support = (labels[remaining] > 0).sum(axis=0)
        nonzero = np.flatnonzero(support)
        if len(nonzero):
            rare = rng.choice(nonzero[support[nonzero] == support[nonzero].min()])
            candidates = np.flatnonzero(remaining & (labels[:, rare] > 0))
        else:
            rare = None
            candidates = np.flatnonzero(remaining)
        rng.shuffle(candidates)
        # Allocate large explicit groups first; singleton order remains random.
        candidates = candidates[np.argsort(-weights[candidates], kind="stable")]
        for index in candidates:
            available = np.flatnonzero(capacity >= weights[index])
            if not len(available):
                available = np.flatnonzero(capacity == capacity.max())
            if rare is not None:
                need = desired[available, rare]
                available = available[np.isclose(need, need.max())]
            available = available[np.isclose(capacity[available], capacity[available].max())]
            fold = int(rng.choice(available))
            assignments[index] = fold + 1
            capacity[fold] -= weights[index]
            desired[fold] -= labels[index]
            remaining[index] = False
    if set(assignments) != set(range(1, n_splits + 1)):
        raise ValueError("Stratification produced an empty fold; revise the group definitions.")
    return assignments


def load_pool(config: dict, project_root: Path):
    """Validate COCO references and infer a global mapping of annotated classes."""
    settings = config["dataset"]
    root = (project_root / settings["root"]).resolve()
    image_dir = root / settings["images"]
    annotation_path = root / settings["annotations"]
    raw = annotation_path.read_bytes()
    data = json.loads(raw)
    for key in ("images", "annotations", "categories"):
        records = data[key]
        if len({r["id"] for r in records}) != len(records):
            raise ValueError(f"Duplicate COCO {key} IDs.")
    images = sorted(data["images"], key=lambda i: i["id"])
    lookup = {i["id"]: i for i in images}
    categories = {c["id"]: c for c in data["categories"]}
    filenames = [i["file_name"] for i in images]
    if len(set(filenames)) != len(filenames):
        raise ValueError("COCO image filenames must be unique.")
    if len({str(Path(name).with_suffix('.txt')) for name in filenames}) != len(filenames):
        raise ValueError("Image filenames collide after conversion to YOLO label filenames.")
    digest = hashlib.sha256(raw)
    content_hashes = {}
    for image in images:
        path = (image_dir / image["file_name"]).resolve()
        if not path.is_relative_to(image_dir.resolve()):
            raise ValueError("Image paths must stay inside the image directory.")
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        if content in content_hashes:
            raise ValueError(f"Identical images: {content_hashes[content]} and {image['file_name']}. Deduplicate first.")
        content_hashes[content] = image["file_name"]
        digest.update(f"{image['id']}:{image['file_name']}:{content}".encode())
    annotated = set()
    for ann in data["annotations"]:
        if ann["image_id"] not in lookup or ann["category_id"] not in categories:
            raise ValueError(f"Invalid annotation reference: {ann['id']}")
        bbox = np.asarray(ann["bbox"], dtype=float)
        image = lookup[ann["image_id"]]
        if bbox.shape != (4,) or not np.isfinite(bbox).all() or min(bbox[2:]) <= 0:
            raise ValueError(f"Invalid bounding box: {ann['id']}")
        if min(bbox[:2]) < -0.01 or bbox[0] + bbox[2] > image["width"] + 0.01 or bbox[1] + bbox[3] > image["height"] + 0.01:
            raise ValueError(f"Bounding box outside image: {ann['id']}")
        annotated.add(ann["category_id"])
    active = sorted(annotated)
    if not active:
        raise ValueError("No annotated classes in the dataset.")
    if len({categories[c]['name'] for c in active}) != len(active):
        raise ValueError("Annotated class names must be unique.")
    config["dataset"]["category_ids"] = active
    config["dataset"]["class_names"] = ["__background__"] + [categories[c]["name"] for c in active]
    config["dataset"]["num_classes"] = len(active) + 1
    return data, images, image_dir, digest.hexdigest()


def prepare_cross_validation(config: dict, project_root: Path) -> dict:
    """Persist/reuse an immutable manifest, label audit, fold lists and YAMLs."""
    cv = config["cross_validation"]
    if cv.get("strategy") != "iterative_multilabel":
        raise ValueError("Supported CV strategy: iterative_multilabel.")
    if cv.get("checkpoint") != "last":
        raise ValueError("Cross-validation evaluates the final epoch: checkpoint must be 'last'.")
    n_splits, seed = int(cv["n_splits"]), int(config["experiment"]["seed"])
    data, images, image_dir, dataset_hash = load_pool(config, project_root)
    if n_splits < 2 or len(images) < n_splits:
        raise ValueError("Invalid n_splits for this dataset.")
    category_ids = config["dataset"]["category_ids"]
    row_lookup = {i["id"]: index for index, i in enumerate(images)}
    col_lookup = {c: index for index, c in enumerate(category_ids)}
    counts = np.zeros((len(images), len(category_ids)), dtype=int)
    for ann in data["annotations"]:
        counts[row_lookup[ann["image_id"]], col_lookup[ann["category_id"]]] += 1
    labels = (counts > 0).astype(int)
    group_path = cv.get("group_file")
    group_hash = None
    if group_path:
        path = project_root / group_path
        group_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, dtype={"group_id": str})
        if not {"image_id", "group_id"}.issubset(frame.columns) or frame["image_id"].duplicated().any() or frame["group_id"].isna().any():
            raise ValueError("Group CSV needs unique image_id and nonempty group_id columns.")
        group_map = dict(zip(frame.image_id, frame.group_id))
        if set(group_map) != set(row_lookup):
            raise ValueError("Group CSV must describe every image exactly once.")
        groups = [group_map[i["id"]] for i in images]
    else:
        groups = [str(i["id"]) for i in images]
    units = sorted(set(groups))
    unit_lookup = {g: index for index, g in enumerate(units)}
    unit_labels = np.zeros((len(units), labels.shape[1]), dtype=int)
    weights = np.zeros(len(units), dtype=int)
    for index, group in enumerate(groups):
        unit_labels[unit_lookup[group]] += labels[index]
        weights[unit_lookup[group]] += 1
    assignment = iterative_folds(unit_labels, n_splits, seed, weights)
    folds = [int(assignment[unit_lookup[g]]) for g in groups]
    protocol = {
        "version": 1, "dataset_sha256": dataset_hash, "n_splits": n_splits,
        "seed": seed, "strategy": cv["strategy"], "checkpoint": "last",
        "group_sha256": group_hash, "evaluation": config["evaluation"], "statistics": config["statistics"],
        "category_ids": category_ids, "class_names": config["dataset"]["class_names"],
    }
    manifest = {**protocol, "images": [{"image_id": i["id"], "file_name": i["file_name"],
                  "fold": f, "group_id": g} for i, f, g in zip(images, folds, groups)],
                "excluded_categories": [c for c in data["categories"] if c["id"] not in category_ids]}
    manifest["manifest_sha256"] = fingerprint(manifest)
    root = get_results_root(project_root, config)
    path = root / "splits" / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError("Dataset/protocol differs from saved manifest. Use a new experiment.name.")
    else:
        write_json(path, manifest)
    save_table(pd.DataFrame(manifest["images"]), root / "splits" / "image_folds", "Reproducible image allocation")
    distribution = []
    for c, name in enumerate(config["dataset"]["class_names"][1:]):
        record = {"Class": name, "Category_ID": category_ids[c], "Images": int(labels[:, c].sum()), "Objects": int(counts[:, c].sum())}
        for f in range(1, n_splits + 1):
            mask = np.asarray(folds) == f
            record[f"Fold_{f}_images"] = int(labels[mask, c].sum())
            record[f"Fold_{f}_objects"] = int(counts[mask, c].sum())
        distribution.append(record)
    frame = pd.DataFrame(distribution)
    save_table(frame, root / "splits" / "class_distribution", "Class support in the complete pool and held-out folds")
    write_json(root / "splits" / "audit.json", {
        "images": len(images), "annotations": len(data["annotations"]), "active_classes": len(category_ids),
        "excluded_categories": manifest["excluded_categories"], "grouped": bool(group_path),
        "classes_below_n_splits": frame.loc[frame.Images < n_splits, "Class"].tolist(),
        "missing_fold_classes": [{"fold": f, "classes": frame.loc[frame[f"Fold_{f}_images"] == 0, "Class"].tolist()}
                                 for f in range(1, n_splits + 1)],
        "fold_sizes": dict(Counter(folds)),
    })
    export_yolo_pool(data, images, image_dir, category_ids, root, folds, n_splits)
    plot_split_distribution(frame, root)
    return manifest


def export_yolo_pool(data, images, image_dir, category_ids, root, folds, n_splits):
    """Mirror images using symlinks and convert labels once with a global mapping."""
    converted = root / "data"
    by_image = {i["id"]: [] for i in images}
    class_index = {c: index for index, c in enumerate(category_ids)}
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann)
    mirrored = []
    for image in images:
        relative = Path(image["file_name"])
        link = (converted / "images" / relative).absolute()
        link.parent.mkdir(parents=True, exist_ok=True)
        source = (image_dir / relative).resolve()
        if link.is_symlink():
            if link.resolve() != source:
                raise ValueError(f"Unexpected image symlink: {link}")
        elif link.exists():
            raise ValueError(f"Expected a generated image symlink: {link}")
        else:
            link.symlink_to(source)
        label_path = (converted / "labels" / relative).with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for ann in by_image[image["id"]]:
            if ann.get("iscrowd", 0):
                raise ValueError("Ultralytics does not represent COCO crowd boxes; revise dataset/protocol first.")
            x, y, w, h = ann["bbox"]
            lines.append(f"{class_index[ann['category_id']]} {(x+w/2)/image['width']:.9f} {(y+h/2)/image['height']:.9f} {w/image['width']:.9f} {h/image['height']:.9f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        mirrored.append(str(link))
    for f in range(1, n_splits + 1):
        folder = root / "splits" / f"fold_{f:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for partition in ("train", "val"):
            selected = [p for p, fold in zip(mirrored, folds) if (fold == f) == (partition == "val")]
            (folder / f"{partition}.txt").write_text("\n".join(selected) + "\n")
        with (folder / "data.yaml").open("w") as stream:
            yaml.safe_dump({"path": str(converted.absolute()), "train": str((folder/"train.txt").absolute()),
                            "val": str((folder/"val.txt").absolute()),
                            "names": {i: c["name"] for i, c in enumerate(sorted(
                                (c for c in data["categories"] if c["id"] in category_ids), key=lambda c: c["id"]))}},
                           stream, sort_keys=False, allow_unicode=True)


def plot_split_distribution(frame, root):
    import matplotlib.pyplot as plt
    ordered = frame.sort_values("Images")
    figure, axis = plt.subplots(figsize=(9, 10))
    axis.barh(ordered.Class, ordered.Images)
    axis.set_xlabel("Images containing the class")
    axis.set_title("Dataset class distribution")
    save_figure(figure, root / "reports" / "figures" / "class_distribution")
    matrix = frame[[c for c in frame if c.endswith("_images")]].to_numpy()
    figure, axis = plt.subplots(figsize=(8, 11))
    plot = axis.imshow(matrix, aspect="auto", cmap="Blues")
    axis.set_yticks(range(len(frame)), frame.Class)
    axis.set_xticks(range(matrix.shape[1]), [f"Fold {i+1}" for i in range(matrix.shape[1])])
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            axis.text(c, r, str(matrix[r, c]), ha="center", va="center", fontsize=8,
                      color="white" if matrix[r, c] > matrix.max()/2 else "black")
    figure.colorbar(plot, ax=axis, label="Images containing the class")
    axis.set_title("Class support in each held-out fold")
    save_figure(figure, root / "reports" / "figures" / "fold_class_support")
