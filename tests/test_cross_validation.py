"""Integrity, scientific metric, statistics and pipeline regression checks."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

import main
from datasets.cross_validation import iterative_folds, prepare_cross_validation
from datasets.dataset_loader import build_dataloaders
from evaluation.coco_metrics import coco_evaluate
from evaluation.evaluate import evaluate_model, detection_confusion
from evaluation.metrics import compute_count_metrics
from evaluation.reporting import corrected_paired_test, generate_comparisons, generate_reports, holm_adjust
from models.common import PredictionResult
from models.ultralytics_training import fit_ultralytics
from utils.artifacts import save_figure, save_table, write_json

PROJECT = Path(__file__).resolve().parents[1]


def fixture(root, n=10):
    images = root / "dataset" / "images"
    images.mkdir(parents=True)
    data = {"info": {}, "images": [], "annotations": [], "categories": [
        {"id": 0, "name": "objects"}, {"id": 7, "name": "alpha"}, {"id": 42, "name": "beta"}]}
    for index in range(n):
        name = f"image_{index}.png"
        Image.new("RGB", (16, 16), (index, 20, 30)).save(images / name)
        data["images"].append({"id": index, "file_name": name, "width": 16, "height": 16})
        for category in ([7, 42] if index % 2 == 0 else [7]):
            data["annotations"].append({"id": len(data["annotations"])+1, "image_id": index,
                "category_id": category, "bbox": [1, 1, 8, 8], "area": 64, "iscrowd": 0})
    (images / "annotations.json").write_text(json.dumps(data))
    config = {"experiment": {"seed": 42, "name": "test_cv", "output_dir": str(root / "results")},
              "dataset": {"root": str(root / "dataset"), "images": "images", "annotations": "images/annotations.json"},
              "cross_validation": {"enabled": True, "n_splits": 5, "strategy": "iterative_multilabel", "checkpoint": "last", "group_file": None},
              "statistics": {"primary_metric": "mAP50_95", "comparison_metrics": ["mAP50_95", "MAE"], "required_models": ["fasterrcnn"], "alpha": 0.05},
              "training": {"epochs": 2, "batch_size": 2, "num_workers": 0, "image_size": 16, "log_interval": 10,
                           "grad_clip_norm": 1.0, "qualitative_images": 0, "learning_rate": 0.1, "optimizer": "SGD", "weight_decay": 0.0, "momentum": 0.0},
              "models": {"fasterrcnn": {"model_name": "fasterrcnn_resnet50_fpn_v2"}},
              "evaluation": {"score_threshold": 0.25, "ap_score_threshold": 0.001, "iou_threshold": 0.5,
                             "max_detections": 100, "warmup_batches": 0}}
    return config, data


class AllocationTests(unittest.TestCase):
    def test_capacity_coverage_and_reproducibility(self):
        labels = np.zeros((102, 3), dtype=int)
        labels[:, 0] = 1
        labels[:13, 1] = 1
        labels[::3, 2] = 1
        folds = iterative_folds(labels, 5, 42)
        np.testing.assert_array_equal(folds, iterative_folds(labels, 5, 42))
        self.assertEqual(sorted(np.bincount(folds)[1:]), [20, 20, 20, 21, 21])
        self.assertTrue(all(labels[folds == f, 1].sum() >= 2 for f in range(1, 6)))

    def test_empty_labels_and_invalid_n(self):
        folds = iterative_folds(np.zeros((10, 2)), 5, 42)
        self.assertEqual(list(np.bincount(folds)[1:]), [2]*5)
        with self.assertRaises(ValueError):
            iterative_folds(np.ones((4, 2)), 5, 42)

    def test_pool_mapping_export_and_immutability(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"):
            root = Path(directory)
            config, _ = fixture(root)
            first = prepare_cross_validation(config, PROJECT)
            second = prepare_cross_validation(config, PROJECT)
            self.assertEqual(first, second)
            self.assertEqual(config["dataset"]["category_ids"], [7, 42])
            self.assertEqual(config["dataset"]["class_names"], ["__background__", "alpha", "beta"])
            self.assertEqual(first["excluded_categories"][0]["name"], "objects")
            result = root / "results" / "test_cv"
            import yaml
            export = yaml.safe_load((result / "splits" / "fold_01" / "data.yaml").read_text())
            self.assertEqual(export["names"], {0: "alpha", 1: "beta"})
            train = set(Path(export["train"]).read_text().splitlines())
            val = set(Path(export["val"]).read_text().splitlines())
            self.assertFalse(train & val)
            self.assertEqual(len(train | val), 10)
            self.assertTrue((result / "data" / "images" / "image_0.png").is_symlink())
            labels = (result / "data" / "labels" / "image_0.txt").read_text().splitlines()
            self.assertEqual([line.split()[0] for line in labels], ["0", "1"])
            config["experiment"]["seed"] = 43
            with self.assertRaisesRegex(ValueError, "manifest"):
                prepare_cross_validation(config, PROJECT)

    def test_explicit_groups_stay_together(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"):
            root = Path(directory)
            config, _ = fixture(root)
            groups = root / "groups.csv"
            pd.DataFrame({"image_id": range(10), "group_id": [str(i//2) for i in range(10)]}).to_csv(groups, index=False)
            config["cross_validation"]["group_file"] = str(groups)
            manifest = prepare_cross_validation(config, PROJECT)
            frame = pd.DataFrame(manifest["images"])
            self.assertTrue((frame.groupby("group_id").fold.nunique() == 1).all())

    def test_missing_and_identical_files_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"):
            root = Path(directory)
            config, _ = fixture(root)
            first = root / "dataset" / "images" / "image_0.png"
            second = root / "dataset" / "images" / "image_1.png"
            second.write_bytes(first.read_bytes())
            with self.assertRaisesRegex(ValueError, "Identical images"):
                prepare_cross_validation(config, PROJECT)
            second.unlink()
            with self.assertRaises(FileNotFoundError):
                prepare_cross_validation(config, PROJECT)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = {"info": {}, "images": [{"id": 1, "width": 16, "height": 16}],
                   "categories": [{"id": 7, "name": "alpha"}, {"id": 42, "name": "absent"}],
                   "annotations": [{"id": 1, "image_id": 1, "category_id": 7,
                                    "bbox": [1, 1, 8, 8], "area": 64, "iscrowd": 0}]}

    def test_official_ap_perfect_and_missing_class(self):
        det = [{"image_id": 1, "category_id": 7, "bbox": [1, 1, 8, 8], "score": 0.1}]
        metrics, classes, curves = coco_evaluate(self.gt, det, [7, 42])
        self.assertAlmostEqual(metrics["mAP50_95"], 1)
        self.assertTrue(math.isnan(classes.loc[classes.Class == "absent", "AP50"].iloc[0]))
        self.assertEqual(len(curves), 202)

    def test_empty_detections_have_zero_ap(self):
        metrics, _, _ = coco_evaluate(self.gt, [], [7, 42])
        self.assertEqual(metrics["mAP50_95"], 0)

    def test_ap_uses_low_scores_counts_use_fixed_threshold(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"):
            config, _ = fixture(Path(directory))
            manifest = prepare_cross_validation(config, PROJECT)
            config["_cv"] = {"fold": 1, "seed": 42,
                "train_ids": [i["image_id"] for i in manifest["images"] if i["fold"] != 1],
                "val_ids": [i["image_id"] for i in manifest["images"] if i["fold"] == 1]}
            loader = build_dataloaders(config, PROJECT, "fasterrcnn")["val"]
            class Detector:
                class_names = config["dataset"]["class_names"]
                def predict_batch(self, images, device, score_threshold):
                    result = []
                    for image in images:
                        n = 2 if image.getpixel((0, 0))[0] % 2 == 0 else 1
                        result.append(PredictionResult(torch.tensor([[1.,1.,9.,9.]]*n), torch.full((n,), 0.1), torch.arange(1, n+1)))
                    return result
                def num_parameters(self):
                    return 10
            with patch("evaluation.evaluate.plot_evaluation"):
                metrics = evaluate_model(Detector(), loader, torch.device("cpu"), config, output_dir=Path(directory)/"evaluation")
            self.assertAlmostEqual(metrics["mAP50_95"], 1)
            self.assertEqual(metrics["Recall"], 0)
            self.assertGreater(metrics["MAE"], 0)

    def test_count_undefined_values_are_not_zero(self):
        empty = {"boxes": torch.zeros((0, 4))}
        result = compute_count_metrics([empty, empty], [empty, empty])
        self.assertTrue(math.isnan(result["MAPE"]))
        self.assertTrue(math.isnan(result["Pearson_r"]))
        self.assertTrue(math.isnan(result["R2"]))
        self.assertEqual(result["MAPE_valid_images"], 0)

    def test_confusion_counts_class_errors_and_unmatched(self):
        p = {"boxes": torch.tensor([[1.,1.,9.,9.], [10.,10.,12.,12.]]), "labels": torch.tensor([2, 1])}
        t = {"boxes": torch.tensor([[1.,1.,9.,9.], [13.,13.,15.,15.]]), "labels": torch.tensor([1, 2])}
        matrix = detection_confusion(p, t, 2, 0.5)
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(matrix[2, 0], 1)
        self.assertEqual(matrix[1, 2], 1)


class StatisticsTests(unittest.TestCase):
    def test_correction_matches_formula_and_exceeds_naive_se(self):
        diff = np.array([0.01, 0.03, 0.02, -0.01, 0.04])
        result = corrected_paired_test(diff, 0.25)
        expected = np.sqrt((1/5+0.25)*np.var(diff, ddof=1))
        self.assertAlmostEqual(result["Corrected_SE"], expected)
        self.assertGreater(expected, np.std(diff, ddof=1)/np.sqrt(5))
        self.assertEqual(result["df"], 4)
        self.assertLess(result["CI_low"], diff.mean())
        self.assertGreater(result["CI_high"], diff.mean())

    def test_holm_and_degenerate_differences(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(corrected_paired_test([0]*5, 0.25)["p_raw"], 1)
        self.assertIsNone(corrected_paired_test([0.2]*5, 0.25)["p_raw"])

    def test_paired_family_waits_for_complete_pre_specified_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            config["statistics"]["required_models"] = ["yolo", "rtdetr", "fasterrcnn", "detr"]
            records = []
            for model_index, model in enumerate(config["statistics"]["required_models"]):
                for fold in range(1, 6):
                    records.append({"Model": model, "Model_Key": model, "Fold": fold,
                        "Train_Images": 8, "Evaluation_Images": 2,
                        "mAP50_95": 0.3+0.05*model_index+0.001*fold*(model_index+1),
                        "MAE": 3-model_index*0.1+0.002*fold*(model_index+1)})
            frame = pd.DataFrame(records)
            with patch("evaluation.reporting.save_figure"):
                generate_comparisons(frame, ["yolo"], root, config)
                self.assertFalse((root/"paired_comparisons.csv").exists())
                generate_comparisons(frame, config["statistics"]["required_models"], root, config)
            comparisons = pd.read_csv(root/"paired_comparisons.csv")
            self.assertEqual(len(comparisons), 12)  # Six pairs, two pre-specified metrics.
            self.assertTrue((comparisons.p_Holm >= comparisons.p_raw).all())
            self.assertTrue((comparisons.df == 4).all())
            self.assertTrue((comparisons.N_pairs == 5).all())

    def test_artifacts_and_json_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metric.json", {"undefined": float("nan")})
            self.assertIsNone(json.loads((root/"metric.json").read_text())["undefined"])
            save_table(pd.DataFrame({"class_name": ["a&b"]}), root/"table", "Test")
            self.assertIn(r"class\_name", (root/"table.tex").read_text())
            self.assertIn(r"a\&b", (root/"table.tex").read_text())
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots()
            axis.plot([0,1], [0,1])
            save_figure(figure, root/"figure")
            for suffix in (".png", ".pdf", ".svg"):
                self.assertGreater((root/('figure'+suffix)).stat().st_size, 100)


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, images, targets):
        return {"loss": self.weight.square()}


class TinyDetector:
    config_key = "fasterrcnn"
    name = "Tiny test detector"
    architecture_type = "Test"
    is_ultralytics = False
    def __init__(self, config):
        self.config = config
        self.class_names = config["dataset"]["class_names"]
        self.model = TinyLossModel()
    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)
    def build_scheduler(self, optimizer):
        return None
    def prepare_batch(self, images, targets, device):
        return images, targets
    def predict_batch(self, images, device, score_threshold):
        return [PredictionResult(torch.zeros((0,4)), torch.zeros(0), torch.zeros(0,dtype=torch.long)) for _ in images]
    def num_parameters(self):
        return 1


class PipelineTests(unittest.TestCase):
    def test_training_cli_has_no_resume_option(self):
        with patch("sys.argv", ["main.py", "--model", "all"]):
            args = main.parse_args()
        self.assertFalse(hasattr(args, "resume"))

    def test_all_archives_previous_models_before_first_training(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"), \
             patch("evaluation.reporting.generate_reports"), patch("main.log_config"), patch("main.get_logger"):
            config, _ = fixture(Path(directory))
            root = Path(directory)/"results"/"test_cv"
            for model in ("yolo", "rtdetr", "fasterrcnn", "detr"):
                folder = root / ("previous_"+model)
                write_json(folder/"run.json", {"resolved": {"model_key": model, "source_sha256": "old",
                                                           "environment_sha256": "old"}})
                write_json(folder/"fold_01"/"completed.json", {})
            args = SimpleNamespace(model="all", fold=None, prepare_only=False, reports_only=False)
            with patch("main.build_model", side_effect=RuntimeError("simulated interruption")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    main.run_cross_validation(args, config, PROJECT)
            self.assertFalse(list(root.glob("*/fold_*/completed.json")))
            self.assertEqual(len(list(root.glob("archive/*/previous_*/fold_01/completed.json"))), 4)

    def test_incompatible_other_models_are_not_mixed_or_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            other = root/"DETR"
            write_json(other/"run.json", {"resolved": {"model_key": "detr", "source_sha256": "old",
                                                      "environment_sha256": "old"}})
            with self.assertRaisesRegex(ValueError, "--model all"):
                main.archive_previous_runs(root, ["yolo"], [root/"YOLO26m"], "new", "new")
            self.assertTrue((other/"run.json").exists())
            self.assertFalse((root/"archive").exists())

    def test_torchvision_wrapper_preprocessing_and_inference_without_downloads(self):
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        from models.faster_rcnn_model import FasterRCNNDetector
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            config["dataset"].update(num_classes=3, class_names=["__background__", "alpha", "beta"])
            config["training"]["image_size"] = 64
            config["models"]["fasterrcnn"]["model_name"] = "fasterrcnn_mobilenet_v3_large_fpn"
            def factory(**kwargs):
                kwargs.pop("weights")
                return fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, **kwargs)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with patch.dict("models.faster_rcnn_model.FASTER_R_CNN_MODEL_FACTORIES",
                                {"fasterrcnn_mobilenet_v3_large_fpn": factory}):
                    detector = FasterRCNNDetector(3, config["dataset"]["class_names"], torch.device("cpu"), config)
                image = Image.new("RGB", (16,16), "white")
                target = {"boxes_xywh": torch.tensor([[1.,1.,8.,8.]]), "labels": torch.tensor([1]),
                          "image_id": torch.tensor([1]), "area": torch.tensor([64.]), "iscrowd": torch.tensor([0])}
                tensors, _ = detector.prepare_batch([image], [target], torch.device("cpu"))
                self.assertTrue(torch.all(tensors[0] == 1))  # No second ImageNet normalization.
                self.assertEqual(detector.model.transform.min_size, (64,))
                outputs = detector.predict_batch([image], torch.device("cpu"), 0.001)
                self.assertEqual(len(outputs), 1)
                self.assertTrue(torch.isfinite(outputs[0].boxes).all())
                self.assertLessEqual(len(outputs[0].boxes), 100)
            finally:
                torch.set_num_threads(previous_threads)

    def test_detr_processor_size_and_zero_based_class_targets(self):
        from models.detr_model import DETRDetector
        from transformers import DetrImageProcessor
        with tempfile.TemporaryDirectory() as directory:
            config, _ = fixture(Path(directory))
            config["dataset"].update(num_classes=3, class_names=["__background__", "alpha", "beta"])
            config["training"]["image_size"] = 64
            processor = DetrImageProcessor(size={"shortest_edge": 64, "longest_edge": 64})
            with patch("models.detr_model.DetrImageProcessor.from_pretrained", return_value=processor) as loader, \
                 patch("models.detr_model.DetrForObjectDetection.from_pretrained", return_value=TinyLossModel()):
                detector = DETRDetector(3, config["dataset"]["class_names"], torch.device("cpu"), config)
            self.assertEqual(loader.call_args.kwargs["size"], {"shortest_edge":64,"longest_edge":64})
            target = {"boxes_xywh": torch.tensor([[1.,1.,8.,8.], [2.,2.,8.,8.]]), "labels": torch.tensor([1,2]),
                      "image_id": torch.tensor([1]), "area": torch.tensor([64.,64.]), "iscrowd": torch.tensor([0,0])}
            pixels, mask, labels = detector.prepare_batch([Image.new("RGB", (16,16))], [target], torch.device("cpu"))
            self.assertEqual(tuple(pixels.shape), (1,3,64,64))
            self.assertEqual(labels[0]["class_labels"].tolist(), [0,1])

    def test_five_fold_training_restarts_completed_runs_and_preserves_reports(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"), \
             patch("training.train.plot_training_history"), patch("training.train.save_prediction_gallery"), \
             patch("evaluation.evaluate.plot_evaluation"), patch("evaluation.reporting.plot_fold_metrics"), \
             patch("evaluation.reporting.generate_class_reports"), patch("evaluation.reporting.generate_learning_curves"), \
             patch("main.log_config"), patch("main.get_logger"):
            config, _ = fixture(Path(directory))
            args = SimpleNamespace(model="fasterrcnn", fold=None, prepare_only=False, reports_only=False)
            with patch("main.build_model", side_effect=lambda model_name, config, device: TinyDetector(config)) as constructor:
                main.run_cross_validation(args, config, PROJECT)
                self.assertEqual(constructor.call_count, 5)
                root = Path(directory)/"results"/"test_cv"
                self.assertEqual(len(list(root.glob("*/fold_*/completed.json"))), 5)
                checkpoints = list(root.glob("*/fold_*/checkpoints/weights/last.pt"))
                self.assertEqual(len(checkpoints), 5)
                self.assertFalse(list(root.glob("*/fold_*/checkpoints/weights/best.pt")))
                for path in checkpoints:
                    saved = torch.load(path, weights_only=True)
                    self.assertEqual(saved["epoch"], 1)
                counts = pd.read_csv(root/"reports"/"out_of_fold"/"FasterRCNN-resnet50-fpn-v2"/"image_counts.csv")
                self.assertEqual(set(counts.Image_ID), set(range(10)))
                self.assertFalse(counts.Image_ID.duplicated().any())
                main.run_cross_validation(args, config, PROJECT)
                self.assertEqual(constructor.call_count, 10)
                archived = list(root.glob("archive/*/FasterRCNN-resnet50-fpn-v2/fold_*/completed.json"))
                self.assertEqual(len(archived), 5)
                self.assertEqual(len(list(root.glob("*/fold_*/completed.json"))), 5)
                config["training"]["epochs"] = 3
                main.run_cross_validation(args, config, PROJECT)
                self.assertEqual(constructor.call_count, 15)
                self.assertEqual(len(list(root.glob("archive/*/FasterRCNN-resnet50-fpn-v2/fold_*/completed.json"))), 10)
                for path in root.glob("*/fold_*/checkpoints/weights/last.pt"):
                    self.assertEqual(torch.load(path, weights_only=True)["epoch"], 2)

    def test_partial_run_is_archived_before_restart(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"), \
             patch("evaluation.reporting.generate_reports"), patch("main.log_config"), patch("main.get_logger"):
            config, _ = fixture(Path(directory))
            args = SimpleNamespace(model="fasterrcnn", fold=[1], prepare_only=False, reports_only=False)
            with patch("main.build_model", side_effect=RuntimeError("simulated interruption")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    main.run_cross_validation(args, config, PROJECT)
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    main.run_cross_validation(args, config, PROJECT)
            archives = list((Path(directory)/"results"/"test_cv").glob("archive/*/FasterRCNN-resnet50-fpn-v2/fold_01/config.json"))
            self.assertEqual(len(archives), 1)

    def test_publication_renderers_with_a_partial_run(self):
        with tempfile.TemporaryDirectory() as directory, patch("datasets.cross_validation.plot_split_distribution"), \
             patch("training.train.plot_training_history"), patch("training.train.save_prediction_gallery"), \
             patch("evaluation.evaluate.plot_evaluation"), patch("main.log_config"), patch("main.get_logger"):
            config, _ = fixture(Path(directory))
            config["training"]["epochs"] = 1
            args = SimpleNamespace(model="fasterrcnn", fold=[1], prepare_only=False, reports_only=False)
            with patch("main.build_model", side_effect=lambda model_name, config, device: TinyDetector(config)):
                main.run_cross_validation(args, config, PROJECT)
            report = Path(directory)/"results"/"test_cv"/"reports"
            self.assertTrue((report/"figures"/"fold_metric_comparison.pdf").exists())
            self.assertTrue((report/"figures"/"per_class_ap.svg").exists())
            self.assertTrue((report/"figures"/"learning_FasterRCNN-resnet50-fpn-v2.png").exists())
            self.assertFalse(pd.read_csv(report/"completion_status.csv").Complete.iloc[0])
            self.assertFalse((report/"out_of_fold").exists())

    def test_ultralytics_uses_fold_yaml_final_weights_and_no_early_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = fixture(root)
            config["models"]["yolo"] = {"model_name": "yolov8n.pt"}
            config["_cv"] = {"fold": 2, "seed": 43, "data_yaml": str(root/"fold2.yaml")}
            observed = {}
            class FakeUltralytics:
                def train(self, **kwargs):
                    observed.update(kwargs)
                    save_dir = Path(kwargs["project"])/kwargs["name"]
                    (save_dir/"weights").mkdir(parents=True)
                    (save_dir/"weights"/"last.pt").touch()
                    pd.DataFrame({"epoch": [1, 2]}).to_csv(save_dir/"results.csv", index=False)
                    self.trainer = SimpleNamespace(save_dir=save_dir, model=TinyLossModel())
            wrapper = SimpleNamespace(config_key="yolo", project_root=PROJECT, model_name="yolov8n.pt",
                                      optimizer_name="SGD", name="YOLO test", model=FakeUltralytics())
            from unittest.mock import Mock
            constructor, logger = Mock(), Mock()
            checkpoint = fit_ultralytics(wrapper, config, torch.device("cpu"), logger, constructor)
            self.assertEqual(checkpoint.name, "last.pt")
            constructor.assert_called_once_with(str(checkpoint))
            self.assertEqual(observed["data"], str(root/"fold2.yaml"))
            self.assertEqual(observed["patience"], 0)
            self.assertFalse(observed["resume"])
            self.assertEqual(observed["seed"], 43)
            self.assertEqual(observed["workers"], 0)
            self.assertEqual(wrapper.parameter_count, 1)


if __name__ == "__main__":
    unittest.main()
