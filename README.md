# torch-detector-bench

Benchmark project for training and evaluating multiple object detection architectures on a COCO-style dataset using PyTorch, Torchvision, Hugging Face Transformers, and Ultralytics.

The repository currently supports:

- YOLO variants through Ultralytics
- RT-DETR variants through Ultralytics
- Faster R-CNN with Torchvision
- DETR with Hugging Face Transformers

## Project Goals

This project provides a single pipeline to:

- train different detector families with a shared dataset layout
- evaluate them with a common set of detection and counting metrics
- save logs, checkpoints, training curves, and qualitative predictions
- compare results across one-stage, two-stage, and transformer-based detectors

## Supported Models

The CLI currently supports the following model keys:

- `yolo`
- `rtdetr`
- `fasterrcnn`
- `detr`

The current configuration file selects these concrete variants:

- `yolo`: `yolo26m.pt`
- `rtdetr`: `rtdetr-l.pt`
- `fasterrcnn`: `fasterrcnn_resnet50_fpn`
- `detr`: `facebook/detr-resnet-50`

## Repository Structure

```text
compare_torch_detectors/
├── config/
│   └── config.yaml
├── datasets/
│   ├── images/
│   └── labels/
├── environment/
│   └── torch-detector-bench.yml
├── evaluation/
├── models/
├── training/
├── utils/
├── main.py
└── results/
```

## Requirements

- Python 3.10
- PyTorch
- Torchvision
- CUDA support if you want GPU training
- Ultralytics
- Transformers
- pycocotools

The Conda environment configuration file is available at [environment/torch-detector-bench.yml](environment/torch-detector-bench.yml).

## Environment Setup

Create the environment with Conda using the name `torch-detector-bench`:

```bash
conda env create -n torch-detector-bench -f environment/torch-detector-bench.yml
conda activate torch-detector-bench
```

## Dataset Format

The project expects a COCO-style dataset structure configured in [config/config.yaml](config/config.yaml):

```text
datasets/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

The annotation files referenced by default are:

- `datasets/images/train/_annotations.coco.json`
- `datasets/images/valid/_annotations.coco.json`
- `datasets/images/test/_annotations.coco.json`

If your dataset classes change, synchronize them into the config file with:

```bash
python datasets/sync_dataset_classes.py
```

## Configuration

Experiment settings are defined in [config/config.yaml](config/config.yaml), including:

- random seed
- output directory
- shared training defaults
- model-specific optimizer and training settings
- evaluation thresholds
- dataset paths and class names

## Running Experiments

Run training and evaluation from the project root:

```bash
python main.py --model yolo
python main.py --model rtdetr
python main.py --model fasterrcnn
python main.py --model detr
```

To use a different configuration file:

```bash
python main.py --model fasterrcnn --config config/config.yaml
```

## Outputs

Generated artifacts are stored under the directory defined by `experiment.output_dir` in the config file, which currently points to `results/`.

Depending on the selected model, the project saves:

- training logs
- checkpoints
- per-epoch history
- training curves
- qualitative prediction galleries
- aggregate metrics in CSV format

## Evaluation Metrics

The shared evaluation pipeline reports detection and counting metrics such as:

- `mAP50`
- `mAP50_95`
- `Precision`
- `Recall`
- `F1`
- `MAE`
- `RMSE`
- `MAPE`
- `Bias`
- `Pearson_r`
- `R2`
- `FPS`
- `Inference_time`
- `Parameters`

## Notes for Version Control

This repository is configured to keep source code and configuration under version control while ignoring:

- local datasets
- generated results
- model weight files
- IDE settings
- Python cache files

That keeps the repository lightweight and avoids publishing large artifacts or experiment leftovers by default.
