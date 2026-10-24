# Pork-Rasher Packaging Defect Detection

MMA3001 (Numerical Methods and Machine Learning) individual project, by Lachlan Lim.

This project trains YOLO11 object-detection models to find packaging and product defects in RGB images of frozen pork-rasher packs. It compares model sizes and input resolutions to choose a suitable detector for automated inspection.

## Engineering problem

On a meat-packing line, packs with poor seals, loose or twisted meat, or packaging faults should be removed before shipping. Manual visual inspection is slow and inconsistent. An image-based detector could flag defective packs automatically.

| | Description |
|---|---|
| **Input** | RGB image of a pork-rasher pack (JPEG/PNG, any size; the model resizes it to the training resolution) |
| **Output** | Bounding boxes in pixel coordinates, each with a defect class and a confidence score from 0 to 1 |
| **Defect classes** | `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle` |

An image with no detections means no defect was found above the confidence threshold. This does not guarantee that the pack is defect-free.

## Dataset

[Pork-Rasher-Error(Packaging), version 4](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4) from Roboflow Universe, licensed under **CC BY 4.0**. It is downloaded in YOLO11 format and is not stored in this repository.

| Split | Images | Boxes | Images with no boxes |
|---|---|---|---|
| train | 3349 | 2923 | 684 |
| valid | 120 | 121 | 14 |
| test | 80 | 79 | 12 |

The classes are strongly imbalanced. `packaging-error` (1712 training boxes) and `unsealed` (1055) dominate, while `wrinkle` has only 18 training boxes and **none** in the validation or test splits.

Each label file holds one box per line as `class_id x_center y_center width height`. The four box values are normalised from 0 to 1. `src/validate_annotations.py` checks every label against these rules.

## Repository structure

```
├── src/                     Source code
│   ├── download_data.py       Download the dataset from Roboflow
│   ├── inspect_dataset.py     Print class names and split sizes
│   ├── validate_annotations.py  Check every YOLO label is valid
│   ├── visualise_labels.py    Draw ground-truth boxes on sample images
│   ├── train_yolo11n.py       Baseline: YOLO11n at 640 px
│   ├── train_yolo11n_480.py   Resolution study: 480 px
│   ├── train_yolo11n_800.py   Resolution study: 800 px
│   ├── train_yolo11s.py       Model-size study: YOLO11s at 640 px
│   ├── resume_yolo11s.py      Resume interrupted YOLO11s training
│   ├── evaluate.py            Compare models: accuracy, speed, memory, FLOPs
│   └── make_test_report.py    Build the HTML test report
├── tests/                   pytest tests for the dataset and annotations
├── results/                 Metrics, plots and test reports
│   ├── yolo11n_480/ …         Per-experiment training curves and confusion matrices
│   ├── final_test/            Evaluation of the selected model on the test split
│   ├── label_checks/          Sample images with ground-truth boxes drawn
│   ├── test_reports/          pytest XML and HTML test report
│   └── *.csv                  Comparison and final test metrics
├── docs/                    HTML code documentation (pdoc)
├── requirements.txt
└── LICENSE
```

`data/`, `runs/` (Ultralytics training output) and the `.pt` model weights are ignored by Git.

## Setup

Requires Python 3.12 and, for training, a GPU.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (use source .venv/bin/activate on macOS/Linux)
```

Install PyTorch first, choosing the build for your hardware at <https://pytorch.org/get-started/locally/> (CUDA, ROCm or CPU). Then install everything else:

```bash
pip install -r requirements.txt
```

## Workflow

Run every command from the project root.

1. **Download the data.** You need a free Roboflow API key.
   ```bash
   set ROBOFLOW_API_KEY=your_key      # Windows (use export on macOS/Linux)
   python src/download_data.py
   ```
2. **Check the data.**
   ```bash
   python src/inspect_dataset.py
   python src/validate_annotations.py
   python src/visualise_labels.py
   ```
3. **Train the models.** Each run takes about 1–2 hours on a GPU.
   ```bash
   python src/train_yolo11n.py
   python src/train_yolo11n_480.py
   python src/train_yolo11n_800.py
   python src/train_yolo11s.py
   ```
   Ultralytics saves weights, metrics and plots under `runs/detect/results/training/<experiment>/`.
4. **Evaluate.** This compares all models on the validation split, measures their speed, memory and FLOPs, then tests the selected model once on the test split. It takes a few minutes and does not retrain anything.
   ```bash
   python src/evaluate.py
   ```
   This writes the three CSV files in `results/`, and Ultralytics plots to `runs/detect/evaluation/`.

All experiments use 50 epochs, batch size 8 and seed 42, starting from COCO-pretrained weights.

## Tests

```bash
pytest --junitxml=results/test_reports/all_tests.xml
python src/make_test_report.py
```

The tests check the dataset configuration, the split folders and every YOLO label file. They also check the annotation validator against valid and invalid examples. The tests that need the dataset are skipped if it has not been downloaded. The report is saved to [`results/test_reports/test_summary.html`](results/test_reports/test_summary.html).

## Documentation

HTML documentation of the source code is in [`docs/`](docs/index.html). Open `docs/index.html` in a browser. To regenerate it after changing any docstrings:

```bash
set PYTHONPATH=src                  # Windows (use export on macOS/Linux)
pdoc -d google --mermaid --footer-text "MMA3001 Project · Lachlan Lim" src -o docs
```

## Results

All results below are produced by `src/evaluate.py`.

Accuracy on the validation split (120 images):

| Model | Resolution | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO11n | 480 | 0.714 | 0.565 | 0.576 | 0.342 |
| YOLO11n | 640 | 0.705 | 0.622 | 0.647 | 0.355 |
| YOLO11n | 800 | 0.775 | 0.642 | 0.646 | 0.343 |
| YOLO11s | 640 | 0.573 | 0.551 | 0.563 | 0.387 |

Computational cost (AMD Radeon RX 9070 XT, one image at a time):

| Model | Resolution | Parameters | GFLOPs per image | Inference (ms) | End-to-end (ms) | Peak GPU memory (MB) | Training (h) |
|---|---|---|---|---|---|---|---|
| YOLO11n | 480 | 2.6 M | 3.6 | 7.6 | 10.0 | 65 | 0.73 |
| YOLO11n | 640 | 2.6 M | 6.4 | 7.7 | 10.2 | 94 | 1.15 |
| YOLO11n | 800 | 2.6 M | 10.1 | 7.4 | 10.5 | 120 | 1.72 |
| YOLO11s | 640 | 9.4 M | 21.4 | 7.9 | 10.5 | 143 | 1.35 |

"Inference" is the network forward pass only. "End-to-end" adds image resizing and box post-processing.

YOLO11n at 640 px was selected. It has the best mAP50, and 800 px gives no mAP50 gain for about 50% more training time. YOLO11s needs 3.3× more computation per image and has a lower mAP50.

The inference time is almost the same (about 7.5 ms) for every model, even though the computation per image varies sixfold. At batch size 1, this GPU is limited by fixed per-call overhead, not by arithmetic: it reaches only 0.5–2.7 TFLOP/s. On this hardware, model size therefore affects memory and training time much more than inspection speed.

Final evaluation of the selected model on the withheld test split (80 images):

| Class | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **all** | 79 | 0.510 | 0.488 | 0.519 | 0.269 |
| loose-meat | 3 | 1.000 | 0.553 | 0.694 | 0.334 |
| packaging-error | 23 | 0.377 | 0.522 | 0.497 | 0.161 |
| twisted-meat | 4 | 0.000 | 0.000 | 0.031 | 0.015 |
| unsealed | 49 | 0.664 | 0.878 | 0.856 | 0.566 |

## Limitations

- **`wrinkle` cannot be evaluated**: it has no examples in the validation or test splits.
- **Results for rare classes are unreliable.** `loose-meat` and `twisted-meat` have only 3–4 test boxes each, so a single detection changes their scores a lot.
- Test mAP50 (0.519) is lower than validation mAP50 (0.647). The validation set was used to choose the model, so it gives an optimistic estimate.
- All images come from one dataset, so performance on other cameras, lighting or packaging is unknown.

## AI use

AI tools (Claude Code) were used to help write documentation, tests and reporting scripts. All AI-generated code was reviewed and tested. The full AI-use reflection is in the written project report.

## Licence

The code is released under the [MIT Licence](LICENSE). The dataset is licensed separately under CC BY 4.0 by its Roboflow authors.
