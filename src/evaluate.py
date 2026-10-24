"""
Evaluate the trained models and measure their computational cost.

For each training experiment, this module:

1. measures accuracy (precision, recall, mAP50, mAP50-95) on the
   validation split;
2. measures model size (parameters) and computation per image (GFLOPs);
3. times inference on the validation images one at a time, as on an
   inspection line, and records peak GPU memory;
4. reads the training time from the Ultralytics results.csv.

The selected model is then evaluated once on the withheld test split.

All accuracy and timing results come from the trained ``best.pt``
weights, so no retraining is needed. Run the training scripts first.

Outputs:
    results/model_comparison.csv:
        YOLO11n vs YOLO11s at 640 pixels.
    results/resolution_comparison.csv:
        YOLO11n at 480, 640 and 800 pixels.
    results/final_test_metrics.csv:
        Per-class test-split metrics for the selected model.
    runs/detect/evaluation/:
        Ultralytics plots (confusion matrix, PR curves) for each run.

Run from the project root:
    python src/evaluate.py
"""

import csv
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
DATA_YAML = DATASET_DIR / "data.yaml"

TRAINING_DIR = Path("runs/detect/results/training")
# Ultralytics saves relative project folders inside runs/detect
EVALUATION_DIR = Path("evaluation")
RESULTS_DIR = Path("results")

# Experiment name, model label, input resolution and Ultralytics run folder
EXPERIMENTS = [
    ("yolo11n_480", "YOLO11n", 480, "yolo11n_480"),
    ("yolo11n_640", "YOLO11n", 640, "yolo11n_baseline"),
    ("yolo11n_800", "YOLO11n", 800, "yolo11n_800"),
    ("yolo11s_640", "YOLO11s", 640, "yolo11s_baseline"),
]

# The model chosen from the validation results, evaluated on the test split
SELECTED_EXPERIMENT = "yolo11n_640"

# Images run before timing starts, so GPU start-up cost is not measured
WARMUP_IMAGES = 10

DEVICE = 0


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

def validate(model, imgsz, split, name, plots=False):
    """
    Run Ultralytics validation on one dataset split.

    Args:
        model (YOLO): Trained model.
        imgsz (int): Input resolution in pixels, matching training.
        split (str): ``"val"`` or ``"test"``.
        name (str): Output folder name inside ``runs/detect/evaluation``.
        plots (bool): Whether to save confusion matrices and PR curves.

    Returns:
        ultralytics.utils.metrics.DetMetrics: Detection metrics.
    """
    return model.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        batch=8,
        device=DEVICE,
        workers=0,
        plots=plots,
        verbose=False,
        project=str(EVALUATION_DIR),
        name=name,
        exist_ok=True,
    )


# --------------------------------------------------
# Computational cost
# --------------------------------------------------

def time_inference(model, images, imgsz):
    """
    Time single-image inference and record peak GPU memory.

    Images are processed one at a time (batch size 1), which matches an
    inspection line where each pack is checked as it arrives. The GPU is
    synchronised around each call so the timer measures finished work.

    Args:
        model (YOLO): Trained model.
        images (list[PIL.Image.Image]): Images preloaded into memory, so
            disk reading is not included in the timing.
        imgsz (int): Input resolution in pixels.

    Returns:
        dict: Median end-to-end time per image (``end_to_end_ms``,
        including pre- and post-processing), median network-only time
        (``inference_ms``) and peak GPU memory (``peak_memory_mb``).
    """
    for image in images[:WARMUP_IMAGES]:
        model.predict(image, imgsz=imgsz, device=DEVICE, verbose=False)

    torch.cuda.reset_peak_memory_stats()
    end_to_end_ms = []
    inference_ms = []

    for image in images:
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = model.predict(image, imgsz=imgsz, device=DEVICE, verbose=False)[0]

        torch.cuda.synchronize()
        end_to_end_ms.append((time.perf_counter() - start) * 1000)

        # Ultralytics also times the network forward pass on its own
        inference_ms.append(result.speed["inference"])

    # The median is less affected by occasional slow calls than the mean
    return {
        "end_to_end_ms": statistics.median(end_to_end_ms),
        "inference_ms": statistics.median(inference_ms),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def training_hours(run_name):
    """
    Read the total training time from an Ultralytics results.csv.

    The ``time`` column holds the elapsed seconds at the end of each
    epoch. It restarts from zero when training is resumed, so each
    training session is added separately.

    Args:
        run_name (str): Folder name inside ``runs/detect/results/training``.

    Returns:
        float: Training time in hours, or None if results.csv is missing.
    """
    results_csv = TRAINING_DIR / run_name / "results.csv"

    if not results_csv.is_file():
        return None

    with open(results_csv, "r", encoding="utf-8") as f:
        times = [float(row["time"].strip()) for row in csv.DictReader(f)]

    # A resumed run restarts the clock, so add up each session's last time
    total_seconds = times[-1]
    for previous, current in zip(times, times[1:]):
        if current < previous:
            total_seconds += previous

    return total_seconds / 3600


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def evaluate_experiment(name, label, imgsz, run_name, images):
    """
    Measure the accuracy and cost of one trained model.

    Args:
        name (str): Experiment name, e.g. ``"yolo11n_640"``.
        label (str): Model label for the tables, e.g. ``"YOLO11n"``.
        imgsz (int): Input resolution in pixels.
        run_name (str): Ultralytics training run folder.
        images (list[PIL.Image.Image]): Validation images for timing.

    Returns:
        dict: One row of the comparison tables.

    Raises:
        FileNotFoundError: If the run has no best.pt weights.
    """
    weights = TRAINING_DIR / run_name / "weights" / "best.pt"

    if not weights.is_file():
        raise FileNotFoundError(f"No trained weights at {weights}.")

    print(f"\n=== {name} ===")
    model = YOLO(str(weights))

    metrics = validate(model, imgsz, "val", f"{name}_val")
    timing = time_inference(model, images, imgsz)

    # Count the fused model (BatchNorm folded into the convolutions),
    # which is the version that actually runs during inference
    model.fuse()
    parameters = sum(p.numel() for p in model.model.parameters())
    gflops = get_flops(model.model, imgsz)

    # Rate of useful computation actually achieved during the forward pass
    achieved_gflops_per_s = gflops / (timing["inference_ms"] / 1000)

    hours = training_hours(run_name)

    return {
        "model": label,
        "resolution": imgsz,
        "precision": round(metrics.box.mp, 3),
        "recall": round(metrics.box.mr, 3),
        "mAP50": round(metrics.box.map50, 3),
        "mAP50_95": round(metrics.box.map, 3),
        "parameters": parameters,
        "gflops": round(gflops, 1),
        "inference_ms": round(timing["inference_ms"], 2),
        "end_to_end_ms": round(timing["end_to_end_ms"], 2),
        "achieved_gflops_per_s": round(achieved_gflops_per_s),
        "peak_memory_mb": round(timing["peak_memory_mb"]),
        "training_hours": round(hours, 3) if hours is not None else "",
    }


def evaluate_test_split(name, imgsz, run_name):
    """
    Evaluate the selected model once on the withheld test split.

    Args:
        name (str): Experiment name.
        imgsz (int): Input resolution in pixels.
        run_name (str): Ultralytics training run folder.

    Returns:
        list[dict]: An ``all`` row followed by one row per class that
        appears in the test split.
    """
    print(f"\n=== Final test: {name} ===")
    model = YOLO(str(TRAINING_DIR / run_name / "weights" / "best.pt"))
    metrics = validate(model, imgsz, "test", f"{name}_test", plots=True)

    test_images = list_images(DATASET_DIR / "test" / "images")

    rows = [{
        "class": "all",
        "images": len(test_images),
        "instances": int(metrics.nt_per_class.sum()),
        "precision": round(metrics.box.mp, 3),
        "recall": round(metrics.box.mr, 3),
        "mAP50": round(metrics.box.map50, 3),
        "mAP50_95": round(metrics.box.map, 3),
    }]

    for row in metrics.summary(decimals=3):
        rows.append({
            "class": row["Class"],
            "images": int(row["Images"]),
            "instances": int(row["Instances"]),
            "precision": row["Box-P"],
            "recall": row["Box-R"],
            "mAP50": row["mAP50"],
            "mAP50_95": row["mAP50-95"],
        })

    return rows


def list_images(image_dir):
    """Return the sorted image files in a folder."""
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)


def write_csv(path, rows):
    """Write a list of dicts to a CSV file, using the first row's keys."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}")


def main():
    """
    Evaluate every experiment and save the comparison tables.

    Raises:
        RuntimeError: If no GPU is available, since timing results on a
            CPU would not be comparable with the training hardware.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for consistent timing results.")

    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

    # Preload the validation images so timing excludes disk reads
    images = [
        Image.open(path).convert("RGB")
        for path in list_images(DATASET_DIR / "valid" / "images")
    ]

    rows = {}
    for name, label, imgsz, run_name in EXPERIMENTS:
        rows[name] = evaluate_experiment(name, label, imgsz, run_name, images)

    write_csv(
        RESULTS_DIR / "model_comparison.csv",
        [rows["yolo11n_640"], rows["yolo11s_640"]],
    )
    write_csv(
        RESULTS_DIR / "resolution_comparison.csv",
        [rows["yolo11n_480"], rows["yolo11n_640"], rows["yolo11n_800"]],
    )

    # The test split is used only once, after the model has been chosen
    selected = next(e for e in EXPERIMENTS if e[0] == SELECTED_EXPERIMENT)
    write_csv(
        RESULTS_DIR / "final_test_metrics.csv",
        evaluate_test_split(selected[0], selected[2], selected[3]),
    )


if __name__ == "__main__":
    main()
