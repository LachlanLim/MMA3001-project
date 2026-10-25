"""
Measure how well each trained model works, and how much it costs to run.

Training produced four candidate models. This script compares them
fairly, chooses one, and gives that one a final exam on photos it has
never been judged on. It does no training itself: it loads each saved
model from ``models/`` and checks photos with it, so it finishes in about a
minute.

## What is measured, and why

| Measurement | Plain meaning | Why it matters on a packing line |
|---|---|---|
| Precision | Of the defects reported, how many were real | Low precision means good packs are wrongly rejected |
| Recall | Of the real defects, how many were found | Low recall means faulty packs reach customers |
| mAP50 | Overall accuracy score from 0 to 1 (see the glossary in `src`) | The main score used to choose the model |
| GFLOPs | Arithmetic needed to check one photo | Whether cheaper hardware could run it |
| Inference time | Time for the model to check one photo | Whether it keeps up with the line |
| Peak memory | Graphics-card memory used | Whether it fits on smaller devices |
| Training time | Hours to train | Cost of retraining when new defect types appear |
| Profile | Which operations the processor spends its time on | Explains *why* the speeds are what they are |

Speed is measured one photo at a time, as on a line where packs arrive
one by one. Ten photos are checked first as a warm-up (the first few
are always slow while the graphics card starts up). Then all 120
validation photos are timed, and the median is reported so a single
slow photo does not distort it.

## Fair testing

The photos are split three ways, like study material, practice exams and
a final exam:

1. The models **learned** from the 3349 training photos.
2. The four models are **compared** on the 120 validation photos, and
   the best is chosen.
3. Only the chosen model is **tested once** on the 80 test photos.

The test photos are kept back until the choice is made. If they were
used to choose, the model would be picked partly for luck on those
particular photos, and its score would overstate how well it works on
new packs.

## Usage

Run from the project root, after training all four models:

```text
$ python src/evaluate.py
GPU: AMD Radeon RX 9070 XT

=== yolo11n_480 ===
...
Saved results/model_comparison.csv
Saved results/resolution_comparison.csv

=== Final test: yolo11n_640 ===
...
Saved results/final_test_metrics.csv
```

## Results

Comparison on the validation photos (AMD Radeon RX 9070 XT):

| Model | Resolution | mAP50 | GFLOPs | Inference (ms) | Memory (MB) | Training (h) |
|---|---|---|---|---|---|---|
| YOLO11n | 480 | 0.576 | 3.6 | 7.5 | 32 | 0.73 |
| **YOLO11n** | **640** | **0.647** | 6.4 | 7.6 | 39 | 1.15 |
| YOLO11n | 800 | 0.646 | 10.1 | 7.4 | 50 | 1.72 |
| YOLO11s | 640 | 0.563 | 21.4 | 7.9 | 109 | 1.35 |

YOLO11n at 640 px was chosen: it is the most accurate, and the larger
options cost more memory and training time for no gain.

Final test of the chosen model (80 unseen photos):

| Defect | Defects in test photos | Precision | Recall | mAP50 |
|---|---|---|---|---|
| **all** | 79 | 0.510 | 0.488 | 0.519 |
| unsealed | 49 | 0.664 | 0.878 | 0.856 |
| packaging-error | 23 | 0.377 | 0.522 | 0.497 |
| twisted-meat | 4 | 0.000 | 0.000 | 0.031 |
| loose-meat | 3 | 1.000 | 0.553 | 0.694 |
| wrinkle | 0 | - | - | - |

**Reading the results:** the model finds 88% of unsealed packs, the
most safety-critical defect. It is weak on rare defect types:
twisted-meat had only 72 training examples, and its single test score
rests on just 4 boxes, so it is not reliable. Wrinkle cannot be tested
at all, because the dataset has no wrinkle examples outside the
training photos.

**A surprising result:** every model takes about the same time per photo
(7.4 to 7.9 ms), even though the largest needs six times the arithmetic
of the smallest.

The profiler explains why:

- Every model runs the same 88 convolution layers and about 1,200
  PyTorch operations per photo. The larger model does more arithmetic
  *inside* each operation, but not more operations.
- Preparing and sending each operation to the graphics card has a fixed
  cost. Together, these costs come to about 7 ms of processor time per
  photo, which is roughly the whole inference time.
- The graphics card finishes the arithmetic faster than the processor
  can hand it new work, so it spends most of its time waiting.

On this hardware, model size therefore affects memory and training time,
not inspection speed. Checking several photos at once (batching) would
spread the fixed cost and make the arithmetic matter again.

## Outputs

- ``results/model_comparison.csv``: YOLO11n vs YOLO11s at 640 px
- ``results/resolution_comparison.csv``: YOLO11n at 480, 640 and 800 px
- ``results/final_test_metrics.csv``: per-defect test results for the
  chosen model
- ``results/profiles/<experiment>.txt``: profiler breakdown of the
  operations that took the most processor time
- ``runs/detect/evaluation/``: Ultralytics plots (confusion matrices,
  precision-recall curves)
"""

import csv
import gc
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
"""Folder holding the downloaded dataset."""

DATA_YAML = DATASET_DIR / "data.yaml"
"""Dataset settings file listing the defect types and photo folders."""

MODELS_DIR = Path("models")
"""The four trained models, named after their experiment (e.g.
``yolo11n_640.pt``). Each is a copy of the ``weights/best.pt`` saved by
its training run, so the evaluation runs without retraining."""

EVALUATION_DIR = Path("evaluation")
"""Where Ultralytics saves evaluation plots. Ultralytics puts relative
folders inside ``runs/detect``, so this becomes ``runs/detect/evaluation``."""

RESULTS_DIR = Path("results")
"""Where the comparison tables (CSV files) are saved."""

PROFILES_DIR = RESULTS_DIR / "profiles"
"""Where the profiler's breakdown for each model is saved."""

EXPERIMENTS = [
    ("yolo11n_480", "YOLO11n", 480),
    ("yolo11n_640", "YOLO11n", 640),
    ("yolo11n_800", "YOLO11n", 800),
    ("yolo11s_640", "YOLO11s", 640),
]
"""The four trained models to compare, each given as
(experiment name, model label, resolution in pixels)."""

SELECTED_EXPERIMENT = "yolo11n_640"
"""The model chosen from the validation results. Only this one is
evaluated on the test photos."""

WARMUP_IMAGES = 10
"""Photos checked before timing starts, so the graphics card's one-off
start-up delay is not counted."""

PROFILE_RUNS = 20
"""Photos checked while the profiler records, to average out variation."""

DEVICE = 0
"""Graphics card to use (0 is the first one)."""


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


def profile_inference(model, image, imgsz, name):
    """
    Profile where the processor's time goes when checking one photo.

    Uses the PyTorch profiler to record every operation PyTorch runs
    (convolutions, additions, reshapes and so on) while checking the same
    photo ``PROFILE_RUNS`` times. Each operation has a fixed cost to
    prepare and send to the graphics card, however little arithmetic it
    does. Counting operations therefore shows how much of the time is
    this fixed overhead rather than arithmetic.

    Only processor-side (CPU) activity is recorded. The profiler cannot
    record work on the graphics card itself for AMD cards on Windows.
    The profiler also slows the code down, so its times are useful for
    comparing operations, not as absolute speeds.

    Args:
        model (YOLO): Trained model.
        image (PIL.Image.Image): Photo to check.
        imgsz (int): Input resolution in pixels.
        name (str): Experiment name, used for the output file name.

    Returns:
        dict: PyTorch operations per photo (``operations_per_image``,
        including operations called inside other operations), convolution
        layers run per photo (``convolutions_per_image``), and processor
        time per photo while profiled (``profiled_cpu_ms``).

    Example:
        For YOLO11n at 640 px, the saved breakdown in
        ``results/profiles/yolo11n_640.txt`` starts with the operations
        that took the most processor time:

        ```text
        Name                       Self CPU %   ...   # of Calls
        aten::miopen_convolution       52.35%   ...         1620
        aten::nonzero                   7.17%   ...           60
        ...
        ```
    """
    for _ in range(WARMUP_IMAGES):
        model.predict(image, imgsz=imgsz, device=DEVICE, verbose=False)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(PROFILE_RUNS):
            model.predict(image, imgsz=imgsz, device=DEVICE, verbose=False)

    events = prof.key_averages()

    # "aten::" operations are PyTorch's own building blocks
    operations = sum(e.count for e in events if e.key.startswith("aten::"))
    convolutions = sum(e.count for e in events if e.key == "aten::conv2d")
    cpu_ms = sum(e.self_cpu_time_total for e in events) / 1000

    summary = {
        "operations_per_image": round(operations / PROFILE_RUNS),
        "convolutions_per_image": round(convolutions / PROFILE_RUNS),
        "profiled_cpu_ms": round(cpu_ms / PROFILE_RUNS, 1),
    }

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILES_DIR / f"{name}.txt", "w", encoding="utf-8") as f:
        f.write(f"PyTorch profile: {name}, {PROFILE_RUNS} photos at {imgsz} px\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(events.table(sort_by="self_cpu_time_total", row_limit=15))

    return summary


def training_hours(name):
    """
    Read the total training time from an Ultralytics results.csv.

    Each training run's ``results.csv`` is kept in ``results/<name>/``.

    The ``time`` column holds the elapsed seconds at the end of each
    epoch. It restarts from zero when training is resumed, so each
    training session is added separately.

    Args:
        name (str): Experiment name, e.g. ``"yolo11n_640"``.

    Returns:
        float: Training time in hours, or None if results.csv is missing.
    """
    results_csv = RESULTS_DIR / name / "results.csv"

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

def evaluate_experiment(name, label, imgsz, images):
    """
    Measure the accuracy and cost of one trained model.

    Args:
        name (str): Experiment name, e.g. ``"yolo11n_640"``.
        label (str): Model label for the tables, e.g. ``"YOLO11n"``.
        imgsz (int): Input resolution in pixels.
        images (list[PIL.Image.Image]): Validation images for timing.

    Returns:
        dict: One row of the comparison tables.

    Raises:
        FileNotFoundError: If the model file is missing from ``models/``.
    """
    weights = MODELS_DIR / f"{name}.pt"

    if not weights.is_file():
        raise FileNotFoundError(f"No trained weights at {weights}.")

    print(f"\n=== {name} ===")

    # Free memory left over from the previous model, then record what is
    # still in use, so it is not counted as this model's memory
    gc.collect()
    torch.cuda.empty_cache()
    baseline_mb = torch.cuda.memory_allocated() / 1e6
    print(f"GPU memory in use before loading: {baseline_mb:.0f} MB")

    model = YOLO(str(weights))

    metrics = validate(model, imgsz, "val", f"{name}_val")
    timing = time_inference(model, images, imgsz)
    profiling = profile_inference(model, images[0], imgsz, name)

    # Count the fused model (BatchNorm folded into the convolutions),
    # which is the version that actually runs during inference
    model.fuse()
    parameters = sum(p.numel() for p in model.model.parameters())
    gflops = get_flops(model.model, imgsz)

    # Rate of useful computation actually achieved during the forward pass
    achieved_gflops_per_s = gflops / (timing["inference_ms"] / 1000)

    hours = training_hours(name)

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
        "peak_memory_mb": round(timing["peak_memory_mb"] - baseline_mb),
        "operations_per_image": profiling["operations_per_image"],
        "convolutions_per_image": profiling["convolutions_per_image"],
        "training_hours": round(hours, 3) if hours is not None else "",
    }


def evaluate_test_split(name, imgsz):
    """
    Evaluate the selected model once on the withheld test split.

    Args:
        name (str): Experiment name.
        imgsz (int): Input resolution in pixels.

    Returns:
        list[dict]: An ``all`` row followed by one row per class that
        appears in the test split.
    """
    print(f"\n=== Final test: {name} ===")
    model = YOLO(str(MODELS_DIR / f"{name}.pt"))
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

    # The first model run on the GPU also sets up memory that stays in use
    # afterwards. Run one model briefly first, so this one-off memory is not
    # counted against whichever model happens to be measured first.
    name, _, imgsz = EXPERIMENTS[0]
    warmup_model = YOLO(str(MODELS_DIR / f"{name}.pt"))
    warmup_model.predict(images[0], imgsz=imgsz, device=DEVICE, verbose=False)
    del warmup_model

    rows = {}
    for name, label, imgsz in EXPERIMENTS:
        rows[name] = evaluate_experiment(name, label, imgsz, images)

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
        evaluate_test_split(selected[0], selected[2]),
    )


if __name__ == "__main__":
    main()
