"""
Choose a practical confidence threshold for the selected model, put error
bars on its test score, and measure how batching changes speed.

mAP summarises the model over *every* confidence threshold, but a real
inspection line must pick *one*: every box above the threshold rejects
the pack. This script answers the questions a line engineer would ask:

- At a given threshold, what fraction of defective packs is caught, and
  what fraction of good packs is wrongly rejected?
- Which threshold should be used? It is chosen on the **validation**
  photos only, then applied once to the test photos, so the test result
  stays an honest estimate.
- How certain is the test mAP50 of 0.519, given only 80 test photos?
  Bootstrapping resamples the test photos 1,000 times to give a 95%
  confidence interval.
- Does checking several photos at once (batching) remove the fixed
  per-operation overhead found by ``src.evaluate``?

## Usage

Run from the project root, after training and ``src/evaluate.py``:

```text
$ python src/operating_point.py
```

It takes about a minute on the training GPU.

## Outputs

- ``results/operating_point.csv``: detection and pack-level results at
  every threshold, for the validation and test photos
- ``results/operating_point_summary.json``: the chosen threshold, its
  test results, the bootstrap interval and a check of the threshold
  Ultralytics uses for its confusion matrix
- ``results/batch_timing.csv``: network time per photo at batch sizes 1,
  8 and 32
"""

import csv
import inspect
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator


DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
"""Folder holding the downloaded dataset."""

WEIGHTS = Path("runs/detect/results/training/yolo11n_baseline/weights/best.pt")
"""Trained weights of the selected model (YOLO11n at 640 px)."""

RESULTS_DIR = Path("results")
"""Where the output tables are saved."""

IMGSZ = 640
"""Input resolution in pixels, matching training."""

IOU_MATCH = 0.5
"""Minimum overlap (IoU) for a predicted box to count as finding a labelled
defect. 0.5 is the threshold used by mAP50."""

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
"""Confidence thresholds to test."""

TARGET_PACK_RECALL = 0.95
"""Acceptance criterion used to choose the threshold: at least 95% of
defective packs must be rejected. Among thresholds that meet this on the
validation photos, the highest is chosen, as it rejects the fewest good
packs."""

BOOTSTRAP_SAMPLES = 1000
"""Number of resampled test sets used for the confidence interval."""

BATCH_SIZES = (1, 8, 32)
"""Batch sizes for the batching experiment."""

TIMING_REPEATS = 30
"""Timed forward passes per batch size."""

DEVICE = 0
"""Graphics card to use (0 is the first one)."""


# --------------------------------------------------
# Predictions and matching
# --------------------------------------------------

def list_images(image_dir):
    """Return the sorted image files in a folder."""
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)


def load_labels(label_path, width, height):
    """
    Read a YOLO label file and convert its boxes to pixel corners.

    Args:
        label_path (Path): The ``.txt`` label file for one photo.
        width (int): Photo width in pixels.
        height (int): Photo height in pixels.

    Returns:
        tuple: ``(boxes, classes)``, where ``boxes`` is an (N, 4) array of
        ``x1, y1, x2, y2`` pixel corners and ``classes`` holds the N class
        IDs. Both are empty for a photo with no defects.
    """
    boxes, classes = [], []

    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])
            boxes.append([
                (x - w / 2) * width, (y - h / 2) * height,
                (x + w / 2) * width, (y + h / 2) * height,
            ])
            classes.append(class_id)

    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int)


def box_iou(a, b):
    """
    Intersection over union between every box in ``a`` and every box in ``b``.

    Args:
        a (numpy.ndarray): (N, 4) boxes as ``x1, y1, x2, y2``.
        b (numpy.ndarray): (M, 4) boxes as ``x1, y1, x2, y2``.

    Returns:
        numpy.ndarray: (N, M) IoU values from 0 to 1.

    Example:
        Two 2 × 2 squares overlapping by half share 2 units of area out
        of 6 in total:

        >>> iou = box_iou(np.array([[0, 0, 2, 2]]), np.array([[1, 0, 3, 2]]))
        >>> round(float(iou[0, 0]), 3)
        0.333
    """
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def match_predictions(record):
    """
    Mark each prediction as a true or false detection.

    Predictions are taken from most to least confident. Each one claims
    the unclaimed labelled box of the same class that it overlaps most,
    if that overlap is at least ``IOU_MATCH``. Because confident
    predictions claim boxes first, raising the threshold simply drops
    predictions from the end of the list, so one matching serves every
    threshold.

    Args:
        record (dict): One photo's predictions and labels; a boolean
            ``"tp"`` array is added in place.
    """
    n_pred = len(record["pred_conf"])
    tp = np.zeros(n_pred, dtype=bool)
    claimed = np.zeros(len(record["gt_cls"]), dtype=bool)

    if n_pred and len(record["gt_cls"]):
        ious = box_iou(record["pred_boxes"], record["gt_boxes"])
        for i in np.argsort(-record["pred_conf"]):
            candidates = (record["gt_cls"] == record["pred_cls"][i]) & ~claimed
            overlap = np.where(candidates, ious[i], 0.0)
            j = int(overlap.argmax())
            if overlap[j] >= IOU_MATCH:
                tp[i] = True
                claimed[j] = True

    record["tp"] = tp


def predict_split(model, split):
    """
    Run the model on every photo of a split and keep all predictions.

    The very low confidence threshold (0.001) matches Ultralytics'
    validation, so thresholds can be applied afterwards.

    Args:
        model (YOLO): Trained model.
        split (str): ``"valid"`` or ``"test"``.

    Returns:
        list[dict]: One record per photo with predicted and labelled boxes.
    """
    records = []

    for path in list_images(DATASET_DIR / split / "images"):
        result = model.predict(
            str(path), imgsz=IMGSZ, conf=0.001, iou=0.7,
            device=DEVICE, verbose=False,
        )[0]
        height, width = result.orig_shape
        gt_boxes, gt_cls = load_labels(
            DATASET_DIR / split / "labels" / f"{path.stem}.txt", width, height
        )
        record = {
            "pred_boxes": result.boxes.xyxy.cpu().numpy(),
            "pred_cls": result.boxes.cls.cpu().numpy().astype(int),
            "pred_conf": result.boxes.conf.cpu().numpy(),
            "gt_boxes": gt_boxes,
            "gt_cls": gt_cls,
        }
        match_predictions(record)
        records.append(record)

    return records


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def metrics_at_threshold(records, threshold):
    """
    Detection and pack-level results at one confidence threshold.

    A pack (photo) is rejected if any prediction reaches the threshold.
    It is defective if it has at least one labelled defect.

    Args:
        records (list[dict]): Matched predictions for one split.
        threshold (float): Confidence threshold.

    Returns:
        dict: Box precision and recall, false detections per photo, the
        fraction of defective packs rejected (pack recall), the fraction
        of good packs rejected (false-reject rate), and unsealed recall.
    """
    tp = fp = n_gt = 0
    unsealed_found = unsealed_total = 0
    defective = defective_rejected = good = good_rejected = 0

    for r in records:
        keep = r["pred_conf"] >= threshold
        tp += int(r["tp"][keep].sum())
        fp += int((~r["tp"][keep]).sum())
        n_gt += len(r["gt_cls"])

        unsealed_total += int((r["gt_cls"] == 3).sum())
        unsealed_found += int((r["tp"] & keep & (r["pred_cls"] == 3)).sum())

        rejected = bool(keep.any())
        if len(r["gt_cls"]):
            defective += 1
            defective_rejected += rejected
        else:
            good += 1
            good_rejected += rejected

    return {
        "threshold": threshold,
        "box_precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "box_recall": round(tp / n_gt, 3) if n_gt else 0.0,
        "false_detections_per_photo": round(fp / len(records), 2),
        "pack_recall": round(defective_rejected / defective, 3) if defective else 0.0,
        "false_reject_rate": round(good_rejected / good, 3) if good else 0.0,
        "unsealed_recall": round(unsealed_found / unsealed_total, 3) if unsealed_total else 0.0,
        "defective_packs": defective,
        "good_packs": good,
    }


def average_precision(records, class_id):
    """
    Average precision at IoU 0.5 for one class.

    Uses the same method as Ultralytics: the precision envelope is
    interpolated at 101 recall points and integrated with the
    trapezoidal rule.

    Args:
        records (list[dict]): Matched predictions.
        class_id (int): Class to score.

    Returns:
        float or None: AP from 0 to 1, or None if the class has no
        labelled boxes in these records.
    """
    n_gt = sum(int((r["gt_cls"] == class_id).sum()) for r in records)
    if n_gt == 0:
        return None

    conf = np.concatenate([r["pred_conf"][r["pred_cls"] == class_id] for r in records])
    tp = np.concatenate([r["tp"][r["pred_cls"] == class_id] for r in records])
    order = np.argsort(-conf)
    tp = tp[order]

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(np.interp(x, mrec, mpre), x))


def mean_ap50(records):
    """Mean AP50 over the classes that have labelled boxes."""
    aps = [average_precision(records, c) for c in range(5)]
    aps = [a for a in aps if a is not None]
    return float(np.mean(aps)) if aps else float("nan")


def bootstrap_map50(records, samples, seed=42):
    """
    95% confidence interval for mAP50 by resampling photos.

    Args:
        records (list[dict]): Matched test predictions.
        samples (int): Number of resampled test sets.
        seed (int): Random seed, so the interval is repeatable.

    Returns:
        tuple: (2.5th percentile, 97.5th percentile).
    """
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(samples):
        idx = rng.integers(0, len(records), len(records))
        scores.append(mean_ap50([records[i] for i in idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def confusion_matrix_threshold_source():
    """
    Return the Ultralytics code lines that set the confusion matrix threshold.

    ``ConfusionMatrix`` has a default of 0.25, but during validation the
    validator overrides it with its own ``conf`` setting, which defaults
    to 0.001. Both places are returned, so the threshold actually used can
    be checked.

    Returns:
        list[str]: Matching source lines, each prefixed with its class.
    """
    sources = {
        "DetectionValidator": inspect.getsource(DetectionValidator),
        "BaseValidator": inspect.getsource(BaseValidator),
    }
    lines = []
    for name, source in sources.items():
        for line in source.splitlines():
            if "confusion_matrix.process_batch" in line or "self.args.conf = " in line:
                lines.append(f"{name}: {line.strip()}")
    return lines


# --------------------------------------------------
# Batching
# --------------------------------------------------

def time_batches(model):
    """
    Time the network forward pass at several batch sizes.

    Random input of the right shape is used, since only the arithmetic
    and overhead are being timed, not accuracy.

    Args:
        model (YOLO): Trained model.

    Returns:
        list[dict]: Batch size, milliseconds per photo and photos per second.
    """
    device = torch.device(f"cuda:{DEVICE}")
    network = model.model.to(device).eval()
    rows = []

    for batch in BATCH_SIZES:
        x = torch.rand(batch, 3, IMGSZ, IMGSZ, device=device)
        with torch.inference_mode():
            for _ in range(5):
                network(x)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(TIMING_REPEATS):
                network(x)
            torch.cuda.synchronize()
        ms_per_photo = (time.perf_counter() - start) / TIMING_REPEATS / batch * 1000
        rows.append({
            "batch_size": batch,
            "ms_per_photo": round(ms_per_photo, 2),
            "photos_per_second": round(1000 / ms_per_photo),
        })
        print(f"batch {batch:>2}: {ms_per_photo:.2f} ms per photo")

    return rows


def write_csv(path, rows):
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def main():
    """Run the threshold study, the bootstrap and the batching experiment."""
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for consistent timing results.")

    model = YOLO(str(WEIGHTS))

    valid = predict_split(model, "valid")
    test = predict_split(model, "test")

    rows = []
    for split, records in (("valid", valid), ("test", test)):
        for t in THRESHOLDS:
            rows.append({"split": split, **metrics_at_threshold(records, t)})
    write_csv(RESULTS_DIR / "operating_point.csv", rows)

    # Choose the threshold on validation photos only
    valid_rows = [r for r in rows if r["split"] == "valid"]
    meeting = [r for r in valid_rows if r["pack_recall"] >= TARGET_PACK_RECALL]
    chosen = max(meeting, key=lambda r: r["threshold"]) if meeting else max(
        valid_rows, key=lambda r: r["pack_recall"]
    )
    threshold = chosen["threshold"]

    low, high = bootstrap_map50(test, BOOTSTRAP_SAMPLES)

    summary = {
        "target_pack_recall": TARGET_PACK_RECALL,
        "chosen_threshold": threshold,
        "valid_at_chosen": chosen,
        "test_at_chosen": metrics_at_threshold(test, threshold),
        "test_at_0.25": metrics_at_threshold(test, 0.25),
        "valid_map50_recomputed": round(mean_ap50(valid), 3),
        "test_map50_recomputed": round(mean_ap50(test), 3),
        "test_map50_95ci": [round(low, 3), round(high, 3)],
        "test_false_detections_at_0.25": int(sum(
            (~r["tp"][r["pred_conf"] >= 0.25]).sum() for r in test
        )),
        "confusion_matrix_threshold_code": confusion_matrix_threshold_source(),
        "batch_timing": time_batches(model),
    }

    with open(RESULTS_DIR / "operating_point_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'operating_point_summary.json'}")

    write_csv(RESULTS_DIR / "batch_timing.csv", summary["batch_timing"])

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
