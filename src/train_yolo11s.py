"""
Model-size experiment: train the larger YOLO11 small model at 640
pixels.

**Question:** Does a bigger model, with more capacity to learn, detect
defects better?

**Why it matters:** Larger models can learn subtler patterns, but need
more computation and memory, and with limited data they can memorise the
training photos instead of learning general rules (over-fitting).

**Result:** mAP50 was lower (0.563 vs 0.647), although its boxes were
drawn more tightly (mAP50-95 0.387 vs 0.355). For inspection, finding
the defect matters more than a tight box, so the smaller model was
preferred.

## Settings

| Setting | Value | Meaning |
|---|---|---|
| Starting model | `yolo11s.pt` | YOLO11 small (9.4 million parameters, 3.6 times nano), pretrained on everyday photos (COCO) |
| Resolution | 640 px | Each photo is resized to 640 × 640 before the model sees it |
| Epochs | 50 | Passes through all 3349 training photos |
| Batch size | 8 | Photos processed together per update |
| Seed | 42 | Makes the run repeatable |
| Device | GPU 0 | The first graphics card |

All four training experiments use these same settings apart from the
model and resolution, so their results can be compared fairly.

## Usage

Run from the project root, after downloading the data:

```text
$ python src/train_yolo11s.py
```

Training takes about 1.4 hours on an AMD Radeon RX 9070 XT. Progress is
printed after every epoch.

Ultralytics saves its outputs to
``runs/detect/results/training/yolo11s_baseline/``, including:

- ``weights/best.pt``: the trained model, from the epoch that scored best on the validation photos
- ``weights/last.pt``: the model after the final epoch
- ``results.csv`` and ``results.png``: accuracy and loss for every epoch, to check the training settled down
- ``confusion_matrix.png``: which defect types get mixed up with each other
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)
"""Dataset settings file, telling Ultralytics where the photos and labels are."""


def main():
    """Train the model and save the results (see the module description)."""
    # Load pretrained YOLO11 Small model
    model = YOLO("yolo11s.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="results/training",
        name="yolo11s_baseline",
        pretrained=True,
        seed=42,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
