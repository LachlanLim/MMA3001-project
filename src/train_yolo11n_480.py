"""
Resolution experiment: train YOLO11 nano on smaller, 480-pixel photos.

**Question:** Can the photos be shrunk to save computation without
losing accuracy?

**Why it matters:** Smaller photos need about 44% less computation than
640 px, which matters if the model must run on cheap hardware beside a
packing line. But small defects such as a loose strip of meat may become
too few pixels to see.

**Result:** mAP50 fell from 0.647 (640 px) to 0.576, so detail was lost.
Training was faster (0.73 vs 1.15 hours), but checking a photo was no
faster on this GPU (see `src.evaluate`).

## Settings

| Setting | Value | Meaning |
|---|---|---|
| Starting model | `yolo11n.pt` | YOLO11 nano (2.6 million parameters), pretrained on everyday photos (COCO) |
| Resolution | 480 px | Each photo is resized to 480 × 480 before the model sees it |
| Epochs | 50 | Passes through all 3349 training photos |
| Batch size | 8 | Photos processed together per update |
| Seed | 42 | Makes the run repeatable |
| Device | GPU 0 | The first graphics card |

All four training experiments use these same settings apart from the
model and resolution, so their results can be compared fairly.

## Usage

Run from the project root, after downloading the data:

```text
$ python src/train_yolo11n_480.py
```

Training takes about 45 minutes on an AMD Radeon RX 9070 XT. Progress is
printed after every epoch.

Ultralytics saves its outputs to
``runs/detect/results/training/yolo11n_480/``, including:

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


def main():
    """Train the model and save the results (see the module description)."""
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=480,
        batch=8,
        device=0,
        workers=0,
        project="results/training",
        name="yolo11n_480",
        pretrained=True,
        seed=42,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
