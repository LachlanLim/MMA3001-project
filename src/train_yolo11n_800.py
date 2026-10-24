"""
Resolution experiment: train YOLO11 nano on larger, 800-pixel photos.

**Question:** Does giving the model more detail help it find small
defects?

**Why it matters:** Larger photos keep fine details, such as a thin
unsealed edge, that may be lost when shrinking to 640 px. The cost is
about 1.6 times more computation per photo than at 640 px.

**Result:** No gain: mAP50 was 0.646, against 0.647 at 640 px, while
training took 50% longer (1.72 vs 1.15 hours) and used 28% more GPU
memory. The extra detail did not help, so 640 px was kept.

## Settings

| Setting | Value | Meaning |
|---|---|---|
| Starting model | `yolo11n.pt` | YOLO11 nano (2.6 million parameters), pretrained on everyday photos (COCO) |
| Resolution | 800 px | Each photo is resized to 800 × 800 before the model sees it |
| Epochs | 50 | Passes through all 3349 training photos |
| Batch size | 8 | Photos processed together per update |
| Seed | 42 | Makes the run repeatable |
| Device | GPU 0 | The first graphics card |

All four training experiments use these same settings apart from the
model and resolution, so their results can be compared fairly.

## Usage

Run from the project root, after downloading the data:

```text
$ python src/train_yolo11n_800.py
```

Training takes about 1.7 hours on an AMD Radeon RX 9070 XT. Progress is
printed after every epoch.

Ultralytics saves its outputs to
``runs/detect/results/training/yolo11n_800/``, including:

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
        imgsz=800,
        batch=8,
        device=0,
        workers=0,
        project="results/training",
        name="yolo11n_800",
        pretrained=True,
        seed=42,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
