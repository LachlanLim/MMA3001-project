"""
Train the baseline model: YOLO11 nano at 640-pixel resolution.

**Question:** How well does the smallest YOLO11 model detect pack
defects at the standard resolution?

**Why it matters:** Every other experiment changes exactly one setting
from this baseline (resolution or model size). Any difference in results
can then be traced to that one change, as in a controlled lab
experiment.

**Result:** mAP50 of 0.647 on the validation photos, the best of the
four experiments. This model was selected and scored 0.519 on the unseen
test photos, finding 88% of unsealed packs.

## Settings

| Setting | Value | Meaning |
|---|---|---|
| Starting model | `yolo11n.pt` | YOLO11 nano (2.6 million parameters), pretrained on everyday photos (COCO) |
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
$ python src/train_yolo11n.py
```

Training takes about 1.1 hours on an AMD Radeon RX 9070 XT. Progress is
printed after every epoch.

Ultralytics saves its outputs to
``runs/detect/results/training/yolo11n_baseline/``, including:

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
    # Load pretrained YOLO11 Nano model
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="results/training",
        name="yolo11n_baseline",
        pretrained=True,
        seed=42,
    )

    print("Training completed.")


if __name__ == "__main__":
    main()
