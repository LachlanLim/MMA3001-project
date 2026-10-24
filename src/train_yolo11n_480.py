"""
Train YOLO11n at 480-pixel input resolution.

This module investigates the effect of reducing the input resolution
from the 640-pixel baseline to 480 pixels.

The experiment forms part of the input-resolution sensitivity analysis
for automated pork-rasher packaging defect detection.

All main training parameters are kept consistent with the baseline
to allow a controlled comparison.

Run from the project root:
    python src/train_yolo11n_480.py
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)


def main():
    """
    Train the YOLO11n model using 480-pixel inputs.

    Loads pretrained YOLO11n weights and trains the model on the
    five pork-packaging defect categories.

    Training configuration:
        Epochs: 50
        Image size: 480 pixels
        Batch size: 8
        Device: GPU 0
        Workers: 0
        Random seed: 42

    Model checkpoints, validation metrics, training curves and
    diagnostic figures are saved by Ultralytics under
    ``runs/detect/results/training/yolo11n_480``.
    """
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
