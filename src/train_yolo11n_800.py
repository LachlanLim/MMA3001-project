"""
Train YOLO11n at 800-pixel input resolution.

This module investigates the effect of increasing the input
resolution from the 640-pixel baseline to 800 pixels.

The experiment forms part of the resolution sensitivity analysis
for automated pork-rasher packaging defect detection.

The model architecture and main training parameters are kept
consistent with the 480- and 640-pixel experiments to allow
comparison of detection performance and computational cost.

Run from the project root:
    python src/train_yolo11n_800.py
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)


def main():
    """
    Train the YOLO11n model using 800-pixel inputs.

    Loads pretrained YOLO11n weights and fine-tunes the model
    using the five annotated pork-packaging defect categories.

    Training configuration:
        Epochs: 50
        Image size: 800 pixels
        Batch size: 8
        Device: GPU 0
        Workers: 0
        Random seed: 42

    Model checkpoints, validation metrics, training curves and
    diagnostic figures are saved by Ultralytics under
    ``runs/detect/results/training/yolo11n_800``.
    """
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
