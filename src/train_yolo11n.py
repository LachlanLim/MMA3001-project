"""
Train the YOLO11n baseline model at 640-pixel input resolution.

This is the baseline experiment. The 480- and 800-pixel experiments
and the YOLO11s comparison use the same settings so that only one
factor changes at a time.

Run from the project root:
    python src/train_yolo11n.py
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)


def main():
    """
    Fine-tune pretrained YOLO11n weights on the pork-packaging dataset.

    Training configuration:
        Epochs: 50
        Image size: 640 pixels
        Batch size: 8
        Device: GPU 0
        Workers: 0
        Random seed: 42

    Model checkpoints, validation metrics, training curves and
    diagnostic figures are saved by Ultralytics under
    ``runs/detect/results/training/yolo11n_baseline``.
    """
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
