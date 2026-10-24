"""
Train the larger YOLO11s model at 640-pixel input resolution.

This experiment is the model-size comparison against the YOLO11n
baseline. All training settings match the baseline, so differences in
accuracy and speed come from the model size alone.

Run from the project root:
    python src/train_yolo11s.py
"""

from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)


def main():
    """
    Fine-tune pretrained YOLO11s weights on the pork-packaging dataset.

    Training configuration:
        Epochs: 50
        Image size: 640 pixels
        Batch size: 8
        Device: GPU 0
        Workers: 0
        Random seed: 42

    Model checkpoints, validation metrics, training curves and
    diagnostic figures are saved by Ultralytics under
    ``runs/detect/results/training/yolo11s_baseline``. If training is
    interrupted, continue it with ``src/resume_yolo11s.py``.
    """
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
