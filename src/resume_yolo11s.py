"""
Resume an interrupted YOLO11s training experiment.

This module resumes the YOLO11s comparison experiment using
the last.pt checkpoint saved by Ultralytics.

The checkpoint preserves training progress so that the experiment
can continue towards its original 50-epoch target without
starting from epoch 1 again.

The script is intended for recovery from interrupted training,
not for starting a new experiment.

Run from the project root:
    python src/resume_yolo11s.py
"""

from pathlib import Path
from ultralytics import YOLO

CHECKPOINT = Path(
    "runs/detect/results/training/yolo11s_baseline/weights/last.pt"
)


def main():
    """
    Resume YOLO11s training from its latest checkpoint.

    Loads the last.pt checkpoint from the previous YOLO11s
    experiment and continues training using the saved settings.
    Updated results and checkpoints are written back to the
    original Ultralytics experiment directory.

    Raises:
        FileNotFoundError: If the last.pt checkpoint does not exist.
    """
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT}. "
            "Run src/train_yolo11s.py first."
        )

    model = YOLO(str(CHECKPOINT))

    model.train(resume=True)


if __name__ == "__main__":
    main()
