"""
Continue the YOLO11 small training run after an interruption.

Training takes over an hour, and can be cut short by a crash, a power
cut or the computer going to sleep. Ultralytics saves a checkpoint
(``last.pt``) after every epoch, holding the model and the training
progress. This script picks up from that checkpoint instead of starting
again from epoch 1.

This happened during the project: the YOLO11 small run stopped after
epoch 18 and was resumed with this script to finish all 50 epochs.

Use it only to recover an interrupted run, not to start a new one.

## Usage

Run from the project root:

```text
$ python src/resume_yolo11s.py
Resuming training ... from epoch 19 to 50 total epochs
```

**Note:** resuming restarts the training clock in Ultralytics'
``results.csv``. ``src.evaluate`` allows for this when it reports the
total training time.
"""

from pathlib import Path
from ultralytics import YOLO

CHECKPOINT = Path(
    "runs/detect/results/training/yolo11s_baseline/weights/last.pt"
)
"""The checkpoint saved after the last completed epoch, which training resumes from."""


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
