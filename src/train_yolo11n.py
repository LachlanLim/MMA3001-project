from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)


def main():
    # Load pretrained YOLO11 Nano model
    model = YOLO("yolo11n.pt")

    # Short GPU smoke test
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