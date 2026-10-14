from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path(
    "data/Pork-Rasher-Error(Packaging)-4/data.yaml"
)

# Load pretrained YOLO11 Nano model
model = YOLO("yolo11n.pt")

# Short smoke test
model.train(
    data=str(DATA_YAML),
    epochs=2,
    imgsz=640,
    batch=8,
    project="results/training",
    name="yolo11n_smoke_test",
    pretrained=True,
    seed=42
)

print("Training completed.")