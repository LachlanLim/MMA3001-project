from pathlib import Path
import yaml

DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
YAML_PATH = DATASET_DIR / "data.yaml"

# Read dataset configuration
with open(YAML_PATH, "r") as f:
    config = yaml.safe_load(f)

print("Dataset configuration:")
print(config)

print("\nClass names:")
print(config["names"])

# Count images and labels in each split
for split in ["train", "valid", "test"]:
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    images = list(image_dir.glob("*"))
    labels = list(label_dir.glob("*.txt"))

    print(f"\n{split.upper()}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")