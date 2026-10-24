"""
Inspect the pork-rasher packaging defect dataset.

This module reads the dataset's data.yaml configuration and
checks the number of images and YOLO annotation files in the
training, validation and test splits.

The inspection helps verify the dataset structure before
training the YOLO11 object-detection models.

Run from the project root:
    python src/inspect_dataset.py
"""

from pathlib import Path
import yaml


# Dataset configuration paths, relative to the project root.
DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
YAML_PATH = DATASET_DIR / "data.yaml"


def main():
    """
    Display the dataset configuration and split file counts.

    Reads the YAML configuration to identify the defect categories,
    then counts image files and annotation text files in each split.

    Prints the configuration, the class names and the number of
    image and label files in the train, valid and test splits.

    Raises:
        FileNotFoundError: If data.yaml does not exist.
        yaml.YAMLError: If data.yaml cannot be parsed.
    """

    # Load the dataset configuration.
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Dataset configuration:")
    print(config)

    print("\nClass names:")
    print(config["names"])

    # Count the image and annotation files in each split.
    for split in ["train", "valid", "test"]:
        image_dir = DATASET_DIR / split / "images"
        label_dir = DATASET_DIR / split / "labels"

        images = list(image_dir.glob("*"))
        labels = list(label_dir.glob("*.txt"))

        print(f"\n{split.upper()}")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")


if __name__ == "__main__":
    main()
