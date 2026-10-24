"""
Show the defect types and count the photos in each part of the dataset.

Run this straight after downloading, as a quick check that the download
is complete before spending hours on training. Every photo should have
exactly one label file; a mismatch means files are missing or corrupt.

## Usage

Run from the project root:

```text
$ python src/inspect_dataset.py
Dataset configuration:
{'train': '../train/images', 'val': '../valid/images', 'test': '../test/images',
 'nc': 5, 'names': ['loose-meat', 'packaging-error', 'twisted-meat',
 'unsealed', 'wrinkle'], 'roboflow': {...}}

Class names:
['loose-meat', 'packaging-error', 'twisted-meat', 'unsealed', 'wrinkle']

TRAIN
Images: 3349
Labels: 3349

VALID
Images: 120
Labels: 120

TEST
Images: 80
Labels: 80
```

Here every split has matching image and label counts, so the download
is complete. The same checks run automatically in
``tests/test_dataset.py``.
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
