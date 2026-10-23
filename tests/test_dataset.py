
"""Tests for the pork-packaging defect dataset configuration."""

from pathlib import Path
import yaml


DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
YAML_PATH = DATASET_DIR / "data.yaml"

EXPECTED_CLASSES = [
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
]


def test_dataset_configuration_exists():
    """Check that the required dataset configuration is available."""
    assert YAML_PATH.is_file(), "Dataset data.yaml is missing."


def test_dataset_has_five_classes():
    """Check that the dataset declares five defect categories."""
    with YAML_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["nc"] == 5


def test_class_names():
    """Check that the class names and their order are correct."""
    with YAML_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["names"] == EXPECTED_CLASSES


def test_dataset_split_directories():
    """Check that image and label folders exist for every split."""
    for split in ("train", "valid", "test"):
        assert (DATASET_DIR / split / "images").is_dir()
        assert (DATASET_DIR / split / "labels").is_dir()


def test_dataset_has_images():
    """Check that each dataset split contains image files."""
    extensions = {".jpg", ".jpeg", ".png"}

    for split in ("train", "valid", "test"):
        image_dir = DATASET_DIR / split / "images"

        images = [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in extensions
        ]

        assert len(images) > 0, f"No images in {split} split."
