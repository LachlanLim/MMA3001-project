
"""Validate the format and values of YOLO bounding-box annotations."""

from pathlib import Path
import pytest


DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")


def validate_annotation(line):
    """Validate a single YOLO annotation line.

    Returns:
        Tuple containing class ID and four bounding-box values.

    Raises:
        ValueError: If the annotation format or values are invalid.
    """
    values = line.strip().split()

    if len(values) != 5:
        raise ValueError("An annotation must have exactly 5 values.")

    try:
        class_id = int(values[0])
        x, y, width, height = map(float, values[1:])
    except ValueError as exc:
        raise ValueError("Annotation contains invalid numeric values.") from exc

    if not 0 <= class_id < 5:
        raise ValueError("Class ID must be between 0 and 4.")

    if not all(0 <= value <= 1 for value in (x, y, width, height)):
        raise ValueError("Bounding-box values must be between 0 and 1.")

    if width <= 0 or height <= 0:
        raise ValueError("Bounding-box dimensions must be positive.")

    if not (0 <= x - width / 2 and x + width / 2 <= 1):
        raise ValueError("Bounding box exceeds horizontal image boundaries.")

    if not (0 <= y - height / 2 and y + height / 2 <= 1):
        raise ValueError("Bounding box exceeds vertical image boundaries.")

    return class_id, x, y, width, height


def test_valid_annotation():
    """Accept correctly formatted bounding-box annotations."""
    result = validate_annotation("3 0.5 0.5 0.4 0.2")
    assert result == (3, 0.5, 0.5, 0.4, 0.2)


@pytest.mark.parametrize(
    "annotation",
    [
        "",
        "3 0.5 0.5 0.4",
        "3 0.5 0.5 0.4 0.2 0.1",
        "unknown 0.5 0.5 0.4 0.2",
        "5 0.5 0.5 0.4 0.2",
        "-1 0.5 0.5 0.4 0.2",
        "3 1.2 0.5 0.4 0.2",
        "3 0.5 0.5 0 0.2",
        "3 0.9 0.5 0.4 0.2",
        "3 0.5 0.05 0.4 0.2",
    ],
)
def test_invalid_annotations(annotation):
    """Reject malformed annotations and invalid bounding boxes."""
    with pytest.raises(ValueError):
        validate_annotation(annotation)


def test_real_dataset_annotations():
    """Check that all available dataset annotation files are valid."""
    label_files = []

    for split in ("train", "valid", "test"):
        label_dir = DATASET_DIR / split / "labels"
        label_files.extend(label_dir.glob("*.txt"))

    assert label_files, "No annotation files were found."

    for label_path in label_files:
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            try:
                validate_annotation(line)
            except ValueError as exc:
                pytest.fail(
                    f"{label_path}, line {line_number}: {exc}"
                )
