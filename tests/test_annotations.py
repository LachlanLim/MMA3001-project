"""Validate the format and values of YOLO bounding-box annotations."""

import pytest

from validate_annotations import (
    DATASET_DIR,
    find_invalid_annotations,
    validate_annotation,
)


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


@pytest.mark.skipif(
    not DATASET_DIR.is_dir(),
    reason="Dataset not downloaded (run src/download_data.py).",
)
def test_real_dataset_annotations():
    """Check that all available dataset annotation files are valid."""
    assert list(DATASET_DIR.glob("*/labels/*.txt")), "No annotation files were found."

    problems = find_invalid_annotations()
    assert not problems, "\n".join(problems)
