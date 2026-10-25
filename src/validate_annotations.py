"""
Check that every defect label in the dataset is valid.

**Why this matters:** the model learns from the labels, so a single
corrupt label (a box outside the photo, or an unknown defect type) can
silently teach it the wrong thing. Labelling tools and dataset exports
do occasionally produce such errors, so all 3,123 labels are checked
before any training is done.

Each photo has a matching ``.txt`` label file with one line per defect:

```text
class_id x_center y_center width height
```

For example, this real line from the training set marks a packaging
error (class 1) near the bottom-right of the photo, about 6% of the
photo wide and 9% tall:

```text
1 0.8827 0.7624 0.0586 0.0926
```

Usage, from the project root:

```text
$ python src/validate_annotations.py

0 invalid annotation(s) found.
```

Any invalid lines are listed with their file, line number and reason.
The same checks also run automatically in the test suite
(``tests/test_annotations.py``).
"""

from pathlib import Path


DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")
"""Folder holding the downloaded dataset."""

NUM_CLASSES = 5
"""Number of defect types. Valid class IDs are 0 to 4."""


def validate_annotation(line):
    """
    Validate a single YOLO annotation line.

    Args:
        line (str): One line from a YOLO label file, e.g.
            ``"3 0.5 0.5 0.4 0.2"``.

    Returns:
        tuple: ``(class_id, x, y, width, height)`` with ``class_id`` as an
        int and the box values as floats in the range 0 to 1.

    Raises:
        ValueError: If the line does not have exactly five values, contains
            non-numeric values, uses an unknown class ID, or describes a box
            with non-positive size or that extends outside the image.

    Examples:
        A valid unsealed-pack label (class 3) in the centre of the photo:

        >>> validate_annotation("3 0.5 0.5 0.4 0.2")
        (3, 0.5, 0.5, 0.4, 0.2)

        Class 7 does not exist (there are only five defect types, 0-4):

        >>> validate_annotation("7 0.5 0.5 0.4 0.2")
        Traceback (most recent call last):
            ...
        ValueError: Class ID must be between 0 and 4.

        The centre (0.9) is inside the photo, but a box 0.4 wide would
        stretch from 0.7 to 1.1, past the right-hand edge:

        >>> validate_annotation("3 0.9 0.5 0.4 0.2")
        Traceback (most recent call last):
            ...
        ValueError: Bounding box exceeds horizontal image boundaries.
    """
    values = line.strip().split()

    if len(values) != 5:
        raise ValueError("An annotation must have exactly 5 values.")

    try:
        class_id = int(values[0])
        x, y, width, height = map(float, values[1:])
    except ValueError as exc:
        raise ValueError("Annotation contains invalid numeric values.") from exc

    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError("Class ID must be between 0 and 4.")

    if not all(0 <= value <= 1 for value in (x, y, width, height)):
        raise ValueError("Bounding-box values must be between 0 and 1.")

    if width <= 0 or height <= 0:
        raise ValueError("Bounding-box dimensions must be positive.")

    # The box edges, not just its centre, must stay inside the image.
    if not (0 <= x - width / 2 and x + width / 2 <= 1):
        raise ValueError("Bounding box exceeds horizontal image boundaries.")

    if not (0 <= y - height / 2 and y + height / 2 <= 1):
        raise ValueError("Bounding box exceeds vertical image boundaries.")

    return class_id, x, y, width, height


def find_invalid_annotations(dataset_dir=DATASET_DIR):
    """
    Check every label file in the train, valid and test splits.

    Empty label files are allowed, since they represent images with no
    defects.

    Args:
        dataset_dir (Path): Root folder of the downloaded dataset.

    Returns:
        list[str]: One message per invalid line, giving the file, line
        number and reason. The list is empty if every annotation is valid.

    Example:
        ```python
        problems = find_invalid_annotations()
        if problems:
            print("\\n".join(problems))
        ```

        A problem message looks like:

        ```text
        data/.../train/labels/frame_12.txt, line 2: Class ID must be between 0 and 4.
        ```
    """
    problems = []

    for split in ("train", "valid", "test"):
        for label_path in sorted((dataset_dir / split / "labels").glob("*.txt")):
            lines = label_path.read_text(encoding="utf-8").splitlines()

            for line_number, line in enumerate(lines, start=1):
                try:
                    validate_annotation(line)
                except ValueError as exc:
                    problems.append(f"{label_path}, line {line_number}: {exc}")

    return problems


def main():
    """Print any invalid annotations found in the dataset."""
    problems = find_invalid_annotations()

    for problem in problems:
        print(problem)

    print(f"\n{len(problems)} invalid annotation(s) found.")


if __name__ == "__main__":
    main()
