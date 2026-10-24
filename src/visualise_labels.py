"""
Draw the labelled defect boxes onto sample photos, to check them by eye.

Automated checks (``src.validate_annotations``) confirm that every
label is *well-formed*, but not that it is *correct*: a box could be
valid yet sit on the wrong part of the pack, or carry the wrong defect
type. Looking at a sample of labelled photos catches these mistakes,
which would otherwise teach the model the wrong thing.

## Usage

Run from the project root:

```text
$ python src/visualise_labels.py
Classes:
0 loose-meat
1 packaging-error
2 twisted-meat
3 unsealed
4 wrinkle
Saved: results/label_checks/frame_with_red_....jpg
...

Finished visualising labels.
```

Ten training photos are chosen at random (with a fixed seed, so the same
ten every time) and saved to ``results/label_checks/`` with each
labelled defect drawn as a red box and named. Open them and check that
each box sits on a real defect of the named type.
"""

from pathlib import Path
import random

from PIL import Image, ImageDraw
import yaml


# Dataset location
DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")

IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"

# Where labelled examples will be saved
OUTPUT_DIR = Path("results/label_checks")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def draw_labels(image_path, label_path, class_names):
    """
    Draw the YOLO bounding boxes from one label file onto its image.

    Lines that do not have exactly five values are skipped.

    Args:
        image_path (Path): Image file to annotate.
        label_path (Path): Matching YOLO label file. If it does not exist,
            the image is returned without boxes.
        class_names (list[str]): Class names indexed by class ID.

    Returns:
        PIL.Image.Image: RGB copy of the image with boxes and class names.

    Example:
        ```python
        image = draw_labels(
            Path("data/.../train/images/frame_35.jpg"),
            Path("data/.../train/labels/frame_35.txt"),
            ["loose-meat", "packaging-error", "twisted-meat", "unsealed", "wrinkle"],
        )
        image.show()   # opens the photo with its defects outlined in red
        ```
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    if not label_path.exists():
        return image

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(float, values[1:])

            # YOLO coordinates are normalised from 0 to 1
            x_center *= image_width
            y_center *= image_height
            box_width *= image_width
            box_height *= image_height

            # Convert centre coordinates into box corners
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 12)), class_names[class_id], fill="red")

    return image


def main():
    """
    Save 10 randomly chosen training images with their labels drawn on.

    The random seed is fixed at 42 so the same images are chosen each run.
    Annotated images are saved to ``results/label_checks``.

    Raises:
        FileNotFoundError: If the dataset has not been downloaded.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the class names from data.yaml
    with open(DATASET_DIR / "data.yaml", "r", encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["names"]

    print("Classes:")
    for i, name in enumerate(class_names):
        print(i, name)

    # Sort so the seeded sample does not depend on file-system order
    images = sorted(
        image_path
        for image_path in IMAGE_DIR.iterdir()
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    )

    random.seed(42)
    sample_images = random.sample(images, min(10, len(images)))

    for image_path in sample_images:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        image = draw_labels(image_path, label_path, class_names)

        output_path = OUTPUT_DIR / image_path.name
        image.save(output_path)
        print(f"Saved: {output_path}")

    print("\nFinished visualising labels.")


if __name__ == "__main__":
    main()
