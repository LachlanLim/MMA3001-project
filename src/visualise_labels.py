from pathlib import Path
from PIL import Image, ImageDraw
import yaml
import random


# Dataset location
DATASET_DIR = Path("data/Pork-Rasher-Error(Packaging)-4")

IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"

# Where labelled examples will be saved
OUTPUT_DIR = Path("results/label_checks")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Read the class names from data.yaml
# --------------------------------------------------

with open(DATASET_DIR / "data.yaml", "r") as f:
    config = yaml.safe_load(f)

class_names = config["names"]

print("Classes:")
for i, name in enumerate(class_names):
    print(i, name)


# --------------------------------------------------
# Choose 10 random training images
# --------------------------------------------------

image_extensions = {".jpg", ".jpeg", ".png"}

images = [
    image_path
    for image_path in IMAGE_DIR.iterdir()
    if image_path.suffix.lower() in image_extensions
]

random.seed(42)
sample_images = random.sample(images, min(10, len(images)))


# --------------------------------------------------
# Draw the ground-truth bounding boxes
# --------------------------------------------------

for image_path in sample_images:

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    # Find the matching YOLO label file
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if label_path.exists():

        with open(label_path, "r") as f:

            for line in f:

                values = line.strip().split()

                if len(values) != 5:
                    continue

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                box_width = float(values[3])
                box_height = float(values[4])

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

                class_name = class_names[class_id]

                # Draw box and class name
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=3
                )

                draw.text(
                    (x1, max(0, y1 - 12)),
                    class_name,
                    fill="red"
                )

    # Save annotated image
    output_path = OUTPUT_DIR / image_path.name
    image.save(output_path)

    print(f"Saved: {output_path}")


print("\nFinished visualising labels.")