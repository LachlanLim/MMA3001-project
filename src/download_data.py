"""
Download the pork-pack photos and their defect labels from Roboflow.

The dataset is *Pork-Rasher-Error (Packaging)*, version 4, hosted on
[Roboflow Universe](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4)
under the CC BY 4.0 licence (free to use with credit). It holds 3,549
photos, each with a label file marking where the defects are.

The photos are not stored in this repository, because they belong to
the dataset's authors and are about 240 MB. Anyone reproducing the
project downloads them with this script instead.

## Usage

1. Create a free account at [roboflow.com](https://roboflow.com) and
   copy your API key from *Settings → API Keys*.
2. Store the key in an environment variable, so it is never written
   into the code or uploaded to GitHub:

   ```text
   set ROBOFLOW_API_KEY=your_key_here         (Windows)
   export ROBOFLOW_API_KEY=your_key_here      (macOS/Linux)
   ```

3. Run from the project root:

   ```text
   $ python src/download_data.py
   Dataset downloaded successfully.
   .../data/Pork-Rasher-Error(Packaging)-4
   ```

The download creates:

```text
data/Pork-Rasher-Error(Packaging)-4/
├── data.yaml          defect type names and folder locations
├── train/images/      3349 photos the model learns from
├── train/labels/      one .txt label file per photo
├── valid/             120 photos for choosing settings
└── test/              80 photos for the final check
```
"""

import os
from pathlib import Path
from roboflow import Roboflow


def main():
    """
    Download Version 4 of the pork-packaging dataset.

    Reads the API key from the ROBOFLOW_API_KEY environment variable.
    The images, YOLO annotations and data.yaml are saved to
    ``data/Pork-Rasher-Error(Packaging)-4``.

    Raises:
        RuntimeError: If ROBOFLOW_API_KEY has not been set. This is
            checked first, so a missing key fails immediately with a
            clear message instead of an unclear network error.
    """

    # Read API key securely from environment variables.
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY has not been set.")

    # Connect to Roboflow.
    rf = Roboflow(api_key=api_key)

    # Select the pork-rasher dataset.
    project = rf.workspace("hello-2aqe0").project(
        "pork-rasher-error-packaging"
    )

    # Use dataset Version 4.
    version = project.version(4)

    # Create the local dataset directory.
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Roboflow downloads into the current directory, so move into data/.
    os.chdir(data_dir)

    # Download the dataset in YOLO11 format.
    dataset = version.download("yolov11")

    print("Dataset downloaded successfully.")
    print(dataset.location)


if __name__ == "__main__":
    main()
