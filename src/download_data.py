"""
Download the pork-rasher packaging defect dataset from Roboflow.

Downloads Version 4 of the Pork-Rasher-Error(Packaging) dataset
in YOLO11 object-detection format.

The Roboflow API key is read from the ROBOFLOW_API_KEY
environment variable rather than being stored in the code.

Run from the project root:
    python src/download_data.py
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
        RuntimeError: If ROBOFLOW_API_KEY has not been set.
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
