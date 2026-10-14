import os
from pathlib import Path
from roboflow import Roboflow

# Read API key from an environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")

if api_key is None:
    raise RuntimeError("ROBOFLOW_API_KEY has not been set.")

# Connect to Roboflow
rf = Roboflow(api_key=api_key)

# Select the pork rasher dataset
project = rf.workspace("hello-2aqe0").project(
    "pork-rasher-error-packaging"
)

# Use dataset Version 4
version = project.version(4)

# Keep downloaded dataset inside the data folder
Path("data").mkdir(exist_ok=True)
os.chdir("data")

# Download in YOLO11 format
dataset = version.download("yolov11")

print("Dataset downloaded successfully.")
print(dataset.location)