from ultralytics import YOLO


def main():
    model = YOLO(
        "runs/detect/results/training/yolo11s_baseline/weights/last.pt"
    )

    model.train(resume=True)


if __name__ == "__main__":
    main()