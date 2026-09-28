"""Entrena YOLOv8n para deteccion de huevos buenos/malos.

Uso:
    python scripts/train.py --data dataset/data.yaml --epochs 80
"""
import argparse
import multiprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"dataset/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--imgsz", default=224, type=int)
    parser.add_argument("--batch", default=32, type=int)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=20,
        workers=4,
        project="runs",
        name="huevos",
        exist_ok=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()