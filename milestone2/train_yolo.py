"""Fine-tune a COCO-pretrained YOLOv8 model on the generated multi-object dataset (transfer learning).

Usage:
    python train_yolo.py --model yolov8n.pt --name grid_yolov8n --epochs 50
    python train_yolo.py --model yolov8s.pt --name grid_yolov8s --epochs 40

imgsz=1120 matches the largest grid (5x5 = 1120 x 1120), as recommended, so small objects are not shrunk.

Results (weights, curves, confusion matrix, metrics) are written to runs/<name>/.
On Colab, pass --device 0 to use the GPU; on Apple silicon the default is "mps".
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from evaluate_yolo import evaluate

RUNS = Path(__file__).resolve().parent / "runs"   # absolute, so Ultralytics doesn't nest it under runs/detect/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", default="grid_yolov8n")
    ap.add_argument("--data", default="yolo_grids/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1120)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("0" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    model = YOLO(args.model)                     # pretrained on COCO -> transfer learning
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, patience=15, seed=42, deterministic=False,
                project=str(RUNS), name=args.name, exist_ok=True, plots=True, workers=4,
                fliplr=0.5, mosaic=0.5, close_mosaic=10, save_period=5)  # YOLO augmentation on top of ours; checkpoint every 5 epochs

    evaluate(RUNS / args.name, args.data, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
