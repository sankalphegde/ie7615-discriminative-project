"""Fine-tune a COCO-pretrained YOLOv8 model on the generated multi-object dataset (transfer learning).

Usage:
    python train_yolo.py --model yolov8n.pt --name yolov8n --epochs 60
    python train_yolo.py --model yolov8s.pt --name yolov8s --epochs 60

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
    ap.add_argument("--name", default="yolov8n")
    ap.add_argument("--data", default="yolo_data/data.yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    ("0" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    model = YOLO(args.model)                     # pretrained on COCO -> transfer learning
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, patience=15, seed=42, deterministic=False,
                project=str(RUNS), name=args.name, exist_ok=True, plots=True, workers=4,
                fliplr=0.5, mosaic=1.0, close_mosaic=10)  # YOLO's own augmentation on top of ours

    evaluate(RUNS / args.name, args.data, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
