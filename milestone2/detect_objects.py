"""Detect, identify and locate every object in one or more images with the trained YOLOv8 model.

Usage:
    python detect_objects.py path/to/image.jpg [more images or a folder] [--weights runs/yolov8n/weights/best.pt]

For each image it prints one line per detected object (Object ID, object name, confidence and the
bounding box in pixels) and saves an annotated copy to detections/.
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def load_names():
    candidates = [HERE.parent / "data" / "object_ids.csv", HERE.parent / "object_ids.csv", HERE / "object_ids.csv"]
    f = next((c for c in candidates if c.exists()), candidates[0])
    return {r["OBJECT_ID"]: r["OBJECT_NAME"] for r in csv.DictReader(open(f))} if f.exists() else {}


def detect(model, path, names, conf, out_dir):
    res = model.predict(str(path), conf=conf, imgsz=1120, verbose=False)[0]
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    found = []
    for b in sorted(res.boxes, key=lambda b: (float(b.xyxy[0][1]), float(b.xyxy[0][0]))):
        obj_id = res.names[int(b.cls)]                       # e.g. "OBJ064"
        x1, y1, x2, y2 = (round(float(v)) for v in b.xyxy[0])
        score = float(b.conf)
        found.append((obj_id, names.get(obj_id, ""), score, (x1, y1, x2, y2)))
        label = f"{obj_id} {names.get(obj_id, '')} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(230, 30, 60), width=3)
        tw = draw.textlength(label)
        draw.rectangle([x1, max(0, y1 - 16), x1 + tw + 8, y1], fill=(230, 30, 60))
        draw.text((x1 + 4, max(0, y1 - 15)), label, fill="white")
    out_dir.mkdir(exist_ok=True)
    img.save(out_dir / f"{Path(path).stem}_detected.jpg", quality=90)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights", default=str(HERE / "weights" / "grid_yolov8s_best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="detections")
    args = ap.parse_args()

    model, names = YOLO(args.weights), load_names()
    paths = []
    for p in map(Path, args.images):
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")) if p.is_dir() else [p]
    for p in paths:
        found = detect(model, p, names, args.conf, Path(args.out))
        print(f"\n{p.name}: {len(found)} object(s)")
        for obj_id, name, score, (x1, y1, x2, y2) in found:
            print(f"  {obj_id}  {name:<22s} conf {score:.2f}   box x1={x1} y1={y1} x2={x2} y2={y2}")


if __name__ == "__main__":
    main()
