"""Generate a YOLO multi-object detection dataset by compositing single-object images.

Each output image is a 640x640 canvas with 2-6 randomly selected single-object photos pasted onto
a background, plus a YOLO label file (one line per object: class x_center y_center width height,
all normalised to [0, 1]). Because the objects are placed by this script, every bounding box is
known exactly and no manual annotation is needed.

No leakage: train / val / test composites are built only from the train / val / test single-object
images of the Milestone 1 split (split.csv), so the detector is never tested on a photo it trained on.

Usage:
    python generate_multi_object.py --data "../data/raw/IE7615 Deep Learning for AI" \
        --split ../outputs/split.csv --out yolo_data --n-train 2000 --n-val 300 --n-test 300
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

CANVAS = 640


def load_split(split_csv):
    pools = {"train": [], "val": [], "test": []}
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            pools[row["split"]].append((row["path"], row["label"]))
    return pools


def class_index(label):
    """OBJ001 -> 0 ... OBJ073 -> 72 (the YOLO class id is the Object ID minus one)."""
    return int(label[3:]) - 1


def make_background(rng, bg_paths, data_dir):
    """Background-only class photos (*_bg_*) 60% of the time, otherwise a plain or textured canvas."""
    r = rng.random()
    if bg_paths and r < 0.6:
        with Image.open(data_dir / rng.choice(bg_paths)) as im:
            bg = ImageOps.exif_transpose(im).convert("RGB")
        bg = ImageOps.fit(bg, (CANVAS, CANVAS), method=Image.BICUBIC)
        if rng.random() < 0.5:
            bg = ImageOps.mirror(bg)
        return bg
    if r < 0.8:
        colour = tuple(int(c) for c in rng.integers(30, 235, 3))
        return Image.new("RGB", (CANVAS, CANVAS), colour)
    noise = rng.integers(0, 255, (CANVAS // 8, CANVAS // 8, 3), dtype=np.uint8)
    return Image.fromarray(noise).resize((CANVAS, CANVAS), Image.BICUBIC).filter(ImageFilter.GaussianBlur(6))


def augment_tile(img, rng, size):
    """Scale, flip, colour-jitter and slightly rotate one object photo. Returns RGB image + alpha mask."""
    img = img.resize((size, size), Image.BICUBIC)
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    mask = Image.new("L", img.size, 255)
    if rng.random() < 0.4:
        angle = rng.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        mask = mask.rotate(angle, resample=Image.BICUBIC, expand=True)
    return img, mask


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def place_random(rng, w, h, boxes, max_iou=0.1, tries=60):
    for _ in range(tries):
        x, y = int(rng.integers(0, CANVAS - w + 1)), int(rng.integers(0, CANVAS - h + 1))
        box = (x, y, x + w, y + h)
        if all(iou(box, b) <= max_iou for b in boxes):
            return box
    return None


def compose(rng, pool, bg_paths, data_dir):
    canvas = make_background(rng, bg_paths, data_dir)
    k = int(rng.choice([2, 3, 4, 5, 6], p=[0.15, 0.25, 0.25, 0.2, 0.15]))
    picks = [pool[i] for i in rng.choice(len(pool), size=k * 3, replace=False)]
    chosen, seen = [], set()
    for path, label in picks:                   # distinct objects within one image
        if label not in seen:
            chosen.append((path, label))
            seen.add(label)
        if len(chosen) == k:
            break

    layout = "grid" if rng.random() < 0.3 else "random"
    if layout == "grid":
        g = 2 if k <= 4 else 3
        cells = [(r, c) for r in range(g) for c in range(g)]
        rng.shuffle(cells)
        cell = CANVAS // g

    boxes, labels, sources = [], [], []
    for i, (path, label) in enumerate(chosen):
        with Image.open(data_dir / path) as im:
            src = ImageOps.exif_transpose(im).convert("RGB")
        if layout == "grid":
            size = int(cell * rng.uniform(0.62, 0.9))
        else:
            size = int(rng.uniform(120, 270))
        tile, mask = augment_tile(src, rng, size)
        w, h = tile.size
        if layout == "grid":
            r, c = cells[i]
            x = c * cell + int(rng.integers(0, max(1, cell - w + 1)))
            y = r * cell + int(rng.integers(0, max(1, cell - h + 1)))
            box = (x, y, min(CANVAS, x + w), min(CANVAS, y + h))
        else:
            box = place_random(rng, w, h, boxes)
            if box is None:
                continue
        canvas.paste(tile, box[:2], mask)
        boxes.append(box)
        labels.append(class_index(label))
        sources.append(path)

    if rng.random() < 0.3:                      # whole-image photometric jitter
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.8, 1.2))
    return canvas, boxes, labels, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with images_OBJ001 ... images_OBJ073")
    ap.add_argument("--split", required=True, help="Milestone 1 split.csv")
    ap.add_argument("--out", default="yolo_data")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    pools = load_split(args.split)
    bg_paths = sorted(str(p.relative_to(data_dir)) for p in data_dir.glob("images_OBJ*/*")
                      if "_bg_" in p.name.lower() and p.suffix.lower() in (".jpg", ".jpeg"))
    classes = sorted({lab for pool in pools.values() for _, lab in pool})
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    manifest = []
    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_obj = 0
        for i in range(n):
            img, boxes, labels, sources = compose(rng, pools[split], bg_paths, data_dir)
            name = f"{split}_{i:05d}"
            img.save(out / "images" / split / f"{name}.jpg", quality=90)
            with open(out / "labels" / split / f"{name}.txt", "w") as f:
                for (x1, y1, x2, y2), c in zip(boxes, labels):
                    f.write(f"{c} {(x1 + x2) / 2 / CANVAS:.6f} {(y1 + y2) / 2 / CANVAS:.6f} "
                            f"{(x2 - x1) / CANVAS:.6f} {(y2 - y1) / CANVAS:.6f}\n")
            for (x1, y1, x2, y2), c, s in zip(boxes, labels, sources):
                manifest.append([split, name, f"OBJ{c + 1:03d}", x1, y1, x2, y2, s])
            n_obj += len(boxes)
        print(f"{split:5s}: {n} images, {n_obj} objects ({n_obj / n:.1f} per image)")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "object_id", "x1", "y1", "x2", "y2", "source_image"])
        w.writerows(manifest)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
                                   f"test: images/test\nnc: {len(classes)}\nnames:\n{names}\n")
    print(f"wrote {out / 'data.yaml'} ({len(classes)} classes) and manifest.csv")


if __name__ == "__main__":
    main()
