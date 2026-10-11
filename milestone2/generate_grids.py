"""Build multi-object grid images (224x224 photos side by side) with YOLO labels.

Grid sizes follow the course specification:
    2x2 ->  4 objects,  448 x 448      3x3 ->  9 objects,  672 x 672      4x4 -> 16 objects, 896 x 896
    5x4 -> 20 objects, 1120 x 896      5x5 -> 25 objects, 1120 x 1120
Every cell holds a different object. Each object's box is the labelled box from label_objects.py
(box_labels.csv), shifted into its cell, so the labels cover the object itself, not the whole photo.

No leakage: grids for train / val / test use only photos from that split of split.csv, so no source photo
appears in more than one split. Only photos with a kept label are used; OBJ054 is excluded upstream.
Augmentation per cell: horizontal flip (box mirrored), brightness / contrast / saturation jitter.

Usage:
    python generate_grids.py --data "../data/raw/IE7615 Deep Learning for AI" --labels labeling/box_labels.csv \
        --out yolo_grids --n-train 1000 --n-val 150 --n-test 150
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

TILE = 224
GRIDS = [(2, 2), (3, 3), (4, 4), (5, 4), (5, 5)]        # (columns, rows)


def load_tile(data_dir, row, rng):
    """Open one photo as a 224x224 tile and return it with its box in tile pixels (after augmentation)."""
    with Image.open(data_dir / row.path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    sx, sy = TILE / img.width, TILE / img.height             # resizes the odd-sized OBJ003 photo too
    img = img.resize((TILE, TILE), Image.BICUBIC)
    x1, y1, x2, y2 = row.x1 * sx, row.y1 * sy, row.x2 * sx, row.y2 * sy
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
        x1, x2 = TILE - x2, TILE - x1
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    return img, (max(0, x1), max(0, y1), min(TILE, x2), min(TILE, y2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", default="labeling/box_labels.csv")
    ap.add_argument("--out", default="yolo_grids")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_dir, out = Path(args.data), Path(args.out)
    labels = pd.read_csv(args.labels)
    labels = labels[~labels["status"].str.startswith("excluded")].reset_index(drop=True)
    classes = sorted(labels["label"].unique())
    cls_index = {c: i for i, c in enumerate(classes)}
    rng = np.random.default_rng(args.seed)

    manifest = []
    for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        pool = labels[labels["split"] == split]
        by_class = {c: g for c, g in pool.groupby("label")}
        avail = sorted(by_class)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_obj = 0
        for i in range(n):
            cols, rows = GRIDS[i % len(GRIDS)]                 # equal share of every grid size
            k = cols * rows
            chosen = rng.choice(avail, size=k, replace=False)  # distinct objects per image
            canvas = Image.new("RGB", (cols * TILE, rows * TILE))
            W, H = canvas.size
            lines = []
            for cell, oid in enumerate(chosen):
                row = by_class[oid].iloc[int(rng.integers(len(by_class[oid])))]
                tile, (x1, y1, x2, y2) = load_tile(data_dir, row, rng)
                ox, oy = (cell % cols) * TILE, (cell // cols) * TILE
                canvas.paste(tile, (ox, oy))
                X1, Y1, X2, Y2 = ox + x1, oy + y1, ox + x2, oy + y2
                lines.append(f"{cls_index[oid]} {(X1 + X2) / 2 / W:.6f} {(Y1 + Y2) / 2 / H:.6f} "
                             f"{(X2 - X1) / W:.6f} {(Y2 - Y1) / H:.6f}")
                manifest.append([split, f"{split}_{i:05d}", f"{cols}x{rows}", oid, round(X1), round(Y1),
                                 round(X2), round(Y2), row.path, row.status])
            name = f"{split}_{i:05d}"
            canvas.save(out / "images" / split / f"{name}.jpg", quality=92)
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
            n_obj += k
        print(f"{split:5s}: {n} grid images, {n_obj} objects, {len(avail)} classes, "
              f"{pool['path'].nunique()} source photos available")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "grid", "object_id", "x1", "y1", "x2", "y2", "source_image", "label_status"])
        w.writerows(manifest)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                   f"nc: {len(classes)}\nnames:\n{names}\n")
    print(f"wrote {out / 'data.yaml'} ({len(classes)} classes)")


if __name__ == "__main__":
    main()
