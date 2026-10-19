"""Live demo for the professor: shows each image in a window with the model's answer on it.

    python live_demo.py single photo.jpg [more ...]   # (a) one object  -> Object ID (ConvNeXt-Tiny, Milestone 1)
    python live_demo.py multi  grid.jpg  [more ...]   # (b) many objects -> ID + box for every object (YOLOv8s, Milestone 2)

Close the window (or press q) to move to the next image. Results are also printed in the terminal.
Add --save to write the result images to shown/ instead of opening windows.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "milestone2"))


def expand(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")) if p.is_dir() else [p]
    return out


def finish(fig, path, save):
    import matplotlib.pyplot as plt
    if save:
        Path("shown").mkdir(exist_ok=True)
        fig.savefig(Path("shown") / f"{path.stem}_result.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def single(paths, save):
    import matplotlib.pyplot as plt
    from identify_object import identify, load_model
    model = load_model()
    for p in paths:
        preds = identify(model, p, top=3)
        oid, name, conf = preds[0]
        print(f"\n{p.name}:  {oid}  {name}  ({conf:.1%})")
        for o, n, c in preds[1:]:
            print(f"    runner-up: {o}  {n}  ({c:.1%})")
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        fig, ax = plt.subplots(figsize=(6.5, 7.2))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{oid}  ·  {name}", fontsize=20, fontweight="bold", pad=14)
        runners = "    ".join(f"{o} {n} {c:.1%}" for o, n, c in preds[1:])
        fig.text(0.5, 0.04, f"confidence {conf:.1%}      runner-up: {runners}", ha="center", fontsize=10, color="#555")
        fig.canvas.manager.set_window_title(f"Single object: {p.name}") if not save else None
        finish(fig, p, save)


def multi(paths, save):
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    from detect_objects import load_names
    model, names = YOLO(str(HERE.parent / "milestone2" / "weights" / "grid_yolov8s_best.pt")), load_names()
    for p in paths:
        res = model.predict(str(p), conf=0.25, imgsz=1120, verbose=False)[0]
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        fig, ax = plt.subplots(figsize=(9, 9.6))
        ax.imshow(img)
        ax.axis("off")
        rows = []
        for b in sorted(res.boxes, key=lambda b: (float(b.xyxy[0][1]) // 112, float(b.xyxy[0][0]))):
            oid = res.names[int(b.cls)]
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#ff2d55", linewidth=2.2))
            ax.text(x1 + 2, y1 + 2, f"{oid} {float(b.conf):.2f}", fontsize=8, color="white", va="top",
                    bbox=dict(facecolor="#ff2d55", edgecolor="none", pad=1.5))
            rows.append((oid, names.get(oid, ""), float(b.conf), (round(x1), round(y1), round(x2), round(y2))))
        ax.set_title(f"{len(rows)} detections", fontsize=20, fontweight="bold", pad=14)
        print(f"\n{p.name}: {len(rows)} object(s)")
        for oid, n, c, (x1, y1, x2, y2) in rows:
            print(f"  {oid}  {n:<22s} conf {c:.2f}   box ({x1}, {y1}) to ({x2}, {y2})")
        fig.canvas.manager.set_window_title(f"Multiple objects: {p.name}") if not save else None
        finish(fig, p, save)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--save", action="store_true", help="save results to shown/ instead of opening windows")
    args = ap.parse_args()
    if args.save:
        matplotlib.use("Agg")
    (single if args.mode == "single" else multi)(expand(args.images), args.save)


if __name__ == "__main__":
    main()
