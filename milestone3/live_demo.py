"""Live demo: run any image the TAs send through our models and show the result in a window.

    python live_demo.py auto  image.jpg [more ...]    # picks the right model for each image (recommended)
    python live_demo.py watch ~/Downloads             # waits for new images in a folder and shows each one automatically
    python live_demo.py single photo.jpg              # (a) one object   -> Object ID (ConvNeXt-Tiny, Milestone 1)
    python live_demo.py multi  grid.jpg               # (b) many objects -> ID + box for every object (YOLOv8s, Milestone 2)

auto: images whose sides are multiples of 224 px and larger than one tile (448, 672, 896, 1120) are grids and go to
the detector; any other image is shown with both the single-object ID and the detector's boxes.
Close the window (or press q) to continue. Results are also printed in the terminal. --save writes them to shown/.
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.patches as patches
from PIL import Image, ImageOps

try:                                               # iPhone photos (.heic)
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

sys.stdout.reconfigure(line_buffering=True)          # print results immediately, also in watch mode
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "milestone2"))
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
RED = "#ff2d55"
_models = {}


def classifier():
    if "clf" not in _models:
        from identify_object import load_model
        _models["clf"] = load_model()
    return _models["clf"]


def detector():
    if "det" not in _models:
        from ultralytics import YOLO
        from detect_objects import load_names
        _models["det"] = (YOLO(str(HERE.parent / "milestone2" / "weights" / "grid_yolov8s_best.pt")), load_names())
    return _models["det"]


def open_rgb(p):
    with Image.open(p) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def classify(img):
    import torch
    from identify_object import CLASSES, DEVICE, NAMES, PREPROCESS
    with torch.no_grad():
        probs = torch.softmax(classifier()(PREPROCESS(img).unsqueeze(0).to(DEVICE)), 1)[0].cpu()
    best = probs.topk(3)
    return [(CLASSES[i], NAMES.get(CLASSES[i], ""), float(p)) for p, i in zip(best.values, best.indices)]


def _predict(img, conf, scale=1.0):
    model, names = detector()
    res = model.predict(img, conf=conf, imgsz=1120, agnostic_nms=True, iou=0.5, verbose=False)[0]  # one box per object
    rows = []
    for b in res.boxes:
        oid = res.names[int(b.cls)]
        x1, y1, x2, y2 = (float(v) / scale for v in b.xyxy[0])
        rows.append((oid, names.get(oid, ""), float(b.conf), (x1, y1, x2, y2)))
    return rows


def _on_canvas(img, conf):
    """Run the detector at its training scale: the photo shrunk to a 224 px grid cell on a 2x2 (448 px) canvas."""
    s = 224 / max(img.size)
    small = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
    canvas = Image.new("RGB", (448, 448), (127, 127, 127))
    canvas.paste(small, (0, 0))
    return [r for r in _predict(canvas, conf, scale=s) if r[3][0] < img.width and r[3][1] < img.height]


def detect(img, conf=0.25):
    if max(img.size) <= 448 and not is_grid(img):     # a single class photo (224 px): use the training scale directly
        return sorted(_on_canvas(img, conf), key=lambda r: -r[2])
    rows = _predict(img, conf)
    if not rows and not is_grid(img):
        # The detector was trained on grids of 224 px photos. A full-size photo shows objects far larger than
        # anything it saw, so retry with the photo shrunk to that scale (long side 448 px, on a 2x2-grid canvas).
        s = 448 / max(img.size)
        small = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))))
        canvas = Image.new("RGB", (448, 448), (127, 127, 127))
        canvas.paste(small, (0, 0))
        rows = [r for r in _predict(canvas, conf, scale=s) if r[3][0] < img.width and r[3][1] < img.height]
    cell = 112 if min(img.size) <= 1120 else min(img.size) / 10
    return sorted(rows, key=lambda r: (r[3][1] // cell, r[3][0]))


def draw_boxes(ax, rows):
    for oid, _, c, (x1, y1, x2, y2) in rows:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=RED, linewidth=2.2))
        ax.text(x1 + 2, y1 + 2, f"{oid} {c:.2f}", fontsize=8, color="white", va="top",
                bbox=dict(facecolor=RED, edgecolor="none", pad=1.5))


def print_rows(rows):
    for oid, n, c, (x1, y1, x2, y2) in rows:
        print(f"  {oid}  {n:<22s} conf {c:.2f}   box ({round(x1)}, {round(y1)}) to ({round(x2)}, {round(y2)})")


def is_grid(img):
    w, h = img.size
    return w % 224 == 0 and h % 224 == 0 and max(w, h) >= 448


def show(fig, path, save):
    import matplotlib.pyplot as plt
    if save:
        Path("shown").mkdir(exist_ok=True)
        fig.savefig(Path("shown") / f"{Path(path).stem}_result.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    else:
        try:
            fig.canvas.manager.set_window_title(Path(path).name)
        except Exception:
            pass
        plt.show()


def run_single(p, save):
    import matplotlib.pyplot as plt
    img = open_rgb(p)
    preds = classify(img)
    oid, name, conf = preds[0]
    print(f"\n{Path(p).name}  ->  {oid}  {name}  ({conf:.1%})")
    for o, n, c in preds[1:]:
        print(f"    runner-up: {o}  {n}  ({c:.1%})")
    fig, ax = plt.subplots(figsize=(6.5, 7.2))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{oid}  ·  {name}", fontsize=20, fontweight="bold", pad=14)
    runners = "    ".join(f"{o} {n} {c:.1%}" for o, n, c in preds[1:])
    fig.text(0.5, 0.04, f"confidence {conf:.1%}      runner-up: {runners}", ha="center", fontsize=10, color="#555")
    show(fig, p, save)


def run_multi(p, save):
    import matplotlib.pyplot as plt
    img = open_rgb(p)
    rows = detect(img)
    print(f"\n{Path(p).name}: {len(rows)} detection(s)")
    print_rows(rows)
    fig, ax = plt.subplots(figsize=(9, 9.6))
    ax.imshow(img)
    ax.axis("off")
    draw_boxes(ax, rows)
    ax.set_title(f"{len(rows)} detections", fontsize=20, fontweight="bold", pad=14)
    show(fig, p, save)


def run_both(p, save):
    """Not a grid: show the single-object ID and the detector's boxes side by side."""
    import matplotlib.pyplot as plt
    img = open_rgb(p)
    preds, rows = classify(img), detect(img)
    oid, name, conf = preds[0]
    print(f"\n{Path(p).name}")
    print(f"  single-object model:  {oid}  {name}  ({conf:.1%})")
    print(f"  detector: {len(rows)} detection(s)")
    print_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    for ax in axes:
        ax.imshow(img)
        ax.axis("off")
    axes[0].set_title(f"Single-object ID:  {oid} · {name}  ({conf:.0%})", fontsize=15, fontweight="bold", pad=12)
    draw_boxes(axes[1], rows)
    axes[1].set_title(f"Detector:  {len(rows)} detections", fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    show(fig, p, save)


def run_auto(p, save):
    img = open_rgb(p)
    (run_multi if is_grid(img) else run_both)(p, save)


def expand(paths):
    out = []
    for p in map(Path, paths):
        p = p.expanduser()
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in EXTS) if p.is_dir() else [p]
    return out


def watch(folder, save):
    folder = Path(folder).expanduser()
    seen = {p for p in folder.iterdir()}
    classifier(), detector()                       # load both models now so results appear instantly
    print(f"Watching {folder} for new images ... (Ctrl+C to stop)")
    while True:
        for p in sorted(folder.iterdir(), key=lambda q: q.stat().st_mtime):
            if p in seen or p.suffix.lower() not in EXTS:
                continue
            size = -1
            while size != p.stat().st_size:        # wait until the download has finished writing
                size = p.stat().st_size
                time.sleep(0.5)
            seen.add(p)
            try:
                run_auto(p, save)
            except Exception as e:                 # never crash during the demo
                print(f"Could not process {p.name}: {e}")
        time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["auto", "watch", "single", "multi"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--save", action="store_true", help="save results to shown/ instead of opening windows")
    args = ap.parse_args()
    if args.save:
        matplotlib.use("Agg")
    if args.mode == "watch":
        return watch(args.paths[0], args.save)
    run = {"auto": run_auto, "single": run_single, "multi": run_multi}[args.mode]
    for p in expand(args.paths):
        try:
            run(p, args.save)
        except Exception as e:
            print(f"Could not process {p.name}: {e}")


if __name__ == "__main__":
    main()
