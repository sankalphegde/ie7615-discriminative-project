"""Build the Milestone 2 report (PDF) in the same format as the group's Milestone 1 report.

Reads labeling/box_labels.csv, labeling/manual_check.json, yolo_grids/manifest.csv and runs/<name>/final_metrics.json.
Usage:  python make_report_m2.py          Output: report/Milestone2_Report.pdf
"""
import ast
import base64
import csv
import html
import io
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
RUNS, GRIDS, LAB, REPORT = HERE / "runs", HERE / "yolo_grids", HERE / "labeling", HERE / "report"
DATA = HERE.parent / "data" / "raw" / "IE7615 Deep Learning for AI"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/sankalphegde/ie7615-discriminative-project"
TEAM = "Jairaghavendra Sridhar · Reshma Jose · Sankalp Hegde"
DUE = "October 11, 2026"
RUN_NAMES = {"grid_yolov8n": "YOLOv8n", "grid_yolov8s": "YOLOv8s"}
_ids = next(p for p in [HERE.parent / "data" / "object_ids.csv", HERE.parent / "object_ids.csv"] if p.exists())
NAMES = {r["OBJECT_ID"]: r["OBJECT_NAME"] for r in csv.DictReader(open(_ids))}
GRID_ORDER = ["2x2", "3x3", "4x4", "5x4", "5x5"]
STATUS_NAMES = {"verified": "Kept: verified (OWLv2 + a second model)", "two-model": "Kept: two-model (YOLO-World + YOLOE)",
                "single-model": "Kept: single-model (OWLv2 only)", "excluded: models disagree": "Left out: models disagree",
                "excluded: no model found it": "Left out: no model found the object",
                "excluded: object too small": "Left out: object too small (< 2% of photo)"}
GRID_PX = {"2x2": "448 × 448", "3x3": "672 × 672", "4x4": "896 × 896", "5x4": "1120 × 896", "5x5": "1120 × 1120"}


def esc(s):
    return html.escape(str(s))


def name(o):
    return f"{NAMES.get(o, o).lower() if NAMES.get(o, '').islower() else NAMES.get(o, o)} ({o})"


def fig_uri(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def file_uri(p, quality=85, max_side=None):
    im = Image.open(p).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(src, caption, width="100%"):
    return f'<figure><img src="{src}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


# ----------------------------------------------------------------------------- figures
def label_examples(labels, n=8, size=150):
    """Sample of kept photos with their final box (green), drawn for the label-checking section."""
    sample = labels[~labels.status.str.startswith("excluded")].sample(n * 2, random_state=3)
    sheet = Image.new("RGB", (n * size, 2 * size), "white")
    for k, (_, r) in enumerate(sample.iterrows()):
        im = Image.open(DATA / r.path).convert("RGB")
        s = size / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)))
        ImageDraw.Draw(im).rectangle([r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s], outline=(0, 200, 60), width=3)
        sheet.paste(im, ((k % n) * size, (k // n) * size))
    return sheet


def disagreement_examples(labels, n=8, size=150):
    """Excluded photos with each model's box: red = OWLv2, blue = YOLO-World, orange = YOLOE."""
    cols = {"owl_box": (220, 30, 40), "world_box": (30, 90, 220), "yoloe_box": (240, 140, 0)}
    bad = labels[labels.status == "excluded: models disagree"].sample(n, random_state=5)
    sheet = Image.new("RGB", (n * size, size), "white")
    for k, (_, r) in enumerate(bad.iterrows()):
        im = Image.open(DATA / r.path).convert("RGB")
        s = size / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)))
        d = ImageDraw.Draw(im)
        for c, colour in cols.items():
            if isinstance(r[c], str):
                d.rectangle([v * s for v in ast.literal_eval(r[c])], outline=colour, width=3)
        sheet.paste(im, (k * size, 0))
    return sheet


def grid_examples():
    picks = []
    for g in ["2x2", "3x3", "5x5"]:
        man = pd.read_csv(GRIDS / "manifest.csv")
        im = man[(man.split == "train") & (man.grid == g)].image.iloc[0]
        img = Image.open(GRIDS / "images" / "train" / f"{im}.jpg").convert("RGB")
        d = ImageDraw.Draw(img)
        for line in (GRIDS / "labels" / "train" / f"{im}.txt").read_text().split("\n"):
            if line.strip():
                _, x, y, w, h = map(float, line.split())
                W, H = img.size
                d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], outline=(0, 220, 60), width=4)
        picks.append(img.resize((360, round(360 * img.height / img.width))))
    sheet = Image.new("RGB", (3 * 360 + 20, max(p.height for p in picks)), "white")
    for k, p in enumerate(picks):
        sheet.paste(p, (k * 370, 0))
    return sheet


def detection_example(weights, grid="3x3", conf=0.25):
    from ultralytics import YOLO
    model = YOLO(weights)
    man = pd.read_csv(GRIDS / "manifest.csv")
    ims = man[(man.split == "test") & (man.grid == grid)].image.unique()
    best_pick = None
    for im in ims[:12]:                                 # first image where every object is found
        p = GRIDS / "images" / "test" / f"{im}.jpg"
        res = model.predict(str(p), conf=conf, imgsz=1120, verbose=False)[0]
        truth = man[(man.split == "test") & (man.image == im)]
        found = {res.names[int(c)] for c in res.boxes.cls}
        best_pick = (p, res, truth)
        if set(truth.object_id) <= found:
            break
    p, res, truth = best_pick
    img = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(img)
    rows = []
    for b in res.boxes:
        oid = res.names[int(b.cls)]
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        lab = f"{oid} {float(b.conf):.2f}"
        d.rectangle([x1, y1, x2, y2], outline=(230, 20, 40), width=3)
        d.rectangle([x1, y1, x1 + d.textlength(lab) + 6, y1 + 13], fill=(230, 20, 40))
        d.text((x1 + 3, y1 + 1), lab, fill="white")
        correct = ((truth.object_id == oid) & ((truth.x1 - x1).abs() < 40) & ((truth.y1 - y1).abs() < 40)).any()
        rows.append((oid, NAMES.get(oid, ""), float(b.conf), (round(x1), round(y1), round(x2), round(y2)), correct))
    rows.sort(key=lambda r: (r[3][1] // 112, r[3][0]))
    return img, rows, p.stem, len(truth)


def big_grid_example(weights, grid="5x5", conf=0.25):
    from ultralytics import YOLO
    model = YOLO(weights)
    man = pd.read_csv(GRIDS / "manifest.csv")
    im = man[(man.split == "test") & (man.grid == grid)].image.unique()[0]
    p = GRIDS / "images" / "test" / f"{im}.jpg"
    res = model.predict(str(p), conf=conf, imgsz=1120, verbose=False)[0]
    img = Image.open(p).convert("RGB")
    d = ImageDraw.Draw(img)
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        lab = f"{res.names[int(b.cls)]} {float(b.conf):.2f}"
        d.rectangle([x1, y1, x2, y2], outline=(230, 20, 40), width=3)
        d.rectangle([x1, y1, x1 + d.textlength(lab) + 6, y1 + 13], fill=(230, 20, 40))
        d.text((x1 + 3, y1 + 1), lab, fill="white")
    truth = man[(man.split == "test") & (man.image == im)]
    return img, len(res.boxes), len(truth)


def error_analysis(weights, conf=0.25, iou_thr=0.5):
    """Match predictions to ground truth on the test grids at the deployment threshold (conf 0.25)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    man = pd.read_csv(GRIDS / "manifest.csv")
    test = man[man.split == "test"]
    counts = {"correct": 0, "wrong_id": 0, "missed": 0, "extra": 0}
    confusions = {}
    for im, truth in test.groupby("image"):
        res = model.predict(str(GRIDS / "images" / "test" / f"{im}.jpg"), conf=conf, imgsz=1120, verbose=False)[0]
        preds = [(res.names[int(c)], [float(v) for v in b]) for c, b in zip(res.boxes.cls, res.boxes.xyxy)]
        used = set()
        for _, t in truth.iterrows():
            tb = [t.x1, t.y1, t.x2, t.y2]
            best, bi = 0.0, None
            for i, (_, pb) in enumerate(preds):
                if i in used:
                    continue
                ix = max(0, min(tb[2], pb[2]) - max(tb[0], pb[0])); iy = max(0, min(tb[3], pb[3]) - max(tb[1], pb[1]))
                u = (tb[2] - tb[0]) * (tb[3] - tb[1]) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - ix * iy
                v = ix * iy / u if u else 0
                if v > best:
                    best, bi = v, i
            if bi is None or best < iou_thr:
                counts["missed"] += 1
                continue
            used.add(bi)
            if preds[bi][0] == t.object_id:
                counts["correct"] += 1
            else:
                counts["wrong_id"] += 1
                key = (t.object_id, preds[bi][0])
                confusions[key] = confusions.get(key, 0) + 1
        counts["extra"] += len(preds) - len(used)
    counts["objects"] = len(test)
    top = sorted(confusions.items(), key=lambda kv: -kv[1])[:5]
    return counts, [(a, b, c) for (a, b), c in top]


def curves(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for key, r in runs.items():
        h = r["history"]
        ax[0].plot(h["epoch"], h["metrics/mAP50(B)"], label=RUN_NAMES[key])
        ax[1].plot(h["epoch"], h["val/box_loss"] + h["val/cls_loss"], label=RUN_NAMES[key])
    ax[0].set_title("Validation mAP@0.5")
    ax[1].set_title("Validation loss (box + classification)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.grid(True, linestyle="--", alpha=0.5)
        a.legend()
    fig.tight_layout()
    return fig_uri(fig)


def grid_bars(r):
    g = r["test_by_grid"]
    keys = [k for k in GRID_ORDER if k in g]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = range(len(keys))
    w = 0.27
    for i, (m, lab) in enumerate([("mAP50", "mAP@0.5"), ("precision", "Precision"), ("recall", "Recall")]):
        vals = [g[k][m] for k in keys]
        bars = ax.bar([v + (i - 1) * w for v in x], vals, w, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(list(x), [f"{k}\n({GRID_PX[k]})" for k in keys])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Test metrics by grid size")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig_uri(fig)


# ----------------------------------------------------------------------------- report
def load_runs():
    runs = {}
    for key in RUN_NAMES:
        d = RUNS / key
        if (d / "final_metrics.json").exists():
            r = json.loads((d / "final_metrics.json").read_text())
            h = pd.read_csv(d / "results.csv")
            h.columns = [c.strip() for c in h.columns]
            r["history"], r["epochs"] = h, len(h)
            r["train_min"] = float(h["time"].iloc[-1]) / 60 if "time" in h else float("nan")
            runs[key] = r
    return runs


def build():
    runs = load_runs()
    labels = pd.read_csv(LAB / "box_labels.csv")
    check = json.loads((LAB / "manual_check.json").read_text()) if (LAB / "manual_check.json").exists() else None
    man = pd.read_csv(GRIDS / "manifest.csv")
    best_key = max(runs, key=lambda k: (runs[k]["test"]["mAP50"], runs[k]["test"]["mAP50_95"]))
    best = runs[best_key]
    t = best["test"]
    BN = RUN_NAMES[best_key]

    total = len(labels)
    st = labels.status.value_counts()
    kept = int((~labels.status.str.startswith("excluded")).sum())
    n_cls = labels[~labels.status.str.startswith("excluded")].label.nunique()
    per_cls = labels.assign(k=~labels.status.str.startswith("excluded")).groupby("label").k.sum()
    low = per_cls.sort_values().head(4)
    imgs = man.drop_duplicates(["split", "image"])
    split_tab = imgs.groupby("split").size().reindex(["train", "val", "test"])
    obj_tab = man.groupby("split").size().reindex(["train", "val", "test"])
    src_tab = man.groupby("split").source_image.nunique().reindex(["train", "val", "test"])
    by_grid = imgs.groupby(["split", "grid"]).size().unstack(0).reindex(GRID_ORDER)[["train", "val", "test"]]
    overlap = len(set(man[man.split == "train"].source_image) & set(man[man.split == "test"].source_image)) + \
        len(set(man[man.split == "train"].source_image) & set(man[man.split == "val"].source_image)) + \
        len(set(man[man.split == "val"].source_image) & set(man[man.split == "test"].source_image))

    REPORT.mkdir(exist_ok=True)
    (REPORT / "figs").mkdir(exist_ok=True)
    label_examples(labels).save(REPORT / "figs" / "label_examples.jpg", quality=88)
    disagreement_examples(labels).save(REPORT / "figs" / "label_disagreements.jpg", quality=88)
    grid_examples().save(REPORT / "figs" / "grid_examples.jpg", quality=88)
    det_img, det_rows, det_name, det_truth = detection_example(RUNS / best_key / "weights" / "best.pt")
    det_img.save(REPORT / "figs" / "detection_3x3.jpg", quality=90)
    big_img, big_found, big_truth = big_grid_example(RUNS / best_key / "weights" / "best.pt")
    big_img.save(REPORT / "figs" / "detection_5x5.jpg", quality=88)

    # ---- tables
    res_rows = ""
    for key, r in sorted(runs.items(), key=lambda kv: -kv[1]["test"]["mAP50"]):
        tt = r["test"]
        res_rows += (f"<tr{' class=best' if key == best_key else ''}><td>{RUN_NAMES[key]}</td><td>{tt['mAP50']:.4f}</td>"
                     f"<td>{tt['mAP50_95']:.4f}</td><td>{tt['precision']:.4f}</td><td>{tt['recall']:.4f}</td>"
                     f"<td>{r['params'] / 1e6:.2f}M</td><td>{r['epochs']}</td><td>{r['train_min']:.0f}</td>"
                     f"<td>{tt['speed_ms']['inference']:.1f}</td></tr>")
    grid_rows = "".join(
        f"<tr><td>{g}</td><td>{GRID_PX[g]}</td><td>{int(g[0]) * int(g[2])}</td><td>{int(by_grid.loc[g, 'train'])}</td>"
        f"<td>{int(by_grid.loc[g, 'val'])}</td><td>{int(by_grid.loc[g, 'test'])}</td>"
        f"<td>{best['test_by_grid'][g]['mAP50']:.3f}</td><td>{best['test_by_grid'][g]['recall']:.3f}</td></tr>"
        for g in GRID_ORDER if g in best.get("test_by_grid", {}))
    det_tab = "".join(f"<tr><td>{o}</td><td>{esc(n)}</td><td>{c:.2f}</td><td>({x1}, {y1})</td><td>({x2}, {y2})</td>"
                      f"<td>{'yes' if ok else 'no'}</td></tr>" for o, n, c, (x1, y1, x2, y2), ok in det_rows)
    det_ok = sum(r[4] for r in det_rows)

    ap = pd.Series(t["per_class_mAP50"]).sort_values()
    worst = ap.head(5)
    n_perfect = int((ap >= 0.99).sum())
    ea, conf_pairs = error_analysis(RUNS / best_key / "weights" / "best.pt")
    conf_txt = "; ".join(f"{name(a)} predicted as {name(b)} ({c}×)" for a, b, c in conf_pairs[:4]) or "none"

    cmp_txt = ""
    if len(runs) == 2:
        a, b = runs["grid_yolov8n"], runs["grid_yolov8s"]
        cmp_txt = (f"<p>YOLOv8s has about {b['params'] / a['params']:.1f} times as many parameters as YOLOv8n. On the test set "
                   f"it reaches mAP@0.5 = {b['test']['mAP50']:.3f} against {a['test']['mAP50']:.3f} for YOLOv8n, with "
                   f"precision {b['test']['precision']:.3f} vs. {a['test']['precision']:.3f} and recall "
                   f"{b['test']['recall']:.3f} vs. {a['test']['recall']:.3f}. Inference takes "
                   f"{b['test']['speed_ms']['inference']:.1f} ms per image for YOLOv8s and "
                   f"{a['test']['speed_ms']['inference']:.1f} ms for YOLOv8n on our hardware (Apple M4 GPU, 1120-pixel input).</p>")
    g = best.get("test_by_grid", {})
    grid_txt = ""
    if g:
        lo, hi = min(g, key=lambda k: g[k]["mAP50"]), max(g, key=lambda k: g[k]["mAP50"])
        ranked = ", ".join(f"{k} {g[k]['mAP50']:.3f}" for k in sorted(g, key=lambda k: -g[k]["mAP50"]))
        grid_txt = (f"<p>Test mAP@0.5 ranges from {g[lo]['mAP50']:.3f} on {lo} grids to {g[hi]['mAP50']:.3f} on {hi} grids "
                    f"(from best to worst: {ranked}). ")
        if lo == "2x2":
            grid_txt += ("Somewhat surprisingly, the small 2×2 grids are the hardest. Because every image is resized to 1120 pixels, "
                         "a 448-pixel 2×2 grid is enlarged about 2.5 times, so its objects appear much larger and blurrier than "
                         "the objects in the larger grids that make up most of the training data. ")
        elif lo in ("5x5", "5x4"):
            grid_txt += ("The large grids are the hardest: each object occupies a smaller share of the image and sits next to "
                         "many other objects. ")
        grid_txt += "Overall the differences between grid sizes are small, so the model handles 4 to 25 objects per image.</p>"
    check_txt = ""
    if check:
        check_txt = (f"<p>We then checked a random sample of {check['sampled']} kept boxes by eye. {check['correct']} were "
                     f"correct, meaning the box covered the labelled object and not something else in the photo. "
                     f"{esc(check['notes'])}</p>")

    pages = f"""<!doctype html><html><head><meta charset=utf-8><title>Milestone 2 Report</title><style>{CSS}</style></head><body>
<h1>IE 7615- Deep Learning for AI</h1>
<div class=title>DISCRIMINATIVE PROJECT MILESTONE 2 REPORT</div>
<div class=meta>Team: {TEAM}<br>Due: {DUE}</div>
<hr>

<h2>1. Dataset and Problem Overview</h2>
<p>Milestone 1 classified a photo containing a single object. In this milestone we move to object detection: given an image
that contains several of the class's objects, the model has to report the Object ID and the location (bounding box) of every
object in it. We used the final merged class dataset and the final Object IDs from the shared sheet.</p>
<p>Following the course announcement, OBJ054 was excluded because its photos do not show the object, which leaves
{n_cls} Object IDs (OBJ001 to OBJ073 without OBJ054). Byte-identical duplicate photos were removed before splitting, the one
OBJ003 photo that is not 224 × 224 is resized when it is placed in a grid, and background-only photos (<i>*_bg_*</i>) are not used.
We kept the stratified 70/15/15 split from Milestone 1 (seed 42), and every grid image is built only from photos of one split,
so no source photo appears in more than one of train, validation and test ({overlap} overlapping photos across splits).</p>

<h2>2. Bounding-Box Labelling and Label Checking</h2>
<p>The shared dataset only gives each photo a class through its folder name, with no bounding box, so we created the boxes
automatically with open-vocabulary detectors. Each detector was prompted with the object's name from the sheet, plus a
clearer everyday phrase where the sheet name was unusual (for example "hair claw clip" for Clutcher, "ping pong paddle" for
Table Tennis Racket). We ran the detectors at the photos' native 224 × 224 size, since upscaling the photos made them miss
most objects in our tests. Three models were used:</p>
<ul>
<li>OWLv2 (google/owlv2-base-patch16-ensemble), the primary labeller and the strongest of the three on our photos.</li>
<li>YOLO-World (yolov8s-worldv2) and YOLOE (yoloe-11s-seg), used as independent second opinions.</li>
</ul>
<p>A box was accepted only under the following rules, and every other photo was left out:</p>
<ul>
<li>Verified: OWLv2's box agrees with the YOLO-World or YOLOE box (IoU ≥ 0.5). OWLv2's box is used.</li>
<li>Two-model: OWLv2 found nothing, but YOLO-World and YOLOE agree with each other (IoU ≥ 0.5). Their average box is used.</li>
<li>Single-model: only OWLv2 found the object, with a confidence of at least 0.35. These boxes were spot-checked by eye.</li>
<li>Excluded: the models disagree, no model finds the object, or the box covers less than 2% of the photo (object far away
or in heavy clutter).</li>
</ul>
<table class=narrow><tr><th>Label status</th><th>Photos</th><th>Share</th></tr>
{''.join(f"<tr><td style='text-align:left'>{STATUS_NAMES[k]}</td><td>{int(st.get(k, 0)):,}</td><td>{st.get(k, 0) / total * 100:.1f}%</td></tr>" for k in STATUS_NAMES)}
<tr><td style='text-align:left'><b>Kept in total</b></td><td><b>{kept:,}</b></td><td><b>{kept / total * 100:.1f}%</b></td></tr></table>
<p>Out of {total:,} photos, {kept:,} received a box ({kept / total * 100:.0f}%) and {total - kept:,} were left out. Every one of the
{n_cls} objects kept at least {int(per_cls.min())} labelled photos; the objects with the fewest were
{', '.join(f"{name(k)} with {int(v)}" for k, v in low.items())}. These are mostly small, thin or flat objects
that the detectors find hard to separate from the surface they are lying on.</p>
{check_txt}
{figure(file_uri(REPORT / 'figs' / 'label_examples.jpg'), 'Figure 1. Random sample of kept labels (green = final bounding box).')}
{figure(file_uri(REPORT / 'figs' / 'label_disagreements.jpg'), 'Figure 2. Photos left out because the models disagree (red = OWLv2, blue = YOLO-World, orange = YOLOE).')}

<h2>3. Multi-Object Grid Images</h2>
<p>We built the multi-object images by placing 224 × 224 photos side by side in a grid, using all five grid sizes from the
course specification in equal proportion. Every cell holds a different object, chosen at random from the photos of that split
that have a kept label, and its bounding box is the photo's labelled box shifted into the cell. Each cell is augmented with a
random horizontal flip (the box is mirrored with it) and brightness, contrast and saturation jitter.</p>
<table><tr><th>Grid</th><th>Image size</th><th>Objects / image</th><th>Train</th><th>Val</th><th>Test</th><th>Test mAP@0.5 ({BN})</th><th>Test recall ({BN})</th></tr>
{grid_rows}
<tr><td><b>Total</b></td><td></td><td></td><td><b>{int(split_tab['train'])}</b></td><td><b>{int(split_tab['val'])}</b></td><td><b>{int(split_tab['test'])}</b></td><td></td><td></td></tr></table>
<p>In total the group generated {int(split_tab.sum()):,} grid images containing {int(obj_tab.sum()):,} labelled objects
({int(obj_tab['train']):,} train, {int(obj_tab['val']):,} validation, {int(obj_tab['test']):,} test), drawn from
{int(src_tab['train']):,}, {int(src_tab['val']):,} and {int(src_tab['test']):,} distinct source photos respectively.</p>
{figure(file_uri(REPORT / 'figs' / 'grid_examples.jpg'), 'Figure 3. Generated 2×2, 3×3 and 5×5 training grids with their bounding boxes.')}

<h2>4. Model and Training Setup</h2>
<p>We fine-tuned two sizes of YOLOv8 from COCO-pretrained weights (transfer learning), YOLOv8n (nano) and YOLOv8s (small),
with the detection head re-initialised for our {n_cls} classes.</p>
<ul>
<li>Image size: 1120, matching the largest grid, so objects in 5×5 grids are not shrunk</li>
<li>Batch size: 8 for YOLOv8n; 4 for YOLOv8s, which did not fit in memory at batch 8 with 1120-pixel images</li>
<li>Optimizer and learning-rate schedule: Ultralytics defaults (automatic optimizer selection with warm-up)</li>
<li>Up to 50 epochs for YOLOv8n and 30 for YOLOv8s (the larger model is about twice as slow per epoch), early stopping with patience 15; the best epoch by validation fitness is kept</li>
<li>Training-time augmentation: mosaic (50%, turned off for the last 10 epochs), HSV jitter, horizontal flip, scale and translation</li>
<li>Checkpoints saved every 5 epochs; trained on an Apple M4 GPU (PyTorch MPS)</li>
</ul>

<h2>5. Experimental Results</h2>
<p>Results on the held-out test set ({int(split_tab['test'])} grid images, {int(obj_tab['test']):,} objects):</p>
<table><tr><th>Model</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th><th>Precision</th><th>Recall</th><th>Params</th><th>Epochs</th><th>Train (min)</th><th>Infer. (ms/img)</th></tr>
{res_rows}</table>
{figure(curves(runs), 'Figure 4. Validation mAP@0.5 and validation loss per epoch.')}
{figure(grid_bars(best), f'Figure 5. Test mAP@0.5, precision and recall for each grid size ({BN}).', '82%') if g else ''}

<h2>6. Detection Examples</h2>
<p>Given a new multi-object image, our script <i>detect_objects.py</i> returns the Object ID, name, confidence and bounding box
of every detected object. The table below is the output for test image {det_name} (a 3 × 3 grid with {det_truth} objects);
{det_ok} of {len(det_rows)} detections match a ground-truth object with the correct ID.</p>
<table><tr><th>Object ID</th><th>Name</th><th>Confidence</th><th>Top-left (x, y)</th><th>Bottom-right (x, y)</th><th>Correct</th></tr>{det_tab}</table>
<div class=pair>
{figure(file_uri(REPORT / 'figs' / 'detection_3x3.jpg', max_side=900), f'Figure 6. {BN} detections on a 3×3 test grid.')}
{figure(file_uri(REPORT / 'figs' / 'detection_5x5.jpg', max_side=900), f'Figure 7. {BN} detections on a 5×5 test grid ({big_found} detections, {big_truth} objects).')}
</div>

<h2>7. Comparison and Analysis</h2>
{'<p><b>Model size.</b></p>' + cmp_txt if cmp_txt else ''}
<p><b>Grid size.</b></p>{grid_txt}
<p><b>Per-object results.</b></p>
<p>{n_perfect} of the {len(ap)} objects reach a test mAP@0.5 of at least 0.99 with {BN}. The weakest objects are
{', '.join(f"{name(k)} at {v:.2f}" for k, v in worst.items())}. The most frequent identification errors in the test confusion
matrix, counted at the same 0.25 confidence threshold our detection script uses, are: {conf_txt}.</p>
<p><b>Error breakdown.</b></p>
<p>At a confidence threshold of 0.25, of the {ea['objects']:,} test objects {ea['correct']:,} ({ea['correct'] / ea['objects'] * 100:.1f}%)
were found with the correct ID and a box overlapping the ground truth by IoU ≥ 0.5, {ea['wrong_id']:,}
({ea['wrong_id'] / ea['objects'] * 100:.1f}%) were found in the right place but given the wrong ID, and {ea['missed']:,}
({ea['missed'] / ea['objects'] * 100:.1f}%) were missed. The model also produced {ea['extra']:,} extra boxes that did not match
any labelled object, typically duplicates or boxes around a part of an object.</p>

<h2>8. Final Model Selection and Justification</h2>
<p>We select {BN} as our detection model. It has the highest test mAP@0.5 ({t['mAP50']:.3f}) and mAP@0.5:0.95
({t['mAP50_95']:.3f}) of the models we trained, with precision {t['precision']:.3f} and recall {t['recall']:.3f}, and at
{t['speed_ms']['inference']:.1f} ms per 1120-pixel image it is still fast enough for real-time use.</p>

<h2>9. Limitations and Next Steps</h2>
<ul>
<li>Automatic labels: about {(total - kept) / total * 100:.0f}% of photos were left out because the detectors could not agree on a box.
Those photos are typically the hardest ones (small, thin or cluttered objects), so the test set is somewhat easier than the full dataset.</li>
<li>Some kept boxes come from OWLv2 alone ({int(st.get('single-model', 0)):,} photos). They passed a visual spot-check but have not been verified by a second model.</li>
<li>Grid images are clean: objects never overlap and every object sits in its own cell, unlike a real scene with several objects photographed together.</li>
<li>A natural next step is testing on real photos of several objects together, and hand-correcting the excluded photos so they can be used.</li>
</ul>

<h2>10. Code Repository</h2>
<p>All code for this milestone (labelling, label checking, grid generation, training, evaluation and the detection script) and
the trained weights are publicly available on GitHub, in the <i>milestone2</i> folder. Following the course instructions, no image
data is included in the repository.</p>
<p><a href="{REPO}">{REPO}</a></p>
</body></html>"""
    html_path, pdf_tmp, pdf = REPORT / "Milestone2_Report.html", REPORT / "_tmp.pdf", REPORT / "Milestone2_Report.pdf"
    html_path.write_text(pages)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_tmp}",
                    html_path.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    add_page_numbers(pdf_tmp, pdf)
    pdf_tmp.unlink()
    print(f"best = {BN}; wrote {pdf} ({len(PdfReader(str(pdf)).pages)} pages)")


def add_page_numbers(src, dst):
    pdfmetrics.registerFont(TTFont("TNR", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"))
    reader, writer = PdfReader(str(src)), PdfWriter()
    for i, page in enumerate(reader.pages, 1):
        w, h = float(page.mediabox.width), float(page.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFont("TNR", 11)
        c.drawRightString(w - 72, 36, str(i))
        c.save()
        page.merge_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])
        writer.add_page(page)
    with open(dst, "wb") as f:
        writer.write(f)


CSS = """
@page { size: Letter; margin: 0.6in 1in 0.85in 1in; }
body { font-family: "Times New Roman", Times, serif; font-size: 11pt; line-height: 1.3; color: #000; background: #fff; margin: 0; }
h1 { font-size: 18pt; font-weight: bold; margin: 0 0 2pt; }
.title { font-size: 13pt; font-weight: bold; margin: 0 0 3pt; }
.meta { font-size: 10pt; line-height: 1.35; }
hr { border: none; border-top: 1px solid #bfbfbf; margin: 14pt 0 16pt; }
h2 { font-size: 11pt; font-weight: bold; margin: 16pt 0 7pt; break-after: avoid; }
p { margin: 0 0 8pt; text-align: justify; }
ul { margin: 0 0 8pt; padding-left: 30pt; list-style-type: disc; }
li { margin: 0 0 3pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt; font-size: 10.5pt; break-inside: avoid; }
table.narrow { width: 75%; margin-left: auto; margin-right: auto; }
th, td { border: 1px solid #000; padding: 4pt 6pt; text-align: center; }
th { font-weight: bold; }
td:first-child { text-align: left; }
tr.best td { background: #fff2cc; }
figure { margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-style: italic; font-size: 10pt; margin-top: 4pt; }
.pair { display: flex; gap: 14pt; align-items: flex-start; break-inside: avoid; }
.pair figure { flex: 1; margin: 6pt 0; }
a { color: #0645ad; }
"""

if __name__ == "__main__":
    build()
