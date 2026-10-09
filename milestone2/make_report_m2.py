"""Build the Milestone 2 report (HTML + PDF) from the YOLOv8 runs in runs/ and the generated dataset.

Usage:  python make_report_m2.py
Output: report/Milestone2_Report.pdf (+ .html, figs/)
"""
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

HERE = Path(__file__).resolve().parent
RUNS, DATA, REPORT = HERE / "runs", HERE / "yolo_data", HERE / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO = "https://github.com/sankalphegde/ie7615-discriminative-project"
TEAM = ["Sankalp Hegde", "Reshma Jose", "Jairaghavendra Sridhar"]
SERIES = {"yolov8n": "#2a78d6", "yolov8s": "#eb6834"}
INK, MUTED, GRID = "#1c1f24", "#5b6270", "#e3e6eb"
_IDS = next(p for p in [HERE.parent / "data" / "object_ids.csv", HERE.parent / "object_ids.csv"] if p.exists())
NAMES = {r["OBJECT_ID"]: r["OBJECT_NAME"] for r in csv.DictReader(open(_IDS))}
LABEL = {"yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s"}


def esc(s):
    return html.escape(str(s))


def uri(path_or_fig, kind="png"):
    if isinstance(path_or_fig, (str, Path)):
        p = Path(path_or_fig)
        mime = "jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "png"
        return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    buf = io.BytesIO()
    path_or_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(path_or_fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def obj(o):
    return f"{o} ({esc(NAMES[o])})" if o in NAMES else o


def load_runs():
    runs = {}
    for name in ["yolov8n", "yolov8s"]:
        d = RUNS / name
        if (d / "final_metrics.json").exists():
            r = json.loads((d / "final_metrics.json").read_text())
            r["history"] = pd.read_csv(d / "results.csv")
            r["history"].columns = [c.strip() for c in r["history"].columns]
            args = d / "args.yaml"
            r["epochs_run"] = len(r["history"])
            r["train_min"] = float(r["history"]["time"].iloc[-1]) / 60 if "time" in r["history"] else None
            runs[name] = r
    return runs


def curves_fig(runs):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    for name, r in runs.items():
        h = r["history"]
        ax[0].plot(h["epoch"], h["metrics/mAP50(B)"] * 100, color=SERIES[name], lw=2, label=LABEL[name])
        ax[1].plot(h["epoch"], h["metrics/mAP50-95(B)"] * 100, color=SERIES[name], lw=2, label=LABEL[name])
    ax[0].set_title("Validation mAP@0.5 (%)", fontsize=10, color=INK, loc="left")
    ax[1].set_title("Validation mAP@0.5:0.95 (%)", fontsize=10, color=INK, loc="left")
    for a in ax:
        style(a)
        a.set_xlabel("epoch", fontsize=8, color=MUTED)
        a.set_ylim(0, 101)
    ax[1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return uri(fig)


def per_class_fig(r):
    ap = pd.Series(r["test"]["per_class_mAP50_95"]).sort_values()
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.bar(range(len(ap)), ap.values * 100, color="#2a78d6", width=0.75)
    ax.set_xticks(range(len(ap)), ap.index, rotation=90, fontsize=5.5)
    ax.set_ylabel("test mAP@0.5:0.95 (%)", fontsize=8, color=MUTED)
    ax.set_ylim(0, 105)
    style(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return uri(fig), ap


def read_labels(p):
    out = []
    if p.exists():
        for line in p.read_text().split("\n"):
            if line.strip():
                c, x, y, w, h = line.split()
                x, y, w, h = (float(v) * 640 for v in (x, y, w, h))
                out.append((f"OBJ{int(c) + 1:03d}", (x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
    return out


def detection_examples(weights, n=4):
    from ultralytics import YOLO
    model = YOLO(weights)
    allimgs = sorted((DATA / "images" / "test").glob("*.jpg"))
    imgs = [p for p in allimgs if len(read_labels(DATA / "labels" / "test" / f"{p.stem}.txt")) >= 4][:n]
    five = [p for p in allimgs if len(read_labels(DATA / "labels" / "test" / f"{p.stem}.txt")) >= 5]
    if five and five[0] not in imgs:
        imgs = [five[0]] + imgs[:n - 1]
    elif five:
        imgs.remove(five[0]); imgs = [five[0]] + imgs
    tiles, listing = [], None
    for k, p in enumerate(imgs):
        res = model.predict(str(p), conf=0.25, imgsz=640, verbose=False)[0]
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        for _, (x1, y1, x2, y2) in read_labels(DATA / "labels" / "test" / f"{p.stem}.txt"):
            d.rectangle([x1, y1, x2, y2], outline=(40, 200, 90), width=3)
        rows = []
        for b in res.boxes:
            oid = res.names[int(b.cls)]
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            lab = f"{oid} {NAMES.get(oid, '')} {float(b.conf):.2f}"
            d.rectangle([x1, y1, x2, y2], outline=(230, 30, 60), width=3)
            d.rectangle([x1, max(0, y1 - 15), x1 + d.textlength(lab) + 6, y1], fill=(230, 30, 60))
            d.text((x1 + 3, max(0, y1 - 14)), lab, fill="white")
            rows.append((oid, NAMES.get(oid, ""), float(b.conf), (round(x1), round(y1), round(x2), round(y2))))
        if k == 0:
            listing = (p.name, sorted(rows, key=lambda r: (r[3][1], r[3][0])),
                       read_labels(DATA / "labels" / "test" / f"{p.stem}.txt"))
        tiles.append(im.resize((480, 480)))
    grid = Image.new("RGB", (2 * 480 + 12, 2 * 480 + 12), "white")
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 2) * 492, (i // 2) * 492))
    (REPORT / "figs").mkdir(parents=True, exist_ok=True)
    grid.save(REPORT / "figs" / "detections.jpg", quality=88)
    return uri(REPORT / "figs" / "detections.jpg"), listing


def first(*paths):
    return next((p for p in paths if p.exists()), None)


def build():
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs/<name>/final_metrics.json found")
    best_name = max(runs, key=lambda k: (runs[k]["test"]["mAP50_95"], runs[k]["test"]["mAP50"]))
    best = runs[best_name]
    man = pd.read_csv(DATA / "manifest.csv")
    per_split = man.groupby("split").agg(objects=("object_id", "size"), images=("image", "nunique"))
    per_split = per_split.reindex(["train", "val", "test"])
    inst = man[man["split"] == "train"]["object_id"].value_counts()

    REPORT.mkdir(exist_ok=True)
    gen_uri = uri(HERE / "figs" / "generated_examples.jpg")
    curves_uri = curves_fig(runs)
    pc_uri, ap = per_class_fig(best)
    det_uri, listing = detection_examples(RUNS / best_name / "weights" / "best.pt")
    ev = RUNS / f"{best_name}_eval_test"
    cm = first(ev / "confusion_matrix_normalized.png", RUNS / best_name / "confusion_matrix_normalized.png")
    pr = first(ev / "BoxPR_curve.png", ev / "PR_curve.png", RUNS / best_name / "BoxPR_curve.png")

    rows = ""
    for name, r in sorted(runs.items(), key=lambda kv: -kv[1]["test"]["mAP50_95"]):
        t = r["test"]
        sp = t["speed_ms"]
        rows += (f"<tr{' class=best' if name == best_name else ''}><td><b>{LABEL[name]}</b>{' ★' if name == best_name else ''}</td>"
                 f"<td class=n>{t['precision']:.3f}</td><td class=n>{t['recall']:.3f}</td><td class=n>{t['mAP50']:.3f}</td>"
                 f"<td class=n>{t['mAP50_95']:.3f}</td><td class=n>{r['params'] / 1e6:.1f}M</td>"
                 f"<td class=n>{r['epochs_run']}</td><td class=n>{(r['train_min'] or 0):.0f}</td>"
                 f"<td class=n>{sp.get('inference', 0):.1f}</td></tr>")

    worst = ap.head(6)
    strong = int((ap >= 0.95).sum())
    det_rows = ""
    fname, preds, gts = listing
    for oid, name, conf, (x1, y1, x2, y2) in preds:
        det_rows += (f"<tr><td><b>{oid}</b></td><td>{esc(name)}</td><td class=n>{conf:.2f}</td>"
                     f"<td class=n>({x1}, {y1})</td><td class=n>({x2}, {y2})</td></tr>")
    gt_ids = sorted(o for o, _ in gts)
    pred_ids = sorted(o for o, *_ in preds)
    hit = "all ground-truth objects found" if sorted(set(gt_ids)) == sorted(set(pred_ids)) else \
        f"ground truth: {', '.join(gt_ids)}"

    n_cmp = ""
    if len(runs) == 2:
        a, b = runs["yolov8n"]["test"], runs["yolov8s"]["test"]
        n_cmp = (f"<p><b>Model size.</b> YOLOv8s has {runs['yolov8s']['params'] / runs['yolov8n']['params']:.1f}× the "
                 f"parameters of YOLOv8n. On the test set it scores mAP@0.5:0.95 {b['mAP50_95']:.3f} vs. "
                 f"{a['mAP50_95']:.3f} ({(b['mAP50_95'] - a['mAP50_95']) * 100:+.1f} points) and mAP@0.5 "
                 f"{b['mAP50']:.3f} vs. {a['mAP50']:.3f}. Inference takes {b['speed_ms'].get('inference', 0):.1f} ms vs. "
                 f"{a['speed_ms'].get('inference', 0):.1f} ms per image on our hardware.</p>")

    t = best["test"]
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>Milestone 2 Report</title>
<style>{CSS}</style></head><body>
<header class=cover>
  <div class=course>IE 7615 Deep Learning for AI · Discriminative Project</div>
  <h1>Milestone 2: Multi-Object Detection with YOLOv8</h1>
  <div class=subtitle>Detecting, identifying and locating the class's 73 objects in multi-object images</div>
  <div class=meta>Study Group 6: {' · '.join(TEAM)}</div>
</header>

<section class=summary>
  <div class=kpi><div class=v>{t['mAP50']:.3f}</div><div class=l>test mAP@0.5 ({LABEL[best_name]})</div></div>
  <div class=kpi><div class=v>{t['mAP50_95']:.3f}</div><div class=l>test mAP@0.5:0.95</div></div>
  <div class=kpi><div class=v>{t['precision']:.3f} / {t['recall']:.3f}</div><div class=l>precision / recall</div></div>
  <div class=kpi><div class=v>{int(per_split['images'].sum()):,}</div><div class=l>generated multi-object images</div></div>
</section>

<h2><span>1</span>Problem Overview</h2>
<p>Milestone 1 identified a single object per image. Milestone 2 extends this to <b>object detection</b>: given one
image containing several of the class's 73 objects, the model must report the <b>Object ID and the location
(bounding box)</b> of every object. We generate a multi-object dataset by compositing randomly selected
single-object images from the final merged class dataset, and fine-tune a COCO-pretrained YOLOv8 detector on it
(transfer learning). Object IDs OBJ001–OBJ073 match the final Object ID sheet; YOLO class <i>k</i> = OBJ<i>k+1</i>.</p>

<h2><span>2</span>Multi-Object Dataset Generation</h2>
<p>Each generated image is a 640×640 canvas with <b>2–6 distinct objects</b> pasted onto a background. Because our
script places every object, its bounding box is known exactly and the YOLO label file is written automatically;
no manual annotation was needed.</p>
<ul>
<li><b>Object selection:</b> objects are drawn at random from the cleaned single-object dataset of Milestone 1 (6,954 images,
73 classes; background-only <code>*_bg_*</code> photos are never used as objects).</li>
<li><b>No train/test leakage:</b> training composites use only the Milestone 1 <i>training</i> photos, validation composites only
<i>validation</i> photos and test composites only <i>test</i> photos, so the detector is always evaluated on photos it never saw.</li>
<li><b>Backgrounds:</b> 60% use the class's background-only photos (rooms, floors, desks, outdoors); the rest use plain
colours or blurred noise textures.</li>
<li><b>Layouts:</b> 70% random placement (overlap limited to IoU ≤ 0.1, so partial occlusion occurs), 30% jittered 2×2 / 3×3 grids.</li>
<li><b>Augmentation per object:</b> random size (120–270 px), horizontal flip, brightness / contrast / saturation jitter,
occasional blur and ±15° rotation; plus whole-image brightness jitter. YOLOv8's own training augmentation (mosaic, HSV,
flips, scale/translate) is applied on top.</li>
</ul>
<table class=compact><tr><th>Split</th><th>Images</th><th>Objects</th><th>Objects / image</th></tr>
{''.join(f"<tr><td>{s}</td><td class=n>{int(r.images):,}</td><td class=n>{int(r.objects):,}</td><td class=n>{r.objects / r.images:.1f}</td></tr>" for s, r in per_split.iterrows())}
</table>
<p>Every object appears in all three splits; in the training set each object has {inst.min()}–{inst.max()} instances.
The assignment asks each student for at least 100 multi-object images; together the group generated
{int(per_split['images'].sum()):,}.</p>
<figure><img src="{gen_uri}"><figcaption>Figure 1. Generated training images with their automatically written ground-truth boxes.</figcaption></figure>

<h2><span>3</span>Model and Training</h2>
<p>We fine-tuned two sizes of <b>YOLOv8</b> (Ultralytics), both initialised from COCO-pretrained weights, so the backbone
already knows generic object features and only has to learn our 73 objects. The detection head was re-initialised for 73 classes.</p>
<table class=compact>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Models</td><td>YOLOv8n (nano) and YOLOv8s (small), COCO-pretrained</td></tr>
<tr><td>Input size / batch</td><td>640 × 640 / 16</td></tr>
<tr><td>Epochs</td><td>up to 60 (n) and 40 (s), early stopping patience 15, best weights by validation fitness</td></tr>
<tr><td>Optimizer</td><td>Ultralytics default (auto-selected optimizer and learning-rate schedule, warm-up)</td></tr>
<tr><td>Augmentation</td><td>mosaic (off for the last 10 epochs), HSV jitter, horizontal flip, scale/translate</td></tr>
<tr><td>Hardware</td><td>Apple M4 GPU (PyTorch MPS)</td></tr>
</table>

<h2><span>4</span>Results</h2>
<table><tr><th>Model</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th><th>Params</th><th>Epochs</th><th>Train (min)</th><th>ms / image</th></tr>
{rows}</table>
<p class=note>★ = selected model. Metrics are on the held-out test set ({int(per_split.loc['test', 'images'])} images,
{int(per_split.loc['test', 'objects']):,} objects) built from Milestone 1 test photos only.</p>
<figure><img src="{curves_uri}"><figcaption>Figure 2. Validation mAP per epoch.</figcaption></figure>
{n_cmp}
<div class=two>
{f'<figure><img src="{uri(pr)}"><figcaption>Figure 3. Precision–recall curve, {LABEL[best_name]} (test).</figcaption></figure>' if pr else ''}
{f'<figure><img src="{uri(cm)}"><figcaption>Figure 4. Normalised confusion matrix, {LABEL[best_name]} (test).</figcaption></figure>' if cm else ''}
</div>

<h2><span>5</span>Detecting IDs and Locations</h2>
<p>Given a new multi-object image, <code>detect_objects.py</code> returns every detected object's Object ID, name,
confidence and bounding box. Example output for test image <code>{esc(fname)}</code> ({hit}):</p>
<table class=compact><tr><th>Object ID</th><th>Name</th><th>Confidence</th><th>Top-left (x, y)</th><th>Bottom-right (x, y)</th></tr>{det_rows}</table>
<figure><img src="{det_uri}" style="max-width:78%"><figcaption>Figure 5. Test-set detections by {LABEL[best_name]}
(red = prediction with ID, name and confidence; green = ground truth).</figcaption></figure>

<h2><span>6</span>Analysis</h2>
<p><b>Per-object accuracy.</b> {strong} of 73 objects reach a test mAP@0.5:0.95 of at least 0.95. The weakest objects are
{', '.join(f"{obj(k)} {v:.2f}" for k, v in worst.items())}.</p>
<figure><img src="{pc_uri}"><figcaption>Figure 6. Test mAP@0.5:0.95 per object, {LABEL[best_name]}.</figcaption></figure>
<p><b>Localisation.</b> mAP@0.5 is close to mAP@0.5:0.95 because each ground-truth box is the full pasted photo, a
crisp rectangle that is easy to localise precisely. Most remaining errors are <i>identification</i> errors between
similar-looking objects, or objects that are small or partly covered by another photo.</p>
<p><b>Transfer learning.</b> Starting from COCO weights, the detector passed mAP@0.5 = {best['history']['metrics/mAP50(B)'].iloc[min(7, len(best['history']) - 1)]:.2f}
on validation within {min(8, len(best['history']))} epochs.</p>

<h2><span>7</span>Final Model Selection</h2>
<p>We select <b>{LABEL[best_name]}</b>: the highest test mAP@0.5:0.95 ({t['mAP50_95']:.3f}) and mAP@0.5 ({t['mAP50']:.3f}),
with precision {t['precision']:.3f} and recall {t['recall']:.3f}, at {t['speed_ms'].get('inference', 0):.1f} ms per image.</p>

<h2><span>8</span>Limitations and Next Steps</h2>
<ul>
<li><b>Box definition:</b> each label covers the whole pasted photo, not a tight box around the object inside it, so the
model learns to find object <i>photos</i>. Tight boxes would need manual annotation or a segmentation step.</li>
<li><b>Synthetic scenes:</b> composites have sharp tile edges and no real lighting interaction; performance on real photos
with several physical objects will be lower. Testing on real multi-object photos is the natural next step.</li>
<li><b>Similar objects</b> (e.g. two kinds of sunglasses, two watches) remain the main source of identification errors.</li>
</ul>

<h2><span>9</span>Code Repository</h2>
<p>All code (dataset generator, training, evaluation, detection script) and trained weights are on GitHub:<br>
<a href="{REPO}">{REPO}</a> (folder <code>milestone2/</code>).</p>
</body></html>"""
    out_html, out_pdf = REPORT / "Milestone2_Report.html", REPORT / "Milestone2_Report.pdf"
    out_html.write_text(page)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out_pdf}",
                    out_html.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"best = {best_name}; wrote {out_pdf}")


CSS = """
@page { size: Letter; margin: 0.6in 0.65in; }
:root { --ink:#1c1f24; --muted:#5b6270; --line:#dde1e7; --soft:#f4f6f9; --accent:#b3122e; }
* { box-sizing: border-box; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; color: var(--ink); font-size: 10pt;
       line-height: 1.5; margin: 0; background: #fff; }
.cover { border-bottom: 3px solid var(--accent); padding-bottom: 12px; margin-bottom: 14px; }
.course { color: var(--accent); font-weight: 700; font-size: 8.5pt; letter-spacing: .08em; text-transform: uppercase; }
h1 { font-size: 21pt; margin: 4px 0 2px; }
.subtitle { font-size: 11.5pt; color: var(--muted); }
.meta { margin-top: 6px; font-size: 9pt; color: var(--muted); }
h2 { font-size: 13pt; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line);
     display: flex; align-items: center; gap: 8px; break-after: avoid; }
h2 span { background: var(--accent); color: #fff; border-radius: 50%; width: 21px; height: 21px; font-size: 9pt;
          display: inline-flex; align-items: center; justify-content: center; }
p { margin: 5px 0 8px; text-align: justify; }
ul { margin: 4px 0 8px; padding-left: 18px; } li { margin: 3px 0; }
table { width: 100%; border-collapse: collapse; margin: 6px 0 10px; font-size: 8.8pt; break-inside: avoid; }
th { text-align: left; background: var(--soft); font-weight: 600; padding: 5px 7px; border-bottom: 1px solid var(--line); }
td { padding: 5px 7px; border-bottom: 1px solid var(--line); vertical-align: top; }
td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.compact { width: auto; min-width: 55%; }
tr.best td { background: #f1f7ff; }
.note { font-size: 8.5pt; color: var(--muted); margin-top: -4px; }
figure { margin: 10px 0 12px; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption { font-size: 8.5pt; color: var(--muted); margin-top: 4px; }
.two { display: flex; gap: 14px; align-items: flex-start; break-inside: avoid; }
.two figure { flex: 1; max-width: 50%; }
.summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 10px 0 6px; }
.kpi { border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; }
.kpi .v { font-size: 16pt; font-weight: 700; font-variant-numeric: tabular-nums; }
.kpi .l { font-size: 8pt; color: var(--muted); }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.3pt; background: var(--soft); padding: 1px 4px; border-radius: 3px; }
a { color: #0645ad; }
"""

if __name__ == "__main__":
    build()
