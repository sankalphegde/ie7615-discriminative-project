"""Visual check of the auto-generated boxes: draws a random sample of labelled photos per label status.

Green = final box. For excluded photos, every model's box is drawn (red = OWLv2, blue = YOLO-World,
orange = YOLOE) so the disagreement can be inspected.

Usage:  python check_labels.py --data "../data/raw/IE7615 Deep Learning for AI" [--per-status 24]
Output: labeling/check_<status>.jpg and labeling/label_summary.csv
"""
import argparse
import ast
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

COLORS = {"owl_box": (230, 40, 60), "world_box": (40, 110, 230), "yoloe_box": (245, 150, 20)}


def draw_sheet(df, data_dir, out, final=True, cols=8, size=160):
    rows = -(-len(df) // cols)
    sheet = Image.new("RGB", (cols * size, rows * (size + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (_, r) in enumerate(df.iterrows()):
        with Image.open(data_dir / r.path) as im:
            img = im.convert("RGB")
        s = size / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
        dd = ImageDraw.Draw(img)
        if final and pd.notna(r.x1):
            dd.rectangle([r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s], outline=(30, 200, 80), width=3)
        else:
            for col, colour in COLORS.items():
                if isinstance(r[col], str):
                    b = ast.literal_eval(r[col])
                    dd.rectangle([v * s for v in b], outline=colour, width=2)
        x, y = (k % cols) * size, (k // cols) * (size + 14)
        sheet.paste(img, (x, y))
        d.text((x + 2, y + size), f"{r.label} owl={r.owl_conf:.2f}", fill=(60, 60, 60))
    sheet.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", default="labeling/box_labels.csv")
    ap.add_argument("--per-status", type=int, default=24)
    args = ap.parse_args()
    df = pd.read_csv(args.labels)
    data_dir = Path(args.data)
    for status, g in df.groupby("status"):
        sample = g.sample(min(len(g), args.per_status), random_state=0)
        tag = status.replace("excluded: ", "excluded_").replace(" ", "_")
        draw_sheet(sample, data_dir, Path("labeling") / f"check_{tag}.jpg", final=not status.startswith("excluded"))
    summary = df.assign(kept=~df["status"].str.startswith("excluded")).groupby("label").agg(
        photos=("path", "size"), kept=("kept", "sum"),
        verified=("status", lambda s: (s == "verified").sum()),
        single_model=("status", lambda s: (s == "single-model").sum()))
    summary.to_csv("labeling/label_summary.csv")
    print(df["status"].value_counts().to_string())
    print(f"\nkept {int(summary.kept.sum())}/{len(df)} photos; fewest kept: "
          + ", ".join(f"{i} {int(r.kept)}" for i, r in summary.sort_values("kept").head(6).iterrows()))


if __name__ == "__main__":
    main()
