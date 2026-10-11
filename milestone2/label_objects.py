"""Auto-label a tight bounding box for the object in every single-object photo, with two detectors.

The class dataset gives each photo a class (its folder) but no box. Three open-vocabulary detectors are prompted
with the object's name: OWLv2 (google/owlv2-base-patch16-ensemble, the primary labeller), YOLO-World
(yolov8s-worldv2) and YOLOE (yoloe-11s-seg). Rules:
  * verified      - OWLv2's box agrees (IoU >= 0.5) with YOLO-World or YOLOE  -> keep OWLv2's box
  * two-model     - OWLv2 found nothing, but YOLO-World and YOLOE agree        -> keep their average box
  * single-model  - only OWLv2 found it, with score >= 0.35                    -> keep (spot-checked visually)
  * excluded      - anything else, or a box covering < 2% of the photo (object far away / heavy clutter)

Runs at the photos' native 224x224 resolution: upscaling them to 640 makes both detectors miss most objects.

Usage:
    python label_objects.py --data "../data/raw/IE7615 Deep Learning for AI" --split ../outputs/split.csv
Output: labeling/box_labels.csv (one row per photo: both models' boxes and scores, IoU, final box, status)
"""
import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from ultralytics import YOLO, YOLOE

EXCLUDED_IDS = {"OBJ054"}            # per course announcement: not a photo of the object
MIN_IOU, MIN_AREA = 0.5, 0.02        # models must agree; box must cover >= 2% of the photo
CONF = 0.03                          # low floor for the YOLO models: agreement is the real filter
OWL_MIN, OWL_SOLO = 0.1, 0.35        # OWLv2 candidate floor / minimum score to keep without a second model
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Prompts per Object ID: a clearer everyday phrase first, then the sheet's own name.
PROMPTS = {
    "OBJ003": ["notebook"], "OBJ004": ["toy car"], "OBJ005": ["computer mouse"], "OBJ006": ["capybara plush toy", "stuffed animal"],
    "OBJ009": ["bluetooth speaker", "speaker"], "OBJ010": ["sneakers", "shoes"], "OBJ012": ["computer keyboard", "keyboard"],
    "OBJ013": ["mobile phone", "smartphone"], "OBJ015": ["water tumbler", "tumbler cup"], "OBJ016": ["portable ssd", "small device"],
    "OBJ017": ["game controller"], "OBJ018": ["rubiks cube"], "OBJ019": ["potted plant", "plant"], "OBJ020": ["airpods case", "earbuds case"],
    "OBJ021": ["bunny plush toy", "stuffed animal"], "OBJ023": ["bracelet"], "OBJ024": ["plush doll", "stuffed animal"],
    "OBJ025": ["watch", "wristwatch"], "OBJ026": ["baseball cap", "cap"], "OBJ030": ["tennis racket"], "OBJ031": ["card box", "box"],
    "OBJ032": ["audio interface", "electronic device"], "OBJ033": ["toy duck", "duck"], "OBJ034": ["ping pong paddle", "table tennis racket"],
    "OBJ036": ["pencil case", "pouch"], "OBJ038": ["perfume bottle"], "OBJ040": ["hair claw clip", "hair clip"],
    "OBJ041": ["table fan", "fan"], "OBJ042": ["mouth guard case", "small case"], "OBJ043": ["swimming goggles", "goggles"],
    "OBJ044": ["anime figurine", "figurine"], "OBJ045": ["sunglasses"], "OBJ046": ["smartwatch", "watch"], "OBJ048": ["power bank", "wireless charger"],
    "OBJ049": ["pickleball paddle", "paddle"], "OBJ051": ["tissue box", "tissue pack"], "OBJ053": ["lotion bottle", "moisturizer bottle"],
    "OBJ055": ["calculator"], "OBJ056": ["tote bag", "bag"], "OBJ057": ["power bank"], "OBJ058": ["footstool", "stool"],
    "OBJ060": ["sleep mask", "eye mask"], "OBJ061": ["card wallet", "wallet"], "OBJ063": ["measuring bottle", "bottle"],
    "OBJ064": ["galaxy projector", "projector"], "OBJ065": ["analog watch", "watch"], "OBJ066": ["hair oil bottle", "bottle"],
    "OBJ067": ["wallet"], "OBJ068": ["neck pillow", "pillow"], "OBJ069": ["kitchen knife", "knife"],
    "OBJ070": ["astronaut projector", "astronaut figure"], "OBJ071": ["sloth plush toy", "stuffed animal"], "OBJ073": ["bowl"],
}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def top_box(result):
    if not len(result.boxes):
        return None, 0.0
    i = int(result.boxes.conf.argmax())
    return [float(v) for v in result.boxes.xyxy[i]], float(result.boxes.conf[i])


def owl_boxes(proc, model, images, prompts):
    """Best OWLv2 box per image over all prompts, in the image's own pixel coordinates."""
    queries = [[f"a photo of a {p}" for p in prompts]] * len(images)
    inputs = proc(text=queries, images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    sizes = torch.tensor([[max(im.size)] * 2 for im in images])        # OWLv2 pads images to a square
    res = proc.post_process_grounded_object_detection(out, threshold=OWL_MIN, target_sizes=sizes)
    best = []
    for r, im in zip(res, images):
        if len(r["scores"]):
            i = int(r["scores"].argmax())
            b = [float(v) for v in r["boxes"][i]]
            b = [max(0, b[0]), max(0, b[1]), min(im.size[0], b[2]), min(im.size[1], b[3])]
            best.append((b, float(r["scores"][i])))
        else:
            best.append((None, 0.0))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="labeling/box_labels.csv")
    args = ap.parse_args()

    data = Path(args.data)
    root = Path(__file__).resolve().parent.parent
    ids_csv = next(p for p in [root / "data" / "object_ids.csv", root / "object_ids.csv"] if p.exists())
    names = {r["OBJECT_ID"]: r["OBJECT_NAME"] for r in csv.DictReader(open(ids_csv))}
    split = pd.read_csv(args.split)
    split = split[~split["label"].isin(EXCLUDED_IDS)]
    world, yoloe = YOLO("labeling/yolov8s-worldv2.pt"), YOLOE("labeling/yoloe-11s-seg.pt")
    owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()

    rows = []
    for oid, group in split.groupby("label"):
        prompts = PROMPTS.get(oid, []) + [names[oid].lower()]
        prompts = list(dict.fromkeys(prompts))           # de-duplicate, keep order
        world.set_classes(prompts)
        yoloe.set_classes(prompts, yoloe.get_text_pe(prompts))
        paths = [str(data / p) for p in group["path"]]
        for i in range(0, len(paths), 16):
            batch = paths[i:i + 16]
            images = [Image.open(p).convert("RGB") for p in batch]
            rw = world.predict(batch, conf=CONF, imgsz=224, verbose=False)
            re = yoloe.predict(batch, conf=CONF, imgsz=224, verbose=False)
            ro = owl_boxes(owl_proc, owl, images, prompts)
            for p, im, a, b, (bo, co), (_, meta) in zip(batch, images, rw, re, ro, group.iloc[i:i + 16].iterrows()):
                W, H = im.size
                bw, cw = top_box(a)
                be, ce = top_box(b)
                iou_w = iou(bo, bw) if bo and bw else 0.0
                iou_e = iou(bo, be) if bo and be else 0.0
                iou_we = iou(bw, be) if bw and be else 0.0
                if bo and max(iou_w, iou_e) >= MIN_IOU:
                    status, final = "verified", bo
                elif not bo and iou_we >= MIN_IOU:
                    status, final = "two-model", [(x + y) / 2 for x, y in zip(bw, be)]
                elif bo and co >= OWL_SOLO:
                    status, final = "single-model", bo
                else:
                    status, final = ("excluded: models disagree" if bo or bw or be else "excluded: no model found it"), None
                if final:
                    area = (final[2] - final[0]) * (final[3] - final[1]) / (W * H)
                    if area < MIN_AREA:
                        status, final = "excluded: object too small", None
                rows.append({"path": meta["path"], "label": oid, "split": meta["split"], "width": W, "height": H,
                             "owl_conf": round(co, 3), "world_conf": round(cw, 3), "yoloe_conf": round(ce, 3),
                             "iou_owl_world": round(iou_w, 3), "iou_owl_yoloe": round(iou_e, 3), "iou_world_yoloe": round(iou_we, 3),
                             "owl_box": bo and [round(v, 1) for v in bo], "world_box": bw and [round(v, 1) for v in bw],
                             "yoloe_box": be and [round(v, 1) for v in be],
                             "x1": final and round(final[0], 1), "y1": final and round(final[1], 1),
                             "x2": final and round(final[2], 1), "y2": final and round(final[3], 1), "status": status})
        kept = sum(not r["status"].startswith("excluded") for r in rows if r["label"] == oid)
        print(f"{oid} {names[oid]:<24s} kept {kept:3d}/{len(group)}  prompts={prompts}", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("\n", df["status"].value_counts().to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
