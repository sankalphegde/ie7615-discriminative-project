"""Live demo (a): given an image of a single object, print its Object ID.

Uses the fine-tuned ConvNeXt-Tiny classifier from Milestone 1 (98.1% test accuracy, 73 classes).
The weights (106 MB) are downloaded from the GitHub release on first run.

Usage:
    python identify_object.py photo.jpg [more photos or a folder] [--top 3]
"""
import argparse
import csv
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
WEIGHTS = HERE / "weights" / "convnext_tiny_ft.pt"
WEIGHTS_URL = ("https://github.com/sankalphegde/ie7615-discriminative-project/releases/download/"
               "demo-models/convnext_tiny_ft.pt")
CLASSES = [f"OBJ{i:03d}" for i in range(1, 74)]          # the 73 Object IDs, in training order
NAMES = {r["OBJECT_ID"]: r["OBJECT_NAME"] for r in csv.DictReader(open(REPO / "data" / "object_ids.csv"))}
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    if not WEIGHTS.exists():
        WEIGHTS.parent.mkdir(exist_ok=True)
        print("Downloading the Milestone 1 model (106 MB, first run only) ...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.to(DEVICE).eval()


@torch.no_grad()
def identify(model, path, top=3):
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    probs = torch.softmax(model(PREPROCESS(img).unsqueeze(0).to(DEVICE)), 1)[0].cpu()
    best = probs.topk(top)
    return [(CLASSES[i], NAMES.get(CLASSES[i], ""), float(p)) for p, i in zip(best.values, best.indices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    model = load_model()
    paths = []
    for p in map(Path, args.images):
        paths += sorted(q for q in p.iterdir() if q.suffix.lower() in (".jpg", ".jpeg", ".png")) if p.is_dir() else [p]
    for p in paths:
        preds = identify(model, p, args.top)
        oid, name, conf = preds[0]
        print(f"\n{p.name}:  {oid}  {name}  ({conf:.1%})")
        for o, n, c in preds[1:]:
            print(f"    runner-up: {o}  {n}  ({c:.1%})")


if __name__ == "__main__":
    main()
