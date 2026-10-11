"""Evaluate a trained YOLOv8 run on the val and test splits and save final_metrics.json.

Also reports test metrics per grid size (2x2 ... 5x5) when the dataset has a manifest.csv, and the
most frequent class confusions from the test confusion matrix.

Usage:  python evaluate_yolo.py runs/grid_yolov8s --data yolo_grids/data.yaml --imgsz 1120
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


def summarise(m):
    return {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
            "mAP50_95": float(m.box.map), "speed_ms": m.speed}


def evaluate(run_dir, data="yolo_grids/data.yaml", imgsz=1120, batch=8, device=None):
    run_dir = Path(run_dir).resolve()
    device = device or ("mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu"))
    best = YOLO(run_dir / "weights" / "best.pt")
    out = {"model": run_dir.name, "params": sum(p.numel() for p in best.model.parameters()), "imgsz": imgsz}
    kw = dict(imgsz=imgsz, batch=batch, device=device, project=str(run_dir.parent), exist_ok=True, verbose=False)

    for split in ["val", "test"]:
        m = best.val(data=data, split=split, name=f"{run_dir.name}_eval_{split}", plots=True, **kw)
        out[split] = summarise(m) | {
            "per_class_mAP50": {m.names[int(c)]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)},
            "per_class_mAP50_95": {m.names[int(c)]: float(m.box.ap[i]) for i, c in enumerate(m.box.ap_class_index)}}
        if split == "test":
            cm = m.confusion_matrix.matrix                     # rows = predicted, cols = true; last index = background
            n = len(m.names)
            pairs = [(m.names[t], m.names[p], int(cm[p, t])) for t in range(n) for p in range(n) if p != t and cm[p, t] > 0]
            out["test"]["confusions"] = sorted(pairs, key=lambda x: -x[2])[:15]
            out["test"]["missed_objects"] = int(cm[n, :n].sum())    # true objects predicted as background
            out["test"]["false_detections"] = int(cm[:n, n].sum())  # background predicted as an object

    cfg = yaml.safe_load(open(data))                          # test metrics per grid size
    root = Path(cfg["path"])
    if (root / "manifest.csv").exists():
        man = pd.read_csv(root / "manifest.csv")
        test = man[man["split"] == "test"].drop_duplicates("image")
        out["test_by_grid"] = {}
        for grid, g in test.groupby("grid", sort=False):
            lst = root / f"test_{grid}.txt"
            lst.write_text("\n".join(str(root / "images" / "test" / f"{im}.jpg") for im in g["image"]) + "\n")
            sub_yaml = root / f"data_test_{grid}.yaml"
            sub_yaml.write_text(yaml.safe_dump({**cfg, "test": str(lst)}))
            m = best.val(data=str(sub_yaml), split="test", name=f"{run_dir.name}_eval_test_{grid}", plots=False, **kw)
            out["test_by_grid"][grid] = summarise(m) | {"images": len(g)}

    (run_dir / "final_metrics.json").write_text(json.dumps(out, indent=2))
    show = {k: ({kk: vv for kk, vv in v.items() if not kk.startswith("per_class") and kk != "confusions"}
                if isinstance(v, dict) else v) for k, v in out.items()}
    print(json.dumps(show, indent=2, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default="yolo_grids/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1120)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    evaluate(a.run_dir, a.data, a.imgsz, a.batch)
