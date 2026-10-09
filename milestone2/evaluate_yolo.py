"""Evaluate a trained YOLOv8 run on the val and test splits and save final_metrics.json.

Usage:  python evaluate_yolo.py runs/yolov8n
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO


def evaluate(run_dir, data="yolo_data/data.yaml", imgsz=640, batch=16, device=None):
    run_dir = Path(run_dir).resolve()
    device = device or ("mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu"))
    best = YOLO(run_dir / "weights" / "best.pt")
    out = {"model": run_dir.name, "params": sum(p.numel() for p in best.model.parameters())}
    for split in ["val", "test"]:
        m = best.val(data=data, split=split, imgsz=imgsz, batch=batch, device=device,
                     project=str(run_dir.parent), name=f"{run_dir.name}_eval_{split}", exist_ok=True, plots=True)
        out[split] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                      "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                      "per_class_mAP50": {m.names[int(c)]: float(m.box.ap50[i]) for i, c in enumerate(m.box.ap_class_index)},
                      "per_class_mAP50_95": {m.names[int(c)]: float(m.box.ap[i]) for i, c in enumerate(m.box.ap_class_index)},
                      "speed_ms": m.speed}
    (run_dir / "final_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if not kk.startswith("per_class")} if isinstance(v, dict) else v)
                      for k, v in out.items()}, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--data", default="yolo_data/data.yaml")
    a = ap.parse_args()
    evaluate(a.run_dir, a.data)
