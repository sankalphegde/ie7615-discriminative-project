# Milestone 2: Multi-Object Detection with YOLOv8

Given an image that contains several of the class's objects, the model reports the **Object ID and the location
(bounding box) of every object**. Multi-object images are grids of 224×224 photos from the final merged class dataset;
bounding boxes were created with open-vocabulary detectors and cross-checked between models; YOLOv8 was fine-tuned
from COCO weights (transfer learning).

📄 Report: [`report/Milestone2_Report.pdf`](report/Milestone2_Report.pdf)

**No image data is stored in this repository** (per the course instructions). The scripts below regenerate every
label and grid image from the course dataset.

## Results (test set: 150 grid images, 2,220 objects)

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params | Inference (ms/img) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.877 | 0.779 | 0.872 | 0.832 | 3.1M | 3.9 |

Test mAP@0.5 by grid size (YOLOv8n): 2×2 0.845 · 3×3 0.908 · 4×4 0.854 · 5×4 0.885 · 5×5 0.904.
Inference time measured on an Apple M4 GPU at 1120-pixel input. Full metrics, per-class AP and the most frequent
confusions are in [`results/`](results/).

## Pipeline

**Object IDs used:** all 72 final Object IDs except **OBJ054** (excluded per the course announcement: its photos do not
show the object). Duplicate photos are removed before the split, background-only `*_bg_*` photos are not used, and the
odd-sized OBJ003 photo is resized to 224×224. The Milestone 1 split ([`split.csv`](split.csv), stratified 70/15/15,
seed 42) is reused, and every grid image is built from photos of a single split only, so **no source photo appears in
more than one of train / val / test**.

### 1. Bounding-box labels: [`label_objects.py`](label_objects.py)
Each photo is labelled by three open-vocabulary detectors prompted with the object's name (plus a clearer everyday
phrase where the sheet name is unusual), at the photos' native 224×224 size:
OWLv2 (`google/owlv2-base-patch16-ensemble`, primary), YOLO-World (`yolov8s-worldv2`) and YOLOE (`yoloe-11s-seg`).

| Rule | Box used | Photos |
|---|---|---:|
| **Verified**: OWLv2 agrees with YOLO-World or YOLOE (IoU ≥ 0.5) | OWLv2 | 3,947 |
| **Two-model**: OWLv2 found nothing, YOLO-World and YOLOE agree | their average | 93 |
| **Single-model**: only OWLv2, confidence ≥ 0.35 (spot-checked) | OWLv2 | 1,250 |
| Left out: models disagree / nothing found / box < 2% of the photo | — | 1,569 |

5,290 of 6,859 photos (77%) were kept, and every one of the 72 objects keeps at least 13 photos.
All boxes, scores and statuses are in [`labeling/box_labels.csv`](labeling/box_labels.csv); per-object counts in
[`labeling/label_summary.csv`](labeling/label_summary.csv).

### 2. Label check: [`check_labels.py`](check_labels.py)
Draws a random sample of photos per label status with every model's box. Of 64 sampled kept boxes, 63 were correct
and 1 was partial (one shoe of a pair); none was on the wrong item ([`labeling/manual_check.json`](labeling/manual_check.json)).

### 3. Grid images: [`generate_grids.py`](generate_grids.py)
224×224 photos placed side by side, every cell a different object, in equal shares of the five course grid sizes:

| Grid | Objects | Image size | Train | Val | Test |
|---|---:|---|---:|---:|---:|
| 2×2 | 4 | 448 × 448 | 200 | 30 | 30 |
| 3×3 | 9 | 672 × 672 | 200 | 30 | 30 |
| 4×4 | 16 | 896 × 896 | 200 | 30 | 30 |
| 5×4 | 20 | 1120 × 896 | 200 | 30 | 30 |
| 5×5 | 25 | 1120 × 1120 | 200 | 30 | 30 |

1,300 images with 19,240 labelled objects. Per-cell augmentation: horizontal flip (box mirrored) and brightness /
contrast / saturation jitter. YOLO class *k* maps to the *k*-th Object ID in `data.yaml`, whose class names are the
Object IDs themselves.

### 4. Training and evaluation: [`train_yolo.py`](train_yolo.py), [`evaluate_yolo.py`](evaluate_yolo.py)
COCO-pretrained YOLOv8, `imgsz=1120` (the largest grid), batch 8, early stopping (patience 15), checkpoints every
5 epochs. Evaluation reports val/test metrics, test metrics per grid size, per-class AP and confusions.

### 5. Detecting IDs and locations: [`detect_objects.py`](detect_objects.py)
```bash
python detect_objects.py path/to/grid_image.jpg
```
```
test_00001.jpg: 10 object(s)
  OBJ038  Perfume bottle         conf 1.00   box x1=309 y1=41 x2=360 y2=164
  OBJ032  Sound Card             conf 0.98   box x1=525 y1=102 x2=625 y2=154
  OBJ019  Plant                  conf 0.98   box x1=224 y1=225 x2=447 y2=384
  ...
```
It prints the Object ID, name, confidence and box (pixels) of every object and saves an annotated copy to
`detections/`. Uses [`weights/grid_yolov8n_best.pt`](weights/) by default.

## How to run

```bash
pip install -r requirements.txt
DATA="/path/to/IE7615 Deep Learning for AI"          # folder with images_OBJ001 … images_OBJ073

python label_objects.py   --data "$DATA" --split split.csv           # -> labeling/box_labels.csv
python check_labels.py    --data "$DATA"                             # -> labeling/check_*.jpg
python generate_grids.py  --data "$DATA" --labels labeling/box_labels.csv --out yolo_grids
python train_yolo.py      --model yolov8n.pt --name grid_yolov8n --epochs 50     # --device 0 on CUDA / Colab
python train_yolo.py      --model yolov8s.pt --name grid_yolov8s --epochs 40
python detect_objects.py  yolo_grids/images/test/test_00001.jpg
python make_report_m2.py                                             # -> report/Milestone2_Report.pdf
```

## Limitations

- 23% of photos were left out because the detectors could not agree on a box; these are mostly small, thin or cluttered
  objects, so the test set is somewhat easier than the full dataset.
- 1,250 kept boxes come from OWLv2 alone; they passed a visual spot-check but were not confirmed by a second model.
- Grid images never contain overlapping objects, unlike real scenes with several objects photographed together.
