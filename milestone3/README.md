# Milestone 3: Live demo

Two scripts, one per demo task:

| Task | Model | Command |
|---|---|---|
| **(a)** single-object image → Object ID | ConvNeXt-Tiny, fine-tuned (Milestone 1, 98.1% test accuracy) | `python identify_object.py photo.jpg` |
| **(b)** multi-object image → ID + location of every object | YOLOv8s (Milestone 2, 0.893 mAP@0.5) | `python ../milestone2/detect_objects.py grid.jpg` |

## Run locally

```bash
git clone https://github.com/sankalphegde/ie7615-discriminative-project.git
cd ie7615-discriminative-project/milestone3
pip install ultralytics            # also installs torch, torchvision and pillow

python identify_object.py path/to/photo.jpg                       # (a)
python ../milestone2/detect_objects.py path/to/grid.jpg --out shown   # (b), annotated image saved to shown/
```

The Milestone 1 weights (106 MB, larger than GitHub's file limit) are downloaded automatically from the
[`demo-models` release](https://github.com/sankalphegde/ie7615-discriminative-project/releases/tag/demo-models) on the first run.
The YOLOv8s weights are in [`../milestone2/weights/`](../milestone2/weights/).

## Run in Google Colab (no setup)

Open [`demo_colab.ipynb`](demo_colab.ipynb) in Colab, run the first cell, then upload a photo in cell (a) and a
multi-object image in cell (b).

Use photos from the course dataset (test split) or any new photo of a class object. No image data is stored in this repository.
