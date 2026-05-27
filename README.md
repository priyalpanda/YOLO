# Image Object Detection Tool

This tool preprocesses images (removes corrupt/blank images, matches resolution, and sharpens blurry images), then runs an object detection model (YOLO) to produce images annotated with bounding boxes and predicted labels.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run the tool on a folder or single image:

```bash
python app.py path/to/images --out results --model yolov8n.pt
```

Notes

- The default model is `yolov8n.pt` (Ultralytics). The first run will download the weights automatically.
- If images are blurry, the preprocessor applies an unsharp mask; for stronger restoration consider integrating a super-resolution model (Real-ESRGAN) later.
- Output annotated images are saved under the output folder in `annotated/`.

