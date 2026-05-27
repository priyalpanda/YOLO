import os
import cv2
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO


def draw_boxes_and_labels(img, boxes, classes, confs, names):
    for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(int(cls), str(cls))}: {conf:.2f}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def run_detection(image_paths, output_dir, model_name='yolov8n.pt', conf=0.25, imgsz=640):
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_name)
    names = model.names if hasattr(model, 'names') else {}

    results_saved = []
    for p in tqdm(image_paths, desc='Detecting'):
        try:
            res = model.predict(source=p, imgsz=imgsz, conf=conf, verbose=False)
            if not res:
                continue
            r = res[0]
            # read original image
            img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                # extract arrays
                try:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                except Exception:
                    boxes = r.boxes.xyxy.numpy()
                    classes = r.boxes.cls.numpy()
                    confs = r.boxes.conf.numpy()
                img = draw_boxes_and_labels(img, boxes, classes, confs, names)
            out_path = os.path.join(output_dir, os.path.basename(p))
            # write with support for unicode paths
            ext = os.path.splitext(out_path)[1].lower()
            cv2.imencode(ext, img)[1].tofile(out_path)
            results_saved.append(out_path)
        except Exception as e:
            # skip problematic images
            continue
    return results_saved
