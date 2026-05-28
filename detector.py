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


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area1 + area2 - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def non_max_suppression(boxes, confs, classes, iou_thresh=0.5):
    keep = []
    idxs = np.argsort(confs)[::-1]
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = compute_iou(boxes[current], boxes[rest])
        same_class = classes[rest] == classes[current]
        suppressed = (ious > iou_thresh) & same_class
        idxs = rest[~suppressed]
    return boxes[keep], classes[keep], confs[keep]


def run_detection(image_paths, output_dir, model_name='yolov8n.pt', conf=0.4, imgsz=640):
    os.makedirs(output_dir, exist_ok=True)
    pred_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(pred_labels_dir, exist_ok=True)
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

                min_conf = max(conf, 0.4)
                keep_mask = confs >= min_conf
                boxes = boxes[keep_mask]
                classes = classes[keep_mask]
                confs = confs[keep_mask]

                if len(boxes) > 0:
                    boxes, classes, confs = non_max_suppression(boxes, confs, classes, iou_thresh=0.5)
                    if len(boxes) > 0:
                        img = draw_boxes_and_labels(img, boxes, classes, confs, names)
                        # save predicted labels in YOLO normalized format (class x_center y_center w h [conf])
                        h, w = img.shape[:2]
                        pred_label_lines = []
                        for (x1, y1, x2, y2), cls, c in zip(boxes, classes, confs):
                            x_center = (x1 + x2) / 2.0 / w
                            y_center = (y1 + y2) / 2.0 / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            pred_label_lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f} {float(c):.6f}\n")
                        pred_label_path = os.path.join(pred_labels_dir, os.path.splitext(os.path.basename(p))[0] + '.txt')
                        try:
                            with open(pred_label_path, 'w') as f:
                                f.writelines(pred_label_lines)
                        except Exception:
                            pass
            out_path = os.path.join(output_dir, os.path.basename(p))
            # write with support for unicode paths
            ext = os.path.splitext(out_path)[1].lower()
            cv2.imencode(ext, img)[1].tofile(out_path)
            results_saved.append(out_path)
        except Exception as e:
            # skip problematic images
            continue
    return results_saved
