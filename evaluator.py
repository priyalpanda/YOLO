import os
import cv2
import json
import numpy as np
from collections import defaultdict


def load_yolo_labels(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # support both 5-col (class x y w h) and 6-col (class x y w h conf)
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                conf = float(parts[5]) if len(parts) > 5 else None
                boxes.append({'class': cls, 'x': x, 'y': y, 'w': w, 'h': h, 'conf': conf})
    return boxes


def yolo_to_xyxy(box, img_w, img_h):
    xc, yc, bw, bh = box['x'], box['y'], box['w'], box['h']
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return [x1, y1, x2, y2]


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def find_label_file(labels_root, image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # search recursively for stem.txt
    for root, _, files in os.walk(labels_root):
        fname = stem + '.txt'
        if fname in files:
            return os.path.join(root, fname)
    return None


def evaluate_images(original_labels_root, predicted_labels_root, image_paths, out_report_path, iou_thresh=0.5):
    summary = defaultdict(int)
    per_image = {}

    for img_path in image_paths:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_file = find_label_file(original_labels_root, img_path)
        pred_file = os.path.join(predicted_labels_root, os.path.splitext(os.path.basename(img_path))[0] + '.txt')

        gt_boxes = load_yolo_labels(gt_file) if gt_file else []
        pred_boxes = load_yolo_labels(pred_file)

        gt_xy = [yolo_to_xyxy(b, w, h) for b in gt_boxes]
        pred_xy = [yolo_to_xyxy(b, w, h) for b in pred_boxes]

        matched_pred = set()
        image_issues = defaultdict(int)

        if gt_boxes:
            for gi, g in enumerate(gt_boxes):
                best_iou = 0.0
                best_p = -1
                for pi, p in enumerate(pred_boxes):
                    i = iou_xyxy(gt_xy[gi], pred_xy[pi])
                    if i > best_iou:
                        best_iou = i
                        best_p = pi

                if best_p == -1 or best_iou < 0.5:
                    # no good match
                    if best_iou >= 0.3:
                        image_issues['poor_localization'] += 1
                        summary['poor_localization'] += 1
                    else:
                        image_issues['false_negative'] += 1
                        summary['false_negative'] += 1
                else:
                    # matched
                    matched_pred.add(best_p)
                    p = pred_boxes[best_p]
                    # class check
                    if int(p['class']) != int(g['class']):
                        image_issues['misclassification'] += 1
                        summary['misclassification'] += 1
                    # check size ratios
                    area_gt = (gt_xy[gi][2] - gt_xy[gi][0]) * (gt_xy[gi][3] - gt_xy[gi][1])
                    area_p = (pred_xy[best_p][2] - pred_xy[best_p][0]) * (pred_xy[best_p][3] - pred_xy[best_p][1])
                    if area_p > 2.0 * area_gt:
                        image_issues['oversized_box'] += 1
                        summary['oversized_box'] += 1
                    if area_p < 0.5 * area_gt:
                        image_issues['underlocalized'] += 1
                        summary['underlocalized'] += 1

        # false positives = preds not matched
        for pi, p in enumerate(pred_boxes):
            if pi not in matched_pred:
                image_issues['false_positive'] += 1
                summary['false_positive'] += 1

        per_image[os.path.basename(img_path)] = dict(image_issues)

    report = {
        'summary': dict(summary),
        'per_image': per_image,
    }

    # write JSON report
    try:
        with open(out_report_path, 'w') as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    return report
