import argparse
import os
from preprocess import preprocess_images
from detector import run_detection
from evaluator import evaluate_images


def main():
    p = argparse.ArgumentParser(description='Image object detection pipeline')
    p.add_argument('input', nargs='?', default='Images', help='Input image file or folder (default: Images)')
    p.add_argument('--out', '-o', default='results', help='Output folder for processed and annotated images')
    p.add_argument('--tmp', default='.processed', help='Temporary folder for preprocessed images')
    p.add_argument('--model', default='yolov8n.pt', help='Pretrained model name or path (default: yolov8n.pt)')
    p.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    p.add_argument('--labels', default='labels', help='Root folder where original labels are stored')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    print(f'Preprocessing images from "{args.input}"...')
    processed = preprocess_images(args.input, args.tmp)
    if not processed:
        print('No valid images found after preprocessing.')
        return

    print('Running detection...')
    # save outputs in a folder named after the model (e.g. results/yolov8n.pt_result)
    model_folder = os.path.basename(args.model)
    ann_dir = os.path.join(args.out, f"{model_folder}_result")
    results = run_detection(processed, ann_dir, model_name=args.model, conf=args.conf)

    print(f"Saved {len(results)} annotated images to {ann_dir}")

    # evaluate predictions against provided original labels
    pred_labels_dir = os.path.join(ann_dir, 'labels')
    report_path = os.path.join(args.out, f'eval_{os.path.splitext(os.path.basename(args.model))[0]}.json')
    print('Evaluating predictions...')
    report = evaluate_images(args.labels, pred_labels_dir, processed, report_path)
    print(f"Wrote evaluation report to {report_path}")

if __name__ == '__main__':
    main()
