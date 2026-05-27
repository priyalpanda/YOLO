import argparse
import os
from preprocess import preprocess_images
from detector import run_detection


def main():
    p = argparse.ArgumentParser(description='Image object detection pipeline')
    p.add_argument('input', help='Input image file or folder')
    p.add_argument('--out', '-o', default='outputs', help='Output folder for processed and annotated images')
    p.add_argument('--tmp', default='.processed', help='Temporary folder for preprocessed images')
    p.add_argument('--model', default='yolov8n.pt', help='Pretrained model name or path (default: yolov8n.pt)')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    p.add_argument('--no-match', dest='match', action='store_false', help='Do not match resolution across folder')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    print('Preprocessing images...')
    processed = preprocess_images(args.input, args.tmp, match_resolution=args.match)
    if not processed:
        print('No valid images found after preprocessing.')
        return

    print('Running detection...')
    ann_dir = os.path.join(args.out, 'annotated')
    results = run_detection(processed, ann_dir, model_name=args.model, conf=args.conf)

    print(f"Saved {len(results)} annotated images to {ann_dir}")

if __name__ == '__main__':
    main()
