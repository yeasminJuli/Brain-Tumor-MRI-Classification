"""Preprocess or augment a class-folder dataset without changing class names."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.preprocessing.pipeline import preprocess_image
from src.augmentation.transforms import augment_image

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=['preprocess','augment'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--copies', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.output.exists(): p.error('Output already exists; choose a new folder.')
    if args.input.resolve() == args.output.resolve() or args.input.resolve() in args.output.resolve().parents:
        p.error('Output must be outside input.')
    if args.copies < 0: p.error('--copies must be nonnegative')
    random.seed(args.seed); np.random.seed(args.seed)
    files = sorted(f for f in args.input.glob('*/*') if f.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'})
    if not files: p.error('No images found in class folders.')
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f'Cannot read {path}')
        if args.operation == 'preprocess': image = preprocess_image(str(path))
        target = args.output / path.parent.name / (path.name + '.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), image): raise IOError(target)
        if args.operation == 'augment':
            for i in range(args.copies):
                out = target.with_name(path.name + f'_aug{i+1}.png')
                if not cv2.imwrite(str(out), augment_image(image)): raise IOError(out)
    print(f'Processed {len(files)} original images. Output: {args.output}')

if __name__ == '__main__': main()
