"""Augment the full preprocessed training partition before the requested 80/20 file split."""
from pathlib import Path
import hashlib
import random
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "preprocessed_C"
SEED = 42
NUM_AUGMENTATIONS = 4
CLASSES = ("Glioma", "Meningioma", "Healthy", "Pituitary Macroadenoma")
MODEL_CLASS_INDEX = dict(zip(CLASSES, range(4)))

def augment_image(image):
    rows, cols = image.shape

    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    scale = random.uniform(0.9, 1.1)
    M_scale = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    image = cv2.warpAffine(image, M_scale, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    tx = random.randint(-10, 10)
    ty = random.randint(-10, 10)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_shift, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    alpha = random.uniform(0.9, 1.1)  # contrast
    beta = random.randint(-15, 15)    # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.random() < 0.5:
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    output = ROOT / "preprocessed_trainn_augmented_C"
    if output.exists():
        raise FileExistsError(output)
    total = 0
    for label in CLASSES:
        source = ROOT / "preprocessed_train_C" / label
        destination = output / label
        destination.mkdir(parents=True)
        paths = sorted(p for p in source.iterdir() if p.is_file())
        for path in tqdm(paths, desc=label):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(path)
            shutil.copy2(path, destination / path.name)
            for i in range(1, NUM_AUGMENTATIONS + 1):
                target = destination / f"{path.stem}_aug{i}{path.suffix}"
                if target.exists():
                    raise FileExistsError(target)
                if not cv2.imwrite(str(target), augment_image(image)):
                    raise IOError(target)
        for path in destination.iterdir():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            assert image is not None and image.shape == (256, 256), path
        assert len(list(destination.iterdir())) == len(paths) * 5
        total += len(paths) * 5
        print(f"{label}: {len(paths)} originals; {len(paths)*5} including augmentation", flush=True)
    assert total == 4545, total
    print(f"COMPLETE: {total} files. Split 80/20 in the training script after augmentation.")


if __name__ == "__main__":
    main()
