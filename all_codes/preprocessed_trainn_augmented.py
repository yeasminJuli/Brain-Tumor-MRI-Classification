import cv2
import numpy as np
import os
from tqdm import tqdm
import random

# ---------------- Professional Augmentation Function ----------------
def augment_image(image):
    rows, cols = image.shape

    # 1️⃣ Horizontal Flip (50% chance)
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # 2️⃣ Random Rotation (-15° to +15°)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 3️⃣ Random Scaling (0.9 to 1.1)
    scale = random.uniform(0.9, 1.1)
    M_scale = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, scale)
    image = cv2.warpAffine(image, M_scale, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 4️⃣ Random Shift / Translation (-10 to +10 pixels)
    tx = random.randint(-10, 10)
    ty = random.randint(-10, 10)
    M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_shift, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # 5️⃣ Brightness & Contrast
    alpha = random.uniform(0.9, 1.1)  # contrast
    beta = random.randint(-15, 15)    # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 6️⃣ Light Gaussian Noise (50% chance)
    if random.random() < 0.5:
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image

# ---------------- Folder Augmentation (3–5 augmentations per image) ----------------
def augment_folder(input_folder, output_folder, num_aug=3):
    os.makedirs(output_folder, exist_ok=True)

    for subdir in os.listdir(input_folder):
        sub_input = os.path.join(input_folder, subdir)
        sub_output = os.path.join(output_folder, subdir)
        os.makedirs(sub_output, exist_ok=True)

        for filename in tqdm(os.listdir(sub_input), desc=f"Augmenting {subdir}"):
            img_path = os.path.join(sub_input, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Save original image
            cv2.imwrite(os.path.join(sub_output, filename), image)

            # Save multiple augmentations
            name, ext = os.path.splitext(filename)
            for i in range(1, num_aug+1):
                aug_img = augment_image(image)
                cv2.imwrite(os.path.join(sub_output, f"{name}_aug{i}{ext}"), aug_img)

# ---------------- Count Images ----------------
def count_augmented_images(original_folder, augmented_folder):
    print(f"{'Class':15} {'Original':10} {'Augmented':10} {'Total':10}")
    print("-"*50)
    total_original = 0
    total_augmented = 0
    total_all = 0

    for subdir in os.listdir(original_folder):
        orig_path = os.path.join(original_folder, subdir)
        aug_path = os.path.join(augmented_folder, subdir)

        orig_count = len(os.listdir(orig_path))
        aug_count = len(os.listdir(aug_path)) - orig_count
        total = orig_count + aug_count

        print(f"{subdir:15} {orig_count:<10} {aug_count:<10} {total:<10}")

        total_original += orig_count
        total_augmented += aug_count
        total_all += total

    print("-"*50)
    print(f"{'Total':15} {total_original:<10} {total_augmented:<10} {total_all:<10}")

# ---------------- Paths ----------------
preprocessed_train = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_train_A"
train_augmented   = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_train_A_augmented"

# ---------------- Run ----------------
augment_folder(preprocessed_train, train_augmented, num_aug=4)  # generates 4 variations per original
count_augmented_images(preprocessed_train, train_augmented)
