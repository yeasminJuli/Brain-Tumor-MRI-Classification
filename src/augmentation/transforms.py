import cv2
import numpy as np
import random

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
