import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_image(image_path):
    # 1️⃣ Load image in grayscale directly
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 2️⃣ Resize to 256x256
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # 3️⃣ Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # 4️⃣ Remove salt-and-pepper noise using median blur
    img = cv2.medianBlur(img, 3)

    # 5️⃣ Normalize to 0-1 for CNN
    img = img / 255.0

    # 6️⃣ Optional: ROI extraction based on simple threshold mask
    #    This keeps grayscale info while cropping
    _, mask = cv2.threshold((img*255).astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_AREA)
    else:
        roi = img

    return roi

def preprocess_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for subdir in os.listdir(input_folder):
        sub_input = os.path.join(input_folder, subdir)
        sub_output = os.path.join(output_folder, subdir)
        os.makedirs(sub_output, exist_ok=True)
        for filename in tqdm(os.listdir(sub_input), desc=f"Processing {subdir}"):
            img_path = os.path.join(sub_input, filename)
            processed = preprocess_image(img_path)
            if processed is not None:
                # Save as uint8 image
                cv2.imwrite(os.path.join(sub_output, filename), (processed*255).astype(np.uint8))

# Example usage
train_input = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\Data\Train"
test_input  = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\Data\Test"
train_output = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\preprocessed_train"
test_output  = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\preprocessed_test"

preprocess_folder(train_input, train_output)
preprocess_folder(test_input, test_output)
