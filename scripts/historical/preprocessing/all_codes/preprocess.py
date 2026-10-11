import cv2  #readning/writng an img & video,image transformation(resizng,rotatng,croppng),filterng & enhcng img,feature/objct dtctn
import numpy as np  #multi dimensional array data strctr
import os  #create new directory,list the contents of a directory, delete file/directory,rename a file/dir.
from tqdm import tqdm  #progress bars to loops & iterables, tqdm is derived from arabic TAQADAM meaning progress

def preprocess_image(image_path):
    # 1. Load image
    img = cv2.imread(image_path)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Resize to 256x256
    gray = cv2.resize(gray, (256, 256))

    # 4. Normalize (0 to 1)
    gray = gray / 255.0

     # 5. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    enhanced = clahe.apply((gray * 255).astype(np.uint8))

    # 6. Noise detection using Laplacian variance
    lap_var = cv2.Laplacian(enhanced, cv2.CV_64F).var()
    if lap_var < 60:
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)

    # 7. Otsu’s Thresholding
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 8. Morphological operations (remove small noises)
    kernel = np.ones((3,3), np.uint8)  # to create new array of a given shape and data type(unsigned integer-non negative values)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 9. ROI Extraction and Contour-Based Cropping
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)  # Bounding box
        roi_cropped = enhanced[y:y+h, x:x+w]  # Crop the brain
        roi_cropped = cv2.resize(roi_cropped, (256, 256))  # Resize back to 256x256
        roi = roi_cropped
    else:
        roi = cv2.resize(enhanced, (256, 256))  # fallback if no contour found

    return roi

def preprocess_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for subdir in os.listdir(input_folder):
        sub_input = os.path.join(input_folder, subdir)
        sub_output = os.path.join(output_folder, subdir)
        os.makedirs(sub_output, exist_ok=True) # makedirs(path)-create dir recursively
        for filename in tqdm(os.listdir(sub_input), desc=f"Processing {subdir}"):
            img_path = os.path.join(sub_input, filename)
            processed = preprocess_image(img_path)
            cv2.imwrite(os.path.join(sub_output, filename), processed)

# Example usage
train_input = r"C:\user\juli\thesis\original_file\othr folders\Dataset_B_clean\Dataset_B\train"
test_input  = r"C:\user\juli\thesis\original_file\othr folders\Dataset_B_clean\Dataset_B\test"
train_output = r"C:\user\juli\thesis\original_file\othr folders\Dataset_B_clean\Dataset_B\preprocessed_train_B"
test_output  = r"C:\user\juli\thesis\original_file\othr folders\Dataset_B_clean\Dataset_B\preprocessed_test_B"

preprocess_folder(train_input, train_output)
preprocess_folder(test_input, test_output)
