import os
import csv
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.models.proposed import HybridCNNBiLSTMAttention
INPUT_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"C:\user\juli\thesis\original_file\othr folders\preprocessed_trainn"

SAVE_DIR = r"C:\user\juli\thesis\original_file\othr folders\gradcam_trainn_corrected"

MODEL_PATH = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\training_outputs_proposed_model_orgnl\training_outputs"
    r"\hybrid_cnn_bilstm_attention_final.pth"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# EXPECTED CLASS ORDER
# IMPORTANT:
# torchvision ImageFolder assigns classes alphabetically.
# ============================================================

EXPECTED_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor"
}

# ============================================================
# TRANSFORM - SAME AS MODEL INPUT
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

# ============================================================
# LOAD DATASET FIRST AND VERIFY CLASS MAPPING
# ============================================================

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("\nDataset class mapping:")
print(dataset.class_to_idx)

if dataset.classes != EXPECTED_CLASSES:
    raise RuntimeError(
        f"\nCLASS ORDER ERROR!\n"
        f"Dataset classes: {dataset.classes}\n"
        f"Expected:        {EXPECTED_CLASSES}\n"
        f"Do NOT generate Grad-CAM++ until this is resolved."
    )

print("\nVerified class mapping:")
for class_name, idx in dataset.class_to_idx.items():
    print(f"{idx} = {DISPLAY_NAMES[class_name]}")

# ============================================================
# LOAD MODEL
# ============================================================

model = HybridCNNBiLSTMAttention(num_classes=4).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

# If checkpoint contains class names, verify them too
if "class_names" in checkpoint:
    checkpoint_classes = list(checkpoint["class_names"])

    if checkpoint_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            f"\nCHECKPOINT CLASS ORDER ERROR!\n"
            f"Checkpoint: {checkpoint_classes}\n"
            f"Expected:   {EXPECTED_CLASSES}"
        )

model.eval()

# ============================================================
# GRAD-CAM++
# ============================================================

class GradCAMPlusPlus:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        # Better than deprecated register_backward_hook()
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):

        self.model.zero_grad(set_to_none=True)

        output, _ = self.model(input_tensor)

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        if gradients is None or activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture gradients/activations."
            )

        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)

        denominator = (
            2.0 * grad2
            + (activations * grad3).sum(
                dim=(2, 3),
                keepdim=True
            )
        )

        denominator = torch.where(
            torch.abs(denominator) > 1e-12,
            denominator,
            torch.ones_like(denominator)
        )

        alpha = grad2 / denominator

        weights = (
            alpha * F.relu(gradients)
        ).sum(dim=(2, 3), keepdim=True)

        cam = (
            weights * activations
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(256, 256),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        minimum = cam.min()
        maximum = cam.max()

        if maximum > minimum:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = np.zeros_like(cam)

        return cam


gradcam = GradCAMPlusPlus(
    model,
    model.cnn.block5.conv2
)

# ============================================================
# FONT
# ============================================================

def get_font(size=22):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


# ============================================================
# ADD LABEL PANEL
# ============================================================

def add_labels(
    image_rgb,
    true_name,
    predicted_name,
    confidence,
    correct
):

    h, w, _ = image_rgb.shape

    label_height = 90

    canvas = Image.new(
        "RGB",
        (w, h + label_height),
        "white"
    )

    overlay_pil = Image.fromarray(image_rgb)

    canvas.paste(
        overlay_pil,
        (0, label_height)
    )

    draw = ImageDraw.Draw(canvas)

    font = get_font(20)

    status = "CORRECT" if correct else "MISCLASSIFIED"

    line1 = (
        f"True: {DISPLAY_NAMES[true_name]}"
    )

    line2 = (
        f"Predicted: {DISPLAY_NAMES[predicted_name]} "
        f"({confidence * 100:.2f}%)"
    )

    line3 = f"Status: {status}"

    draw.text(
        (10, 5),
        line1,
        fill="black",
        font=font
    )

    draw.text(
        (10, 32),
        line2,
        fill="black",
        font=font
    )

    # Red status only for misclassification;
    # otherwise black.
    status_color = (
        (180, 0, 0)
        if not correct
        else (0, 0, 0)
    )

    draw.text(
        (10, 59),
        line3,
        fill=status_color,
        font=font
    )

    return np.array(canvas)


# ============================================================
# OUTPUT FOLDERS
# ============================================================

correct_dir = os.path.join(
    SAVE_DIR,
    "Correctly_Classified"
)

error_dir = os.path.join(
    SAVE_DIR,
    "Misclassified"
)

os.makedirs(correct_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

for class_name in EXPECTED_CLASSES:

    os.makedirs(
        os.path.join(correct_dir, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(error_dir, class_name),
        exist_ok=True
    )

# ============================================================
# PROCESS EVERY IMAGE
# ============================================================

records = []

for index in tqdm(
    range(len(dataset)),
    desc="Generating Grad-CAM++"
):

    img_path, true_idx = dataset.samples[index]

    true_class = dataset.classes[true_idx]

    # --------------------------------------------------------
    # Load original PREPROCESSED image
    # --------------------------------------------------------

    with Image.open(img_path) as pil_img:

        original_rgb = pil_img.convert("RGB")

        input_tensor = transform(
            original_rgb
        ).unsqueeze(0).to(DEVICE)

        # Use 256x256 display image so heatmap and MRI
        # have exactly the same dimensions
        display_img = original_rgb.resize(
            (256, 256)
        )

        display_np = np.asarray(
            display_img,
            dtype=np.uint8
        )

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        logits, _ = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_idx = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[predicted_idx].item()
        )

    predicted_class = dataset.classes[
        predicted_idx
    ]

    correct = predicted_idx == true_idx

    # --------------------------------------------------------
    # IMPORTANT:
    # Grad-CAM++ is generated for MODEL-PREDICTED CLASS
    # --------------------------------------------------------

    heatmap = gradcam.generate(
        input_tensor,
        class_idx=predicted_idx
    )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    heatmap = cv2.GaussianBlur(
        heatmap,
        (7, 7),
        0
    )

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0, 1) * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        display_np,
        0.70,
        heatmap_rgb,
        0.30,
        0
    )

    # --------------------------------------------------------
    # ADD TRUE + PREDICTED LABELS DIRECTLY TO PNG
    # --------------------------------------------------------

    labelled_overlay = add_labels(
        overlay,
        true_class,
        predicted_class,
        confidence,
        correct
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    img_name = os.path.basename(img_path)

    output_root = (
        correct_dir
        if correct
        else error_dir
    )

    output_class_dir = os.path.join(
        output_root,
        true_class
    )

    stem, _ = os.path.splitext(img_name)

    output_filename = (
        f"{stem}"
        f"__TRUE-{true_class}"
        f"__PRED-{predicted_class}.png"
    )

    save_path = os.path.join(
        output_class_dir,
        output_filename
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            labelled_overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    records.append({
        "filename": img_name,
        "image_path": img_path,
        "true_index": true_idx,
        "true_class": true_class,
        "predicted_index": predicted_idx,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "correct": correct,
        "gradcam_target_class": predicted_class,
        "output_file": save_path
    })


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    SAVE_DIR,
    "GradCAM_Prediction_Audit.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=records[0].keys()
    )

    writer.writeheader()
    writer.writerows(records)


# ============================================================
# SUMMARY
# ============================================================

total = len(records)

num_correct = sum(
    r["correct"]
    for r in records
)

num_errors = total - num_correct

print("\n========================================")
print("GRAD-CAM++ COMPLETE")
print("========================================")

print(f"Total images       : {total}")
print(f"Correct predictions: {num_correct}")
print(f"Misclassifications : {num_errors}")
print(f"Accuracy           : {100*num_correct/total:.4f}%")

print("\nClass mapping used:")
for i, name in enumerate(EXPECTED_CLASSES):
    print(f"{i}: {DISPLAY_NAMES[name]}")

print("\nMisclassified cases:")

for r in records:

    if not r["correct"]:

        print(
            f"{r['filename']} | "
            f"TRUE={r['true_class']} | "
            f"PREDICTED={r['predicted_class']} | "
            f"confidence={r['confidence']:.4f}"
        )

print("\nCSV audit saved to:")
print(csv_path)

print("\nImages saved to:")
print(SAVE_DIR)