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
# 1. PATHS
# ============================================================

# Use TEST dataset for classification error analysis
DATA_DIR = (
    r"C:\user\juli\thesis\original_file"
    r"\othr folders\preprocessed_testt"
)

SAVE_DIR = (
    r"C:\user\juli\thesis\original_file"
    r"\othr folders\gradcam_test_final"
)

MODEL_PATH = (
    r"C:\user\juli\thesis\original_file"
    r"\othr folders\training_outputs_proposed_model_orgnl"
    r"\training_outputs"
    r"\hybrid_cnn_bilstm_attention_final.pth"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 2. VERIFIED CLASS ORDER
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
    "pituitary": "Pituitary"
}


# ============================================================
# 3. MODEL INPUT TRANSFORM
# ============================================================

# IMPORTANT:
# This is the transform used for MODEL INPUT.
# The model still receives 256 x 256 images.
#
# This does NOT mean the final saved Grad-CAM++ image
# will be restricted to 256 x 256.

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
# 4. LOAD DATASET
# ============================================================

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

print("\n========================================")
print("DATASET CLASS VERIFICATION")
print("========================================")

print("\nDataset class mapping:")
print(dataset.class_to_idx)

if dataset.classes != EXPECTED_CLASSES:
    raise RuntimeError(
        "\nCLASS ORDER ERROR!\n"
        f"Dataset classes : {dataset.classes}\n"
        f"Expected classes: {EXPECTED_CLASSES}\n\n"
        "Grad-CAM++ generation stopped."
    )

print("\nVerified class mapping:")

for class_name, idx in dataset.class_to_idx.items():
    print(
        f"{idx} = {DISPLAY_NAMES[class_name]}"
    )

print(
    f"\nTotal test images: {len(dataset)}"
)


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

model = HybridCNNBiLSTMAttention(
    num_classes=4
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" not in checkpoint:
    raise RuntimeError(
        "Checkpoint does not contain 'model_state_dict'."
    )

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)


# ============================================================
# 6. CHECK CHECKPOINT CLASS ORDER IF AVAILABLE
# ============================================================

if "class_names" in checkpoint:

    checkpoint_classes = list(
        checkpoint["class_names"]
    )

    print("\nCheckpoint class names:")
    print(checkpoint_classes)

    if checkpoint_classes != EXPECTED_CLASSES:
        raise RuntimeError(
            "\nCHECKPOINT CLASS ORDER ERROR!\n"
            f"Checkpoint classes: {checkpoint_classes}\n"
            f"Expected classes  : {EXPECTED_CLASSES}"
        )


# ============================================================
# 7. PREPARE MODEL FOR GRAD-CAM++
# ============================================================

# Entire model is placed into evaluation mode first.
model.eval()

# Grad-CAM++ needs backward gradients.
#
# cuDNN does not allow LSTM backward when the LSTM itself
# is in eval mode.
#
# Therefore ONLY torch.nn.LSTM modules are put into train mode.
#
# IMPORTANT:
# This DOES NOT retrain the model.
# No optimizer is used.
# optimizer.step() is never called.
# No model parameters are updated.

lstm_modules_found = 0

for module in model.modules():

    if isinstance(module, torch.nn.LSTM):
        module.train()
        lstm_modules_found += 1


print("\n========================================")
print("MODEL STATUS")
print("========================================")

print(f"Device: {DEVICE}")
print(
    f"LSTM modules found: {lstm_modules_found}"
)

if lstm_modules_found == 0:
    print(
        "WARNING: No torch.nn.LSTM module was detected."
    )

print(
    "Model prepared for Grad-CAM++."
)


# ============================================================
# 8. GRAD-CAM++ CLASS
# ============================================================

class GradCAMPlusPlus:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output.detach()

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = (
            grad_output[0].detach()
        )

    def generate(
        self,
        input_tensor,
        class_idx
    ):

        self.gradients = None
        self.activations = None

        self.model.zero_grad(
            set_to_none=True
        )

        # Forward pass WITH gradient tracking
        output, _ = self.model(
            input_tensor
        )

        # Explain the selected/predicted class
        score = output[
            0,
            class_idx
        ]

        # Backward pass
        score.backward()

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM++ gradients were not captured."
            )

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM++ activations were not captured."
            )

        gradients = self.gradients
        activations = self.activations

        # ----------------------------------------------------
        # GRAD-CAM++ CALCULATION
        # ----------------------------------------------------

        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)

        denominator = (
            2.0 * grad2
            +
            (
                activations * grad3
            ).sum(
                dim=(2, 3),
                keepdim=True
            )
        )

        # Numerical stability
        denominator = torch.where(
            torch.abs(denominator) > 1e-12,
            denominator,
            torch.ones_like(denominator)
        )

        alpha = (
            grad2 / denominator
        )

        positive_gradients = F.relu(
            gradients
        )

        weights = (
            alpha * positive_gradients
        ).sum(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        # IMPORTANT:
        # Keep CAM at native feature-map resolution.
        # We resize it later to the original image size.
        cam = (
            cam[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # NORMALIZE 0-1
        # ----------------------------------------------------

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        if cam_max > cam_min:

            cam = (
                (cam - cam_min)
                /
                (cam_max - cam_min)
            )

        else:

            cam = np.zeros_like(
                cam
            )

        return cam


# ============================================================
# 9. CREATE GRAD-CAM++ OBJECT
# ============================================================

TARGET_LAYER = (
    model.cnn.block5.conv2
)

gradcam = GradCAMPlusPlus(
    model,
    TARGET_LAYER
)

print("\nGrad-CAM++ target layer:")
print(TARGET_LAYER)


# ============================================================
# 10. FONT LOADING
# ============================================================

def load_font(
    size,
    bold=False
):

    if bold:

        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf",
            r"C:\Windows\Fonts\calibrib.ttf"
        ]

    else:

        candidates = [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf"
        ]

    for font_path in candidates:

        if os.path.exists(font_path):

            try:

                return ImageFont.truetype(
                    font_path,
                    size
                )

            except Exception:

                pass

    return ImageFont.load_default()


# ============================================================
# 11. CLEAN COMPACT LABEL ABOVE IMAGE
# ============================================================

def add_publication_label(
    image_rgb,
    true_name,
    predicted_name,
    correct
):

    h, w, _ = image_rgb.shape

    true_display = (
        DISPLAY_NAMES[true_name]
    )

    predicted_display = (
        DISPLAY_NAMES[predicted_name]
    )

    # --------------------------------------------------------
    # FONT
    # --------------------------------------------------------
    # Slightly larger than the previous tiny version,
    # but still compact enough for the figure.

    font_size = max(
        13,
        min(
            17,
            int(w * 0.026)
        )
    )

    normal_font = load_font(
        font_size,
        bold=False
    )

    bold_font = load_font(
        font_size,
        bold=True
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    # Confidence intentionally removed.
    line1 = (
        f"True: {true_display}"
        f"   |   "
        f"Predicted: {predicted_display}"
    )

    if correct:
        line2 = "CORRECT"
    else:
        line2 = "MISCLASSIFIED"

    # --------------------------------------------------------
    # MEASURE TEXT
    # --------------------------------------------------------

    temp_img = Image.new(
        "RGB",
        (10, 10),
        "white"
    )

    temp_draw = ImageDraw.Draw(
        temp_img
    )

    bbox1 = temp_draw.textbbox(
        (0, 0),
        line1,
        font=normal_font
    )

    bbox2 = temp_draw.textbbox(
        (0, 0),
        line2,
        font=bold_font
    )

    line1_width = (
        bbox1[2] - bbox1[0]
    )

    line1_height = (
        bbox1[3] - bbox1[1]
    )

    line2_width = (
        bbox2[2] - bbox2[0]
    )

    line2_height = (
        bbox2[3] - bbox2[1]
    )

    # --------------------------------------------------------
    # COMPACT WHITE HEADER
    # --------------------------------------------------------

    side_padding = 7
    top_padding = 3
    bottom_padding = 3
    line_gap = 2

    header_height = (
        top_padding
        + line1_height
        + line_gap
        + line2_height
        + bottom_padding
    )

    # Expand width ONLY if necessary to avoid clipping text.
    canvas_width = max(
        w,
        line1_width
        + (2 * side_padding)
    )

    # --------------------------------------------------------
    # CREATE CANVAS
    # --------------------------------------------------------

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            h + header_height
        ),
        (255, 255, 255)
    )

    # Center the MRI horizontally.
    image_x = (
        canvas_width - w
    ) // 2

    # IMPORTANT:
    # The Grad-CAM image is pasted unchanged.
    # No blue tint, black background, brightness adjustment,
    # contrast adjustment, or other recoloring is performed.
    canvas.paste(
        Image.fromarray(
            image_rgb
        ),
        (
            image_x,
            header_height
        )
    )

    draw = ImageDraw.Draw(
        canvas
    )

    # --------------------------------------------------------
    # FIRST LINE
    # --------------------------------------------------------

    line1_x = (
        canvas_width
        - line1_width
    ) // 2

    draw.text(
        (
            line1_x,
            top_padding
        ),
        line1,
        fill=(0, 0, 0),
        font=normal_font
    )

    # --------------------------------------------------------
    # SECOND LINE
    # --------------------------------------------------------

    line2_x = (
        canvas_width
        - line2_width
    ) // 2

    line2_y = (
        top_padding
        + line1_height
        + line_gap
    )

    if correct:

        status_color = (
            0,
            110,
            0
        )

    else:

        status_color = (
            190,
            0,
            0
        )

    draw.text(
        (
            line2_x,
            line2_y
        ),
        line2,
        fill=status_color,
        font=bold_font
    )

    return np.asarray(
        canvas
    )


# ============================================================
# 12. CREATE OUTPUT DIRECTORIES
# ============================================================

correct_dir = os.path.join(
    SAVE_DIR,
    "Correctly_Classified"
)

error_dir = os.path.join(
    SAVE_DIR,
    "Misclassified"
)

os.makedirs(
    correct_dir,
    exist_ok=True
)

os.makedirs(
    error_dir,
    exist_ok=True
)

for class_name in EXPECTED_CLASSES:

    os.makedirs(
        os.path.join(
            correct_dir,
            class_name
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            error_dir,
            class_name
        ),
        exist_ok=True
    )


# ============================================================
# 13. PROCESS ALL TEST IMAGES
# ============================================================

records = []

print("\n========================================")
print("GENERATING GRAD-CAM++")
print("========================================\n")


for index in tqdm(
    range(len(dataset)),
    desc="Generating Grad-CAM++"
):

    # --------------------------------------------------------
    # TRUE LABEL COMES FROM DATASET
    # --------------------------------------------------------

    img_path, true_idx = (
        dataset.samples[index]
    )

    true_idx = int(
        true_idx
    )

    true_class = (
        dataset.classes[
            true_idx
        ]
    )

    img_name = os.path.basename(
        img_path
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    try:

        with Image.open(
            img_path
        ) as pil_img:

            original_rgb = (
                pil_img
                .convert("RGB")
            )

            # -----------------------------------------------
            # MODEL INPUT
            # -----------------------------------------------
            # Model still receives 256 x 256.

            input_tensor = (
                transform(
                    original_rgb
                )
                .unsqueeze(0)
                .to(DEVICE)
            )

            # -----------------------------------------------
            # DISPLAY IMAGE
            # -----------------------------------------------
            # DO NOT resize this to 256 x 256.
            # Preserve the stored image's resolution.

            display_np = np.asarray(
                original_rgb,
                dtype=np.uint8
            ).copy()

            original_h, original_w = (
                display_np.shape[:2]
            )

    except Exception as e:

        print(
            "\nIMAGE LOAD ERROR:"
        )

        print(
            img_path
        )

        print(
            e
        )

        continue


    # ========================================================
    # 14. MODEL PREDICTION
    # ========================================================

    with torch.no_grad():

        logits, _ = model(
            input_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_idx = int(
            probabilities
            .argmax()
            .item()
        )

        confidence = float(
            probabilities[
                predicted_idx
            ].item()
        )


    # Predicted class comes from MODEL OUTPUT.
    predicted_class = (
        dataset.classes[
            predicted_idx
        ]
    )

    correct = (
        predicted_idx
        == true_idx
    )


    # ========================================================
    # 15. GENERATE GRAD-CAM++ FOR PREDICTED CLASS
    # ========================================================

    try:

        heatmap = gradcam.generate(
            input_tensor,
            class_idx=predicted_idx
        )

    except Exception as e:

        print(
            "\nGRAD-CAM++ ERROR"
        )

        print(
            f"Image: {img_path}"
        )

        print(
            f"True: {true_class}"
        )

        print(
            f"Predicted: {predicted_class}"
        )

        print(
            f"Error: {e}"
        )

        raise


    # ========================================================
    # 16. RESIZE ONLY THE HEATMAP
    # ========================================================

    # The underlying MRI remains at its stored resolution.
    # Only the Grad-CAM++ activation map is enlarged to match.

    heatmap = cv2.resize(
        heatmap,
        (
            original_w,
            original_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    # Mild heatmap smoothing
    heatmap = cv2.GaussianBlur(
        heatmap,
        (7, 7),
        0
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )


    # ========================================================
    # 17. COLORIZE GRAD-CAM++ ONLY
    # ========================================================

    # JET applies to the HEATMAP.
    # It does NOT recolor the underlying MRI itself.

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # 18. OVERLAY HEATMAP
    # ========================================================

    # Original MRI = 70%
    # Grad-CAM++ heatmap = 30%
    #
    # No additional blue tint or black background is added.

    overlay = cv2.addWeighted(
        display_np,
        0.70,
        heatmap_rgb,
        0.30,
        0
    )


    # ========================================================
    # 19. ADD LABEL ABOVE IMAGE
    # ========================================================

    labelled_overlay = (
        add_publication_label(
            overlay,
            true_class,
            predicted_class,
            correct
        )
    )


    # ========================================================
    # 20. CHOOSE OUTPUT DIRECTORY
    # ========================================================

    if correct:

        output_root = (
            correct_dir
        )

    else:

        output_root = (
            error_dir
        )

    # Subfolder corresponds to TRUE class.
    output_class_dir = os.path.join(
        output_root,
        true_class
    )


    # ========================================================
    # 21. OUTPUT FILENAME
    # ========================================================

    stem = os.path.splitext(
        img_name
    )[0]

    output_filename = (
        f"{stem}"
        f"__TRUE-{true_class}"
        f"__PRED-{predicted_class}"
        f".png"
    )

    save_path = os.path.join(
        output_class_dir,
        output_filename
    )


    # ========================================================
    # 22. SAVE LOSSLESS PNG
    # ========================================================

    success = cv2.imwrite(
        save_path,
        cv2.cvtColor(
            labelled_overlay,
            cv2.COLOR_RGB2BGR
        ),
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            1
        ]
    )

    if not success:

        raise RuntimeError(
            f"Failed to save image:\n"
            f"{save_path}"
        )


    # ========================================================
    # 23. RECORD AUDIT DATA
    # ========================================================

    records.append({

        "dataset_index":
            index,

        "filename":
            img_name,

        "image_path":
            img_path,

        "true_index":
            true_idx,

        "true_class":
            true_class,

        "predicted_index":
            predicted_idx,

        "predicted_class":
            predicted_class,

        # Confidence is retained in CSV for analysis,
        # but is NOT displayed on the figure.
        "confidence":
            confidence,

        "correct":
            correct,

        "gradcam_target_index":
            predicted_idx,

        "gradcam_target_class":
            predicted_class,

        "original_width":
            original_w,

        "original_height":
            original_h,

        "output_file":
            save_path
    })


# ============================================================
# 24. VERIFY PROCESSING
# ============================================================

if len(records) == 0:

    raise RuntimeError(
        "No images were successfully processed."
    )


# ============================================================
# 25. SAVE FULL AUDIT CSV
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
        fieldnames=list(
            records[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(
        records
    )


# ============================================================
# 26. MISCLASSIFIED CASES
# ============================================================

misclassified_records = [

    record
    for record in records
    if not record["correct"]

]


# ============================================================
# 27. SAVE MISCLASSIFIED CSV
# ============================================================

misclassified_csv = os.path.join(
    SAVE_DIR,
    "Misclassified_Cases.csv"
)

if len(misclassified_records) > 0:

    with open(
        misclassified_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                misclassified_records[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            misclassified_records
        )

else:

    no_error_file = os.path.join(
        SAVE_DIR,
        "NO_MISCLASSIFICATIONS.txt"
    )

    with open(
        no_error_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "No misclassified test images "
            "were detected."
        )


# ============================================================
# 28. OVERALL SUMMARY
# ============================================================

total = len(
    records
)

num_correct = sum(
    int(record["correct"])
    for record in records
)

num_errors = (
    total - num_correct
)

accuracy = (
    num_correct / total
)


# ============================================================
# 29. CLASS-WISE SUMMARY
# ============================================================

class_summary = {}

for class_name in EXPECTED_CLASSES:

    class_records = [

        record
        for record in records
        if record["true_class"]
        == class_name

    ]

    class_correct = sum(

        int(record["correct"])
        for record in class_records

    )

    class_errors = (
        len(class_records)
        - class_correct
    )

    class_summary[
        class_name
    ] = {

        "total":
            len(class_records),

        "correct":
            class_correct,

        "errors":
            class_errors
    }


# ============================================================
# 30. PRINT FINAL RESULTS
# ============================================================

print("\n\n========================================")
print("GRAD-CAM++ COMPLETE")
print("========================================")

print(
    f"Total processed images : {total}"
)

print(
    f"Correct predictions    : {num_correct}"
)

print(
    f"Misclassifications     : {num_errors}"
)

print(
    f"Accuracy               : "
    f"{accuracy * 100:.4f}%"
)


print("\n========================================")
print("VERIFIED CLASS MAPPING")
print("========================================")

for idx, class_name in enumerate(
    EXPECTED_CLASSES
):

    print(
        f"{idx} = "
        f"{DISPLAY_NAMES[class_name]}"
    )


print("\n========================================")
print("CLASS-WISE RESULTS")
print("========================================")

for class_name in EXPECTED_CLASSES:

    result = (
        class_summary[
            class_name
        ]
    )

    print(
        f"{DISPLAY_NAMES[class_name]}: "
        f"{result['correct']}/"
        f"{result['total']} correct | "
        f"{result['errors']} errors"
    )


print("\n========================================")
print("MISCLASSIFIED CASES")
print("========================================")

if num_errors == 0:

    print(
        "No misclassified test images detected."
    )

else:

    for record in misclassified_records:

        print(
            f"{record['filename']} | "
            f"TRUE = "
            f"{DISPLAY_NAMES[record['true_class']]} | "
            f"PREDICTED = "
            f"{DISPLAY_NAMES[record['predicted_class']]} | "
            f"CONFIDENCE = "
            f"{record['confidence'] * 100:.2f}%"
        )


print("\n========================================")
print("OUTPUT FILES")
print("========================================")

print(
    f"\nGrad-CAM++ images:\n"
    f"{SAVE_DIR}"
)

print(
    f"\nComplete prediction audit:\n"
    f"{csv_path}"
)

if num_errors > 0:

    print(
        f"\nMisclassified cases audit:\n"
        f"{misclassified_csv}"
    )

print("\nDONE.")