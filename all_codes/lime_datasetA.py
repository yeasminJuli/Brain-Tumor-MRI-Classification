# ============================================================
# FINAL LIME EXPLANATION SCRIPT — DATASET A
# Hybrid CNN -> BiLSTM -> Additive Attention -> FC
# ============================================================

import os
import csv
import random
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from lime import lime_image
from sklearn.linear_model import Ridge
from skimage.segmentation import slic, mark_boundaries


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_SEED = 42

TEST_DIR = (
    r"C:\user\juli\thesis\original_file\Brain_Tumor"
    r"\othr folders\preprocessed_testt"
)

MODEL_PATH = (
    r"C:\user\juli\thesis\original_file\Brain_Tumor"
    r"\othr folders\training_outputs_proposed_model_orgnl"
    r"\training_outputs\hybrid_cnn_bilstm_attention_final.pth"
)

OUTPUT_DIR = (
    r"C:\user\juli\thesis\original_file\Brain_Tumor"
    r"\othr folders\LIME_Output"
)

# Number of images to explain from each TRUE class.
# Change to None if you want every test image.
IMAGES_PER_CLASS = 5

# ------------------------------------------------------------
# LIME parameters reported in manuscript
# ------------------------------------------------------------

NUM_SAMPLES = 1500
NUM_SUPERPIXELS = 50
COMPACTNESS = 10
SIGMA = 1
KERNEL_WIDTH = 0.25
RIDGE_ALPHA = 1.0
NUM_FEATURES = 10

IMAGE_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(BASE_SEED)


def deterministic_image_seed(image_path):
    relative_path = os.path.relpath(
        image_path,
        TEST_DIR
    ).replace("\\", "/")

    digest = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()

    image_seed = (
        BASE_SEED + int(digest[:8], 16)
    ) % (2**32 - 1)

    return image_seed


# ============================================================
# 3. EXPECTED DATASET A CLASS MAPPING
# ============================================================

EXPECTED_CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3,
}


# ============================================================
# 4. MODEL TRANSFORM
# ============================================================
#
# EXACTLY matches training/test preprocessing:
#
# grayscale
# -> replicate to 3 channels
# -> resize 256 x 256
# -> ToTensor
# -> Normalize mean=0.5, std=0.5
#
# ============================================================

model_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# 5. LOAD DATASET A TEST SET
# ============================================================

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"\nTest directory does not exist:\n{TEST_DIR}"
    )


dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=model_transform,
)


print("\n============================================")
print("DATASET A CLASS VERIFICATION")
print("============================================")

print("Classes found:")
print(dataset.classes)

print("\nclass_to_idx:")
print(dataset.class_to_idx)


# ------------------------------------------------------------
# Critical dataset mapping check
# ------------------------------------------------------------

if dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:
    raise RuntimeError(
        "\nCLASS MAPPING ERROR!\n\n"
        f"Expected:\n{EXPECTED_CLASS_TO_IDX}\n\n"
        f"Dataset returned:\n{dataset.class_to_idx}\n\n"
        "LIME generation has been STOPPED.\n"
        "Do not generate explanations until the mapping "
        "is resolved."
    )


IDX_TO_CLASS = {
    idx: class_name
    for class_name, idx
    in dataset.class_to_idx.items()
}


print("\nVerified Dataset A mapping:")

for idx in range(4):
    print(f"{idx} = {IDX_TO_CLASS[idx]}")


# ============================================================
# 6. EXACT MODEL ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        use_pool=False
    ):
        super().__init__()

        self.use_pool = use_pool

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:

            self.skip_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

            self.skip_bn = nn.BatchNorm2d(
                out_channels
            )

        else:
            self.skip_conv = None

        self.pool = (
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
            if use_pool
            else nn.Identity()
        )


    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)

        out += identity
        out = self.relu(out)

        out = self.pool(out)

        return out


class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = ResidualBlock(
            3, 32, use_pool=True
        )

        self.block2 = ResidualBlock(
            32, 64, use_pool=True
        )

        self.block3 = ResidualBlock(
            64, 128, use_pool=True
        )

        self.block4 = ResidualBlock(
            128, 128, use_pool=True
        )

        self.block5 = ResidualBlock(
            128, 256, use_pool=False
        )

        self.final_pool = nn.AdaptiveAvgPool2d(
            (8, 8)
        )


    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.final_pool(x)

        return x


class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(
            hidden_dim * 2,
            128
        )

        self.tanh = nn.Tanh()

        self.score = nn.Linear(
            128,
            1
        )

        self.softmax = nn.Softmax(
            dim=1
        )


    def forward(self, lstm_outputs):

        e = self.tanh(
            self.attn(lstm_outputs)
        )

        scores = self.score(e)

        alpha = self.softmax(scores)

        context = torch.sum(
            alpha * lstm_outputs,
            dim=1
        )

        return context, alpha


class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        num_classes=4,
        lstm_hidden=128,
        lstm_layers=1
    ):
        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.hidden_size = lstm_hidden
        self.num_layers = lstm_layers

        # CNN output = (B, 256, 8, 8)
        # Row-wise sequence:
        # sequence length = 8
        # input size = 256 * 8 = 2048

        self.lstm_input_size = 256 * 8

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                0.3
                if self.num_layers > 1
                else 0.0
            )
        )

        self.attention = AdditiveAttention(
            self.hidden_size
        )

        self.fc = nn.Sequential(

            nn.Linear(
                self.hidden_size * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # (B, 256, 8, 8)
        b, c, h, w = x.size()

        # Row-wise sequence ordering
        x = x.contiguous().view(
            b,
            h,
            c * w
        )

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Attention
        context, attn_weights = (
            self.attention(lstm_out)
        )

        # Classifier
        out = self.fc(context)

        return out, attn_weights


# ============================================================
# 7. LOAD TRAINED CHECKPOINT
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel checkpoint does not exist:\n"
        f"{MODEL_PATH}"
    )


print("\n============================================")
print("LOADING MODEL")
print("============================================")

print("Device:", DEVICE)
print("Checkpoint:", MODEL_PATH)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


# ------------------------------------------------------------
# Extract state dictionary
# ------------------------------------------------------------

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]

else:
    state_dict = checkpoint


# ------------------------------------------------------------
# Read class names saved in checkpoint
# ------------------------------------------------------------

checkpoint_class_names = None

if isinstance(checkpoint, dict):
    checkpoint_class_names = checkpoint.get(
        "class_names",
        None
    )


print("\nCheckpoint class_names:")
print(checkpoint_class_names)


# ------------------------------------------------------------
# CRITICAL CHECKPOINT CLASS ORDER CHECK
# ------------------------------------------------------------

expected_class_names = [
    IDX_TO_CLASS[i]
    for i in range(4)
]


if checkpoint_class_names is not None:

    checkpoint_class_names = list(
        checkpoint_class_names
    )

    if checkpoint_class_names != expected_class_names:

        raise RuntimeError(
            "\nCHECKPOINT CLASS ORDER ERROR!\n\n"

            f"Dataset/model expected:\n"
            f"{expected_class_names}\n\n"

            f"Checkpoint contains:\n"
            f"{checkpoint_class_names}\n\n"

            "The checkpoint class order does not match "
            "Dataset A.\n"

            "LIME generation has been STOPPED to prevent "
            "incorrect class labeling."
        )

else:

    print(
        "\nWARNING: checkpoint does not contain "
        "'class_names'."
    )

    print(
        "Dataset mapping has still been verified "
        "against the expected Dataset A mapping."
    )


# ------------------------------------------------------------
# Create exact architecture
# ------------------------------------------------------------

model = HybridCNNBiLSTMAttention(
    num_classes=4
).to(DEVICE)


# ------------------------------------------------------------
# Remove DataParallel prefix if present
# ------------------------------------------------------------

clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()


print("\nModel weights loaded successfully.")


# ============================================================
# 8. QUICK MODEL OUTPUT TEST
# ============================================================

with torch.no_grad():

    dummy = torch.zeros(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE
    ).to(DEVICE)

    dummy_logits, dummy_attention = model(dummy)


if dummy_logits.shape != (1, 4):

    raise RuntimeError(
        "\nMODEL OUTPUT ERROR!\n"
        f"Expected logits shape (1,4), "
        f"received {tuple(dummy_logits.shape)}"
    )


print(
    "Model output shape:",
    tuple(dummy_logits.shape)
)

print(
    "Attention shape:",
    tuple(dummy_attention.shape)
)


# ============================================================
# 9. LIME INPUT PREPROCESSING
# ============================================================

def preprocess_lime_batch(images):
    """
    LIME provides images as numpy arrays:
        [N, H, W, 3]

    The source image is grayscale replicated into RGB.
    LIME masking uses black pixels, so the three channels
    remain identical.

    Therefore we use channel 0 as the grayscale image,
    replicate it into 3 channels, and apply the exact
    model normalization.
    """

    images = np.asarray(
        images,
        dtype=np.float32
    )

    tensors = []

    for img in images:

        # ----------------------------------------------------
        # Get grayscale
        # ----------------------------------------------------

        if img.ndim == 3:

            # Because source image is grayscale replicated
            # into RGB and masking is black, all channels
            # should remain equivalent.
            gray = img[:, :, 0]

        elif img.ndim == 2:
            gray = img

        else:
            raise ValueError(
                f"Unexpected LIME image shape: "
                f"{img.shape}"
            )


        # ----------------------------------------------------
        # Ensure [0,1]
        # ----------------------------------------------------

        gray = np.clip(
            gray,
            0.0,
            1.0
        )


        # ----------------------------------------------------
        # Replicate grayscale -> 3 channels
        # ----------------------------------------------------

        gray3 = np.stack(
            [gray, gray, gray],
            axis=0
        ).astype(np.float32)


        tensor = torch.from_numpy(gray3)


        # ----------------------------------------------------
        # Exact model normalization
        # (x - 0.5) / 0.5
        # ----------------------------------------------------

        tensor = (
            tensor - 0.5
        ) / 0.5


        tensors.append(tensor)


    return torch.stack(tensors)


# ============================================================
# 10. LIME CLASSIFIER FUNCTION
# ============================================================

def batch_predict(images):

    batch = preprocess_lime_batch(
        images
    ).to(DEVICE)


    model.eval()


    with torch.no_grad():

        # IMPORTANT:
        # Model returns TWO outputs.
        logits, _ = model(batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )


    return probabilities.cpu().numpy()


# ============================================================
# 11. SLIC SEGMENTATION
# ============================================================

def slic_segmentation(image):

    segments = slic(
        image,
        n_segments=NUM_SUPERPIXELS,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0,
        channel_axis=-1
    )

    return segments


# ============================================================
# 12. LOAD ORIGINAL IMAGE FOR LIME
# ============================================================

def load_lime_image(image_path):

    with Image.open(image_path) as pil_image:

        pil_image = pil_image.convert("L")

        pil_image = pil_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image = np.asarray(
            pil_image,
            dtype=np.float32
        )


    # Convert 0-255 -> 0-1
    image /= 255.0


    # Replicate grayscale -> RGB
    image_rgb = np.stack(
        [image, image, image],
        axis=-1
    )


    return image_rgb.astype(
        np.float32
    )


# ============================================================
# 13. DIRECT MODEL PREDICTION
# ============================================================

def direct_prediction(image_path):

    with Image.open(image_path) as image:

        image = image.copy()


    tensor = model_transform(
        image
    )

    tensor = tensor.unsqueeze(0).to(
        DEVICE
    )


    model.eval()


    with torch.no_grad():

        # IMPORTANT:
        # Exact model returns logits + attention.
        logits, _ = model(tensor)

        probs = torch.softmax(
            logits,
            dim=1
        )[0]


    pred_idx = int(
        torch.argmax(probs).item()
    )

    confidence = float(
        probs[pred_idx].item()
    )


    return (
        pred_idx,
        confidence,
        probs.cpu().numpy()
    )


# ============================================================
# 14. CRITICAL DIRECT-vs-LIME CONSISTENCY CHECK
# ============================================================

def verify_lime_prediction_pipeline(
    image_path
):

    direct_idx, direct_conf, direct_probs = (
        direct_prediction(image_path)
    )


    lime_img = load_lime_image(
        image_path
    )


    lime_probs = batch_predict(
        np.expand_dims(
            lime_img,
            axis=0
        )
    )[0]


    lime_idx = int(
        np.argmax(lime_probs)
    )


    # --------------------------------------------------------
    # Prediction must be identical
    # --------------------------------------------------------

    if direct_idx != lime_idx:

        raise RuntimeError(
            "\nLIME PREPROCESSING ERROR!\n\n"

            f"Image:\n{image_path}\n\n"

            f"Direct prediction:\n"
            f"{direct_idx} = "
            f"{IDX_TO_CLASS[direct_idx]}\n\n"

            f"LIME preprocessing prediction:\n"
            f"{lime_idx} = "
            f"{IDX_TO_CLASS[lime_idx]}\n\n"

            "LIME generation has been STOPPED."
        )


    # --------------------------------------------------------
    # Probability consistency
    # --------------------------------------------------------

    max_probability_difference = float(
        np.max(
            np.abs(
                direct_probs - lime_probs
            )
        )
    )


    if max_probability_difference > 1e-4:

        raise RuntimeError(
            "\nLIME PROBABILITY CONSISTENCY ERROR!\n\n"

            f"Image:\n{image_path}\n\n"

            f"Maximum probability difference: "
            f"{max_probability_difference:.8f}\n\n"

            "The LIME preprocessing is not reproducing "
            "the normal model input closely enough."
        )


    return (
        direct_idx,
        direct_conf,
        max_probability_difference
    )


# ============================================================
# 15. GENERATE ONE LIME EXPLANATION
# ============================================================

def generate_lime(
    image_path,
    true_idx
):

    image_seed = deterministic_image_seed(
        image_path
    )


    # --------------------------------------------------------
    # Verify normal model and LIME pipeline
    # --------------------------------------------------------

    (
        pred_idx,
        confidence,
        probability_difference
    ) = verify_lime_prediction_pipeline(
        image_path
    )


    true_name = IDX_TO_CLASS[
        true_idx
    ]

    pred_name = IDX_TO_CLASS[
        pred_idx
    ]


    print(
        "\n--------------------------------------------"
    )

    print(
        "Image:",
        os.path.basename(image_path)
    )

    print(
        f"True class: "
        f"{true_idx} = {true_name}"
    )

    print(
        f"Predicted class: "
        f"{pred_idx} = {pred_name}"
    )

    print(
        f"Confidence: "
        f"{confidence:.6f}"
    )

    print(
        f"Direct/LIME probability difference: "
        f"{probability_difference:.8f}"
    )

    print(
        f"LIME seed: {image_seed}"
    )


    # --------------------------------------------------------
    # Load unnormalized [0,1] image
    # --------------------------------------------------------

    image = load_lime_image(
        image_path
    )


    # --------------------------------------------------------
    # LIME explainer
    # --------------------------------------------------------

    explainer = (
        lime_image.LimeImageExplainer(
            kernel_width=KERNEL_WIDTH,
            kernel=None,
            verbose=False,
            feature_selection="auto",
            random_state=image_seed
        )
    )


    # --------------------------------------------------------
    # Weighted Ridge surrogate
    # --------------------------------------------------------

    ridge_model = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True
    )


    # --------------------------------------------------------
    # Generate LIME explanation
    # --------------------------------------------------------

    explanation = (
        explainer.explain_instance(

            image=image,

            classifier_fn=batch_predict,

            # Explain MODEL-PREDICTED class
            labels=(pred_idx,),

            # Masked superpixels -> black
            hide_color=0,

            num_samples=NUM_SAMPLES,

            batch_size=32,

            segmentation_fn=slic_segmentation,

            distance_metric="cosine",

            model_regressor=ridge_model,

            random_seed=image_seed
        )
    )


    # --------------------------------------------------------
    # Actual number of SLIC superpixels
    # --------------------------------------------------------

    actual_superpixels = len(
        np.unique(
            explanation.segments
        )
    )


    print(
        "Requested SLIC superpixels:",
        NUM_SUPERPIXELS
    )

    print(
        "Actual SLIC superpixels:",
        actual_superpixels
    )


    # --------------------------------------------------------
    # Positive contributing superpixels only
    # --------------------------------------------------------

    explained_image, mask = (
        explanation.get_image_and_mask(

            label=pred_idx,

            positive_only=True,

            num_features=NUM_FEATURES,

            hide_rest=False
        )
    )


    # --------------------------------------------------------
    # Add boundaries
    # --------------------------------------------------------

    boundary_image = mark_boundaries(
        explained_image,
        mask,
        color=(1, 1, 0),
        mode="thick"
    )


    status = (
        "Correct"
        if true_idx == pred_idx
        else "Misclassified"
    )


    safe_filename = Path(
        image_path
    ).stem


    output_name = (
        f"{status}_"
        f"True-{true_name}_"
        f"Pred-{pred_name}_"
        f"{safe_filename}.png"
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    # --------------------------------------------------------
    # High-resolution figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4)
    )


    # Original MRI
    axes[0].imshow(
        image[:, :, 0],
        cmap="gray"
    )

    axes[0].set_title(
        f"Original MRI\n"
        f"True: {true_name}"
    )

    axes[0].axis("off")


    # LIME
    axes[1].imshow(
        boundary_image
    )

    axes[1].set_title(
        f"LIME Explanation\n"
        f"Predicted: {pred_name} "
        f"({confidence * 100:.2f}%)"
    )

    axes[1].axis("off")


    fig.tight_layout()


    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)


    print(
        "Saved:",
        output_path
    )


    return {

        "image_path":
            image_path,

        "true_index":
            true_idx,

        "true_class":
            true_name,

        "predicted_index":
            pred_idx,

        "predicted_class":
            pred_name,

        "confidence":
            confidence,

        "correct":
            true_idx == pred_idx,

        "requested_superpixels":
            NUM_SUPERPIXELS,

        "actual_superpixels":
            actual_superpixels,

        "num_samples":
            NUM_SAMPLES,

        "distance_metric":
            "cosine",

        "kernel_width":
            KERNEL_WIDTH,

        "ridge_alpha":
            RIDGE_ALPHA,

        "num_positive_features":
            NUM_FEATURES,

        "seed":
            image_seed,

        "output_file":
            output_path
    }


# ============================================================
# 16. SELECT IMAGES FROM EACH TRUE CLASS
# ============================================================

samples_by_class = {
    idx: []
    for idx in range(4)
}


for image_path, class_idx in dataset.samples:

    samples_by_class[
        class_idx
    ].append(
        image_path
    )


selected_samples = []


print("\n============================================")
print("TEST SET COUNTS")
print("============================================")


for class_idx in range(4):

    paths = sorted(
        samples_by_class[
            class_idx
        ]
    )


    print(
        f"{class_idx} = "
        f"{IDX_TO_CLASS[class_idx]}: "
        f"{len(paths)} images"
    )


    if IMAGES_PER_CLASS is not None:

        paths = paths[
            :IMAGES_PER_CLASS
        ]


    for path in paths:

        selected_samples.append(
            (
                path,
                class_idx
            )
        )


# ============================================================
# 17. FINAL SAFETY CHECK
# ============================================================

print("\n============================================")
print("FINAL VERIFIED CLASS MAPPING")
print("============================================")


for idx in range(4):

    print(
        f"Model output {idx} -> "
        f"{IDX_TO_CLASS[idx]}"
    )


print(
    "============================================"
)

print(
    "CLASS MAPPING CHECK: PASS"
)

print(
    "============================================\n"
)


# ============================================================
# 18. GENERATE EXPLANATIONS
# ============================================================

results = []


for number, (
    image_path,
    true_idx
) in enumerate(
    selected_samples,
    start=1
):

    print(
        f"\nProcessing "
        f"{number}/{len(selected_samples)}"
    )


    try:

        result = generate_lime(
            image_path,
            true_idx
        )

        results.append(
            result
        )


    except Exception:

        print(
            "\nERROR WHILE PROCESSING:"
        )

        print(
            image_path
        )

        # Stop immediately.
        # Do not silently create incorrect explanations.
        raise


# ============================================================
# 19. SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "LIME_RESULTS.csv"
)


fieldnames = [

    "image_path",

    "true_index",

    "true_class",

    "predicted_index",

    "predicted_class",

    "confidence",

    "correct",

    "requested_superpixels",

    "actual_superpixels",

    "num_samples",

    "distance_metric",

    "kernel_width",

    "ridge_alpha",

    "num_positive_features",

    "seed",

    "output_file",
]


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 20. SUMMARY
# ============================================================

correct_count = sum(
    int(result["correct"])
    for result in results
)

misclassified_count = (
    len(results) - correct_count
)


print("\n============================================")
print("LIME GENERATION COMPLETE")
print("============================================")

print(
    f"Total explanations: "
    f"{len(results)}"
)

print(
    f"Correct predictions: "
    f"{correct_count}"
)

print(
    f"Misclassified predictions: "
    f"{misclassified_count}"
)

print(
    f"Output directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nCSV:\n"
    f"{csv_path}"
)


print("\nCLASS MAPPING USED:")

for idx in range(4):

    print(
        f"{idx} = "
        f"{IDX_TO_CLASS[idx]}"
    )


print("============================================")