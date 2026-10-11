# ============================================================
# FINAL MODEL EVALUATION
# Hybrid CNN + BiLSTM + Additive Attention
#
# This script evaluates ONLY the saved BEST checkpoint.
#
# Outputs:
#   - Accuracy
#   - Macro Precision
#   - Macro Recall
#   - Macro F1
#   - Per-class Precision / Recall / F1
#   - Confusion Matrix
#   - Classification Report
#   - Per-class ROC curves
#   - Per-class AUC
#   - Macro AUC
#   - Prediction CSV
#   - Per-class metrics CSV
#   - Evaluation summary JSON
# ============================================================

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize


# ============================================================
# 1. CONFIGURATION
# ============================================================

TEST_DIR = r"C:\user\juli\thesis\Dataset\orgnl_preprocessed_testt"

MODEL_PATH = (
    r"C:\user\juli\thesis\training_outputs"
    r"\hybrid_cnn_bilstm_attention_best.pth"
)

SAVE_DIR = (
    r"C:\user\juli\thesis\evaluation_results_crosscheck_B_A"
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

INPUT_SIZE = (256, 256)

NUM_CLASSES = 4

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# IMPORTANT:
# These MUST match the class folder names/order used
# during training with ImageFolder.

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("=" * 75)
print("FINAL HYBRID CNN-BiLSTM-ATTENTION MODEL EVALUATION")
print("=" * 75)

print(f"Device       : {DEVICE}")
print(f"Test folder  : {TEST_DIR}")
print(f"Model        : {MODEL_PATH}")
print(f"Output folder: {SAVE_DIR}")

print("=" * 75)


# ============================================================
# 3. CHECK PATHS
# ============================================================

if not os.path.isdir(TEST_DIR):

    raise FileNotFoundError(
        f"\nTest directory does not exist:\n{TEST_DIR}"
    )


if not os.path.isfile(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel checkpoint does not exist:\n{MODEL_PATH}"
    )


# ============================================================
# 4. TEST TRANSFORM
# ============================================================
#
# MUST MATCH THE TRANSFORM USED DURING TRAINING.
#
# Training pipeline:
#   Grayscale -> 3 channels
#   Resize 256x256
#   ToTensor
#   Normalize mean=0.5, std=0.5
#
# No augmentation is used during final testing.
# ============================================================

test_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        INPUT_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )

])


# ============================================================
# 5. LOAD TEST DATASET
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


print("\nDataset Information")
print("-" * 75)

print(
    f"Number of test images : "
    f"{len(test_dataset)}"
)

print(
    f"Detected classes      : "
    f"{test_dataset.classes}"
)

print(
    f"Class-to-index        : "
    f"{test_dataset.class_to_idx}"
)

print("-" * 75)


# ============================================================
# 6. VERIFY CLASS ORDER
# ============================================================

if test_dataset.classes != CLASS_NAMES:

    print("\nERROR: CLASS ORDER / FOLDER NAME MISMATCH")
    print("\nDetected by ImageFolder:")
    print(test_dataset.classes)

    print("\nExpected:")
    print(CLASS_NAMES)

    raise ValueError(
        "\nThe test dataset class order does not match "
        "the class order used by the model."
    )


print("\nClass order verified successfully.")


# ============================================================
# 7. MODEL ARCHITECTURE
# ============================================================


# ------------------------------------------------------------
# Residual Block
# ------------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        pool=True
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )


        # Projection shortcut when the number
        # of channels changes.

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
            self.skip_bn = None


        self.pool = (

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

            if pool

            else None

        )


    def forward(self, x):

        identity = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)


        if self.skip_conv is not None:

            identity = self.skip_conv(
                identity
            )

            identity = self.skip_bn(
                identity
            )


        out = out + identity

        out = self.relu(out)


        if self.pool is not None:

            out = self.pool(out)


        return out


# ------------------------------------------------------------
# Custom CNN
# ------------------------------------------------------------

class CustomCNN(nn.Module):

    def __init__(self):

        super().__init__()


        self.block1 = ResidualBlock(
            3,
            32,
            pool=True
        )


        self.block2 = ResidualBlock(
            32,
            64,
            pool=True
        )


        self.block3 = ResidualBlock(
            64,
            128,
            pool=True
        )


        self.block4 = ResidualBlock(
            128,
            128,
            pool=True
        )


        self.block5 = ResidualBlock(
            128,
            256,
            pool=False
        )


        self.adaptive_pool = (
            nn.AdaptiveAvgPool2d(
                (8, 8)
            )
        )


    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        x = self.adaptive_pool(x)

        return x


# ------------------------------------------------------------
# Additive Attention
# ------------------------------------------------------------

class AdditiveAttention(nn.Module):

    def __init__(
        self,
        hidden_dim=128
    ):

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


    def forward(
        self,
        lstm_outputs
    ):

        energy = self.tanh(
            self.attn(
                lstm_outputs
            )
        )


        scores = self.score(
            energy
        )


        weights = torch.softmax(
            scores,
            dim=1
        )


        context = torch.sum(
            weights * lstm_outputs,
            dim=1
        )


        return context, weights


# ------------------------------------------------------------
# Hybrid CNN-BiLSTM-Attention
# ------------------------------------------------------------

class HybridCNNBiLSTMAttention(
    nn.Module
):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()


        # CNN

        self.cnn = CustomCNN()


        # BiLSTM

        self.bilstm = nn.LSTM(

            input_size=2048,

            hidden_size=128,

            num_layers=1,

            batch_first=True,

            bidirectional=True,

            dropout=0.0
        )


        # Attention

        self.attention = AdditiveAttention(
            hidden_dim=128
        )


        # Fully connected classifier

        self.fc = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                num_classes
            )

        )


    def forward(self, x):

        # CNN feature extraction

        x = self.cnn(x)

        # Expected:
        # [batch, 256, 8, 8]

        b, c, h, w = x.shape


        # Convert CNN feature map into
        # a sequence for BiLSTM.

        x = x.view(
            b,
            h,
            c * w
        )

        # Expected:
        # [batch, 8, 2048]


        # BiLSTM

        x, _ = self.bilstm(x)

        # Expected:
        # [batch, 8, 256]


        # Attention

        x, _ = self.attention(x)

        # Expected:
        # [batch, 256]


        # Classification

        x = self.fc(x)

        return x


# ============================================================
# 8. CREATE MODEL
# ============================================================

print("\nCreating model...")
print("-" * 75)


model = HybridCNNBiLSTMAttention(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# 9. LOAD BEST CHECKPOINT
# ============================================================

print("\nLoading BEST checkpoint...")
print("-" * 75)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Handle checkpoint dictionary
# ------------------------------------------------------------

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print(
            "Checkpoint format: "
            "training checkpoint dictionary"
        )

    else:

        # In case the file itself is a state_dict.

        state_dict = checkpoint

        print(
            "Checkpoint format: "
            "direct state_dict"
        )

else:

    raise TypeError(
        "\nUnsupported checkpoint format."
    )


# ============================================================
# 10. STRICT WEIGHT LOADING
# ============================================================

print("\nLoading model weights...")
print("-" * 75)


model.load_state_dict(
    state_dict,
    strict=True
)


model.eval()


print(
    "Model weights loaded successfully."
)


# ============================================================
# 11. CHECKPOINT INFORMATION
# ============================================================

print("\nCheckpoint Information")
print("-" * 75)


if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch : "
            f"{checkpoint['epoch']}"
        )


    if "val_acc" in checkpoint:

        print(
            f"Checkpoint val acc: "
            f"{checkpoint['val_acc']:.4f}"
        )


    if "val_loss" in checkpoint:

        print(
            f"Checkpoint val loss: "
            f"{checkpoint['val_loss']:.4f}"
        )


print("-" * 75)


# ============================================================
# 12. PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\nModel Information")
print("-" * 75)

print(
    f"Total parameters     : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters : "
    f"{trainable_params:,}"
)

print("-" * 75)


# ============================================================
# 13. FINAL EVALUATION
# ============================================================

print("\nStarting FINAL evaluation...")
print("-" * 75)


all_labels = []

all_predictions = []

all_probabilities = []


# ImageFolder keeps samples in a fixed order
# and shuffle=False is used.
#
# Therefore these paths correspond directly
# to the prediction order.

all_paths = [
    path
    for path, _ in test_dataset.samples
]


model.eval()


with torch.no_grad():

    for images, labels in test_loader:


        images = images.to(
            DEVICE,
            non_blocking=True
        )


        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        # Forward pass

        outputs = model(
            images
        )


        # Convert logits to probabilities

        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        # Predicted class

        predictions = torch.argmax(
            probabilities,
            dim=1
        )


        # Store results

        all_labels.extend(
            labels.cpu().numpy()
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )


        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# Convert to NumPy

y_true = np.asarray(
    all_labels
)

y_pred = np.asarray(
    all_predictions
)

y_prob = np.asarray(
    all_probabilities
)


# ============================================================
# 14. SANITY CHECK
# ============================================================

print("\nEvaluation sanity check")
print("-" * 75)

print(
    f"Labels collected      : {len(y_true)}"
)

print(
    f"Predictions collected : {len(y_pred)}"
)

print(
    f"Probability rows      : {len(y_prob)}"
)

print(
    f"Image paths           : {len(all_paths)}"
)


if not (
    len(y_true)
    == len(y_pred)
    == len(y_prob)
    == len(all_paths)
):

    raise RuntimeError(
        "\nMismatch between predictions, "
        "labels, probabilities, and image paths."
    )


print(
    "All evaluation arrays have matching lengths."
)


# ============================================================
# 15. OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 16. PRINT OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST RESULTS")
print("=" * 75)

print(
    f"Accuracy            : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision     : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall        : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1-score      : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted Precision  : "
    f"{weighted_precision:.4f}"
)

print(
    f"Weighted Recall     : "
    f"{weighted_recall:.4f}"
)

print(
    f"Weighted F1-score   : "
    f"{weighted_f1:.4f}"
)

print("=" * 75)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

report_text = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0
)


report_dict = classification_report(

    y_true,

    y_pred,

    target_names=CLASS_NAMES,

    digits=4,

    output_dict=True,

    zero_division=0
)


print("\nClassification Report")
print("=" * 75)

print(report_text)


# Save text report

with open(

    os.path.join(
        SAVE_DIR,
        "classification_report.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write(report_text)


# Save CSV report

pd.DataFrame(
    report_dict
).transpose().to_csv(

    os.path.join(
        SAVE_DIR,
        "classification_report.csv"
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_true,

    y_pred,

    labels=np.arange(
        NUM_CLASSES
    )
)


print("\nConfusion Matrix")
print("=" * 75)

print(cm)


# Save raw confusion matrix

np.savetxt(

    os.path.join(
        SAVE_DIR,
        "confusion_matrix.csv"
    ),

    cm,

    delimiter=",",

    fmt="%d"
)


# ------------------------------------------------------------
# Plot confusion matrix
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    NUM_CLASSES
)


plt.xticks(

    tick_marks,

    CLASS_NAMES,

    rotation=45,

    ha="right"
)


plt.yticks(

    tick_marks,

    CLASS_NAMES
)


threshold = cm.max() / 2.0


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            str(cm[i, j]),

            horizontalalignment="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )

        )


plt.ylabel(
    "True Label"
)


plt.xlabel(
    "Predicted Label"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        SAVE_DIR,
        "confusion_matrix.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 19. PER-CLASS METRICS
# ============================================================

class_precision = precision_score(

    y_true,

    y_pred,

    average=None,

    labels=np.arange(
        NUM_CLASSES
    ),

    zero_division=0
)


class_recall = recall_score(

    y_true,

    y_pred,

    average=None,

    labels=np.arange(
        NUM_CLASSES
    ),

    zero_division=0
)


class_f1 = f1_score(

    y_true,

    y_pred,

    average=None,

    labels=np.arange(
        NUM_CLASSES
    ),

    zero_division=0
)


# Number of samples per class

class_support = np.bincount(

    y_true,

    minlength=NUM_CLASSES
)


per_class_df = pd.DataFrame({

    "Class": CLASS_NAMES,

    "Support": class_support,

    "Precision": class_precision,

    "Recall": class_recall,

    "F1_Score": class_f1

})


print("\nPer-Class Metrics")
print("-" * 75)

print(
    per_class_df.to_string(
        index=False
    )
)


per_class_df.to_csv(

    os.path.join(
        SAVE_DIR,
        "per_class_metrics.csv"
    ),

    index=False
)


# ============================================================
# 20. ROC / AUC
# ============================================================

y_true_bin = label_binarize(

    y_true,

    classes=np.arange(
        NUM_CLASSES
    )
)


auc_values = {}


fpr_values = {}

tpr_values = {}


plt.figure(
    figsize=(9, 7)
)


for i, class_name in enumerate(
    CLASS_NAMES
):

    fpr, tpr, _ = roc_curve(

        y_true_bin[:, i],

        y_prob[:, i]
    )


    class_auc = auc(

        fpr,

        tpr
    )


    auc_values[class_name] = class_auc

    fpr_values[class_name] = fpr

    tpr_values[class_name] = tpr


    plt.plot(

        fpr,

        tpr,

        label=(
            f"{class_name} "
            f"(AUC = {class_auc:.4f})"
        )
    )


# Diagonal reference

plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "One-vs-Rest ROC Curves"
)


plt.legend(
    loc="lower right"
)


plt.grid(
    True,

    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        SAVE_DIR,
        "roc_curves.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# 21. MACRO AUC
# ============================================================

macro_auc = roc_auc_score(

    y_true_bin,

    y_prob,

    average="macro",

    multi_class="ovr"
)


# Also calculate the simple mean of class AUCs
# for verification.

mean_class_auc = np.mean(
    list(auc_values.values())
)


print("\nAUC Results")
print("-" * 75)


for class_name, value in auc_values.items():

    print(
        f"{class_name:20s}: "
        f"{value:.4f}"
    )


print(
    f"{'Macro AUC':20s}: "
    f"{macro_auc:.4f}"
)


print(
    f"{'Mean Class AUC':20s}: "
    f"{mean_class_auc:.4f}"
)


# ============================================================
# 22. SAVE AUC CSV
# ============================================================

auc_df = pd.DataFrame({

    "Class": CLASS_NAMES,

    "AUC": [
        auc_values[name]
        for name in CLASS_NAMES
    ]

})


auc_df.loc[
    len(auc_df)
] = [
    "Macro AUC",
    macro_auc
]


auc_df.to_csv(

    os.path.join(
        SAVE_DIR,
        "auc_metrics.csv"
    ),

    index=False
)


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================

prediction_rows = []


for i in range(
    len(y_true)
):

    true_index = int(
        y_true[i]
    )

    predicted_index = int(
        y_pred[i]
    )


    row = {

        "Image_Path": all_paths[i],

        "True_Label": CLASS_NAMES[
            true_index
        ],

        "Predicted_Label": CLASS_NAMES[
            predicted_index
        ],

        "Correct": (
            true_index
            == predicted_index
        )

    }


    # Save probability for each class

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        row[
            f"Probability_{class_name}"
        ] = float(
            y_prob[i, class_index]
        )


    prediction_rows.append(
        row
    )


predictions_df = pd.DataFrame(
    prediction_rows
)


predictions_df.to_csv(

    os.path.join(
        SAVE_DIR,
        "predictions.csv"
    ),

    index=False
)


# ============================================================
# 24. SAVE SUMMARY JSON
# ============================================================

summary = {

    "model_path": MODEL_PATH,

    "test_directory": TEST_DIR,

    "model_architecture":
        "Hybrid CNN-BiLSTM-Attention",

    "num_test_images":
        int(len(y_true)),

    "num_classes":
        NUM_CLASSES,

    "classes":
        CLASS_NAMES,

    "num_parameters":
        int(total_params),

    "trainable_parameters":
        int(trainable_params),

    "accuracy":
        float(accuracy),

    "accuracy_percent":
        float(
            accuracy * 100
        ),

    "macro_precision":
        float(
            macro_precision
        ),

    "macro_recall":
        float(
            macro_recall
        ),

    "macro_f1":
        float(
            macro_f1
        ),

    "weighted_precision":
        float(
            weighted_precision
        ),

    "weighted_recall":
        float(
            weighted_recall
        ),

    "weighted_f1":
        float(
            weighted_f1
        ),

    "macro_auc":
        float(
            macro_auc
        ),

    "class_auc": {

        class_name:
            float(
                auc_values[class_name]
            )

        for class_name in CLASS_NAMES

    }

}


# Include checkpoint epoch if available

if isinstance(checkpoint, dict):

    if "epoch" in checkpoint:

        summary[
            "best_checkpoint_epoch"
        ] = int(
            checkpoint["epoch"]
        )


    if "val_acc" in checkpoint:

        summary[
            "best_checkpoint_val_accuracy"
        ] = float(
            checkpoint["val_acc"]
        )


    if "val_loss" in checkpoint:

        summary[
            "best_checkpoint_val_loss"
        ] = float(
            checkpoint["val_loss"]
        )


with open(

    os.path.join(
        SAVE_DIR,
        "evaluation_summary.json"
    ),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        indent=4
    )


# ============================================================
# 25. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    f"Test Images       : "
    f"{len(y_true)}"
)

print(
    f"Accuracy          : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision   : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall      : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1          : "
    f"{macro_f1:.4f}"
)

print(
    f"Macro AUC         : "
    f"{macro_auc:.4f}"
)

print(
    f"\nResults saved to:"
)

print(
    SAVE_DIR
)

print("=" * 75)