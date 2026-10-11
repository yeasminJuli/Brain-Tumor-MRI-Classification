# ================================================================
# EVALUATION OF SPATIAL-TO-SEQUENCE ORDERING ABLATION
#
# Conditions:
#   1. Row-wise Ordering (Original)
#   2. Column-wise Ordering
#   3. Alternative Ordering
#
# Evaluation:
#   - Accuracy
#   - Weighted Precision / Recall / F1
#   - Macro Precision / Recall / F1
#   - Macro ROC-AUC
#   - Per-class Precision / Recall / F1 / ROC-AUC
#   - Confusion matrices
#   - ROC curves
#   - Performance comparison
#   - Delta from original ordering
#
# IMPORTANT:
#   - NO TRAINING
#   - NO VALIDATION SET
#   - ONLY THE INDEPENDENT TEST SET
# ================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)

import matplotlib.pyplot as plt


# ================================================================
# 1. CONFIGURATION
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (256, 256)
NUM_CLASSES = 4
BATCH_SIZE = 16

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ------------------------------------------------
# YOUR TEST DATASET
# ------------------------------------------------
TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\preprocessed_testt"

# ------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------
OUTPUT_DIR = r"C:\user\juli\thesis\original_file\othr folders\evaluation_column_alternative_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------
# CHECKPOINTS
# ------------------------------------------------

# ORIGINAL ROW-WISE MODEL
# CHANGE THIS PATH IF YOUR ORIGINAL CHECKPOINT IS STORED ELSEWHERE.
ORIGINAL_CHECKPOINT = r"C:\user\juli\thesis\original_file\othr folders\training_outputs_proposed_model_orgnl\training_outputs\hybrid_cnn_bilstm_attention_final.pth"

# COLUMN-WISE MODEL
COLUMN_CHECKPOINT = r"C:\user\juli\thesis\original_file\othr folders\training_outputs_proposed_column_zigzag\column_wise\best_column_wise.pth"

# ALTERNATIVE ORDERING MODEL
ALTERNATIVE_CHECKPOINT = r"C:\user\juli\thesis\original_file\othr folders\training_outputs_proposed_column_zigzag\alternative\best_alternative.pth"


# ================================================================
# 2. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ================================================================
# 3. TEST TRANSFORM
# ================================================================

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ================================================================
# 4. MODEL COMPONENTS
# ================================================================

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

            self.skip_bn = nn.BatchNorm2d(out_channels)

        else:

            self.skip_conv = None

        self.pool = (
            nn.MaxPool2d(kernel_size=2, stride=2)
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


# ================================================================
# CNN FEATURE EXTRACTOR
# ================================================================

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


# ================================================================
# ADDITIVE ATTENTION
# ================================================================

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


# ================================================================
# HYBRID MODEL
# ================================================================

class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        ordering="row_wise",
        num_classes=4,
        lstm_hidden=128,
        lstm_layers=1
    ):

        super().__init__()

        self.ordering = ordering

        self.cnn = CNNFeatureExtractor()

        self.hidden_size = lstm_hidden
        self.num_layers = lstm_layers

        # Final CNN feature map:
        # (B, 256, 8, 8)

        self.c_channels = 256
        self.spatial_size = 8

        self.lstm_input_size = (
            self.c_channels *
            self.spatial_size
        )

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

    # ------------------------------------------------------------
    # ROW-WISE ORDERING
    # ------------------------------------------------------------

    def row_wise_ordering(self, x):

        # x = (B, C, H, W)
        #
        # Rows become sequence positions.
        #
        # Output:
        # (B, H, C*W)
        #
        # = (B, 8, 2048)

        b, c, h, w = x.size()

        x = (
            x
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b, h, c * w)
        )

        return x

    # ------------------------------------------------------------
    # COLUMN-WISE ORDERING
    # ------------------------------------------------------------

    def column_wise_ordering(self, x):

        # x = (B, C, H, W)
        #
        # Columns become sequence positions.
        #
        # Output:
        # (B, W, C*H)
        #
        # = (B, 8, 2048)

        b, c, h, w = x.size()

        x = (
            x
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(b, w, c * h)
        )

        return x

    # ------------------------------------------------------------
    # ALTERNATIVE ORDERING
    # ------------------------------------------------------------

    def alternative_ordering(self, x):

        # Alternating spatial traversal:
        #
        # Row 1: left -> right
        # Row 2: right -> left
        # Row 3: left -> right
        # Row 4: right -> left
        # ...
        #
        # Output:
        # (B, 8, 2048)

        b, c, h, w = x.size()

        sequence = []

        for row_idx in range(h):

            row = x[:, :, row_idx, :]
            # (B, C, W)

            if row_idx % 2 == 1:

                row = torch.flip(
                    row,
                    dims=[2]
                )

            row = row.contiguous().view(
                b,
                c * w
            )

            sequence.append(row)

        x = torch.stack(
            sequence,
            dim=1
        )

        return x

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------

    def forward(self, x):

        x = self.cnn(x)

        # x = (B, 256, 8, 8)

        if self.ordering == "row_wise":

            x = self.row_wise_ordering(x)

        elif self.ordering == "column_wise":

            x = self.column_wise_ordering(x)

        elif self.ordering == "alternative":

            x = self.alternative_ordering(x)

        else:

            raise ValueError(
                f"Unknown ordering: {self.ordering}"
            )

        # x = (B, 8, 2048)

        lstm_out, _ = self.bilstm(x)

        context, attn_weights = self.attention(
            lstm_out
        )

        out = self.fc(context)

        return out, attn_weights


# ================================================================
# 5. CHECKPOINT LOADING
# ================================================================

def load_checkpoint(model, checkpoint_path):

    print()
    print("=" * 70)
    print("Loading checkpoint:")
    print(checkpoint_path)

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint_path}\n\n"
            "Please correct the checkpoint path in the CONFIGURATION section."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # ------------------------------------------------------------
    # Handle checkpoint formats
    # ------------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # Could already be a raw state dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Remove possible DataParallel prefix
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_acc" in checkpoint:
            print(
                f"Checkpoint validation accuracy: "
                f"{checkpoint['val_acc']:.4f}"
            )

    return model


# ================================================================
# 6. DATASET
# ================================================================

print()
print("=" * 70)
print("SPATIAL ORDERING ABLATION EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Test directory: {TEST_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"\nTest directory not found:\n{TEST_DIR}"
    )


test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print()
print("Dataset class mapping:")
print(test_dataset.class_to_idx)

print()
print(
    f"Total independent test images: "
    f"{len(test_dataset)}"
)

expected_mapping = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

if test_dataset.class_to_idx != expected_mapping:

    raise ValueError(
        "\nClass mapping mismatch!\n"
        f"Expected: {expected_mapping}\n"
        f"Found: {test_dataset.class_to_idx}"
    )

print("Class mapping verified.")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ================================================================
# 7. EVALUATION FUNCTION
# ================================================================

def evaluate_model(
    model,
    loader,
    class_names,
    condition_name
):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits, _ = model(images)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item() *
                labels.size(0)
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.concatenate(
        all_probabilities,
        axis=0
    )

    # ------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
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

    # ------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------

    try:

        macro_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        macro_auc = np.nan

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    # ------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------

    per_class_precision = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    # ------------------------------------------------------------
    # Per-class ROC-AUC
    # ------------------------------------------------------------

    per_class_auc = []

    y_true_onehot = np.eye(
        NUM_CLASSES
    )[y_true]

    for class_idx in range(NUM_CLASSES):

        try:

            class_auc = roc_auc_score(
                y_true_onehot[:, class_idx],
                y_prob[:, class_idx]
            )

        except ValueError:

            class_auc = np.nan

        per_class_auc.append(
            class_auc
        )

    per_class_auc = np.array(
        per_class_auc
    )

    # ------------------------------------------------------------
    # Correct / incorrect
    # ------------------------------------------------------------

    correct = int(
        np.sum(y_true == y_pred)
    )

    incorrect = int(
        np.sum(y_true != y_pred)
    )

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print(condition_name)
    print("-" * 70)

    print(
        f"Test samples       : {len(y_true)}"
    )

    print(
        f"Correct predictions : {correct}"
    )

    print(
        f"Incorrect           : {incorrect}"
    )

    print(
        f"Accuracy            : {accuracy * 100:.4f}%"
    )

    print(
        f"Weighted Precision  : {weighted_precision:.6f}"
    )

    print(
        f"Weighted Recall     : {weighted_recall:.6f}"
    )

    print(
        f"Weighted F1         : {weighted_f1:.6f}"
    )

    print(
        f"Macro Precision     : {macro_precision:.6f}"
    )

    print(
        f"Macro Recall        : {macro_recall:.6f}"
    )

    print(
        f"Macro F1            : {macro_f1:.6f}"
    )

    print(
        f"Macro ROC-AUC       : {macro_auc:.6f}"
    )

    print()
    print("Per-class metrics:")

    for i, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name:<12} "
            f"P={per_class_precision[i]:.4f}  "
            f"R={per_class_recall[i]:.4f}  "
            f"F1={per_class_f1[i]:.4f}  "
            f"AUC={per_class_auc[i]:.4f}"
        )

    return {

        "condition": condition_name,

        "accuracy": accuracy,

        "weighted_precision":
            weighted_precision,

        "weighted_recall":
            weighted_recall,

        "weighted_f1":
            weighted_f1,

        "macro_precision":
            macro_precision,

        "macro_recall":
            macro_recall,

        "macro_f1":
            macro_f1,

        "macro_auc":
            macro_auc,

        "correct": correct,

        "incorrect": incorrect,

        "y_true": y_true,

        "y_pred": y_pred,

        "y_prob": y_prob,

        "confusion_matrix": cm,

        "per_class_precision":
            per_class_precision,

        "per_class_recall":
            per_class_recall,

        "per_class_f1":
            per_class_f1,

        "per_class_auc":
            per_class_auc
    }


# ================================================================
# 8. EVALUATE ALL THREE CONDITIONS
# ================================================================

conditions = [

    (
        "Row-wise Ordering (Original)",
        "row_wise",
        ORIGINAL_CHECKPOINT
    ),

    (
        "Column-wise Ordering",
        "column_wise",
        COLUMN_CHECKPOINT
    ),

    (
        "Alternative Ordering",
        "alternative",
        ALTERNATIVE_CHECKPOINT
    )
]


results = {}


for display_name, ordering, checkpoint in conditions:

    model = HybridCNNBiLSTMAttention(
        ordering=ordering,
        num_classes=NUM_CLASSES
    )

    model = load_checkpoint(
        model,
        checkpoint
    )

    result = evaluate_model(
        model,
        test_loader,
        CLASS_NAMES,
        display_name
    )

    results[ordering] = result


# ================================================================
# 9. OVERALL COMPARISON TABLE
# ================================================================

comparison_rows = []

for ordering, result in results.items():

    comparison_rows.append({

        "Ordering":
            result["condition"],

        "Accuracy (%)":
            result["accuracy"] * 100,

        "Weighted Precision":
            result["weighted_precision"],

        "Weighted Recall":
            result["weighted_recall"],

        "Weighted F1":
            result["weighted_f1"],

        "Macro Precision":
            result["macro_precision"],

        "Macro Recall":
            result["macro_recall"],

        "Macro F1":
            result["macro_f1"],

        "Macro ROC-AUC":
            result["macro_auc"],

        "Correct":
            result["correct"],

        "Incorrect":
            result["incorrect"]
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ================================================================
# 10. DELTA FROM ORIGINAL
# ================================================================

original = results["row_wise"]

comparison_df["Delta Accuracy (percentage points)"] = (
    comparison_df["Accuracy (%)"]
    - original["accuracy"] * 100
)

comparison_df["Delta Macro Precision"] = (
    comparison_df["Macro Precision"]
    - original["macro_precision"]
)

comparison_df["Delta Macro Recall"] = (
    comparison_df["Macro Recall"]
    - original["macro_recall"]
)

comparison_df["Delta Macro F1"] = (
    comparison_df["Macro F1"]
    - original["macro_f1"]
)

comparison_df["Delta Macro ROC-AUC"] = (
    comparison_df["Macro ROC-AUC"]
    - original["macro_auc"]
)


comparison_csv = os.path.join(
    OUTPUT_DIR,
    "spatial_ordering_overall_comparison.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)


# ================================================================
# 11. PER-CLASS TABLE
# ================================================================

per_class_rows = []

for ordering, result in results.items():

    for class_idx, class_name in enumerate(
        CLASS_NAMES
    ):

        per_class_rows.append({

            "Ordering":
                result["condition"],

            "Class":
                class_name,

            "Precision":
                result["per_class_precision"][class_idx],

            "Recall":
                result["per_class_recall"][class_idx],

            "F1":
                result["per_class_f1"][class_idx],

            "ROC-AUC":
                result["per_class_auc"][class_idx]
        })


per_class_df = pd.DataFrame(
    per_class_rows
)

per_class_csv = os.path.join(
    OUTPUT_DIR,
    "spatial_ordering_per_class_metrics.csv"
)

per_class_df.to_csv(
    per_class_csv,
    index=False
)


# ================================================================
# 12. CONFUSION MATRIX CSV FILES
# ================================================================

for ordering, result in results.items():

    safe_name = ordering.replace(
        " ",
        "_"
    )

    cm_df = pd.DataFrame(
        result["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_confusion_matrix.csv"
        )
    )


# ================================================================
# 13. DETAILED CLASSIFICATION REPORTS
# ================================================================

for ordering, result in results.items():

    safe_name = ordering.replace(
        " ",
        "_"
    )

    report = classification_report(
        result["y_true"],
        result["y_pred"],
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0
    )

    report_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report)


# ================================================================
# 14. PUBLICATION-QUALITY STYLE
# ================================================================

plt.rcParams.update({

    "font.family": "DejaVu Sans",

    "font.size": 11,

    "axes.titlesize": 14,

    "axes.labelsize": 12,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    "legend.fontsize": 10,

    "figure.dpi": 150,

    "savefig.dpi": 600,

    "axes.spines.top": False,

    "axes.spines.right": False,

    "axes.linewidth": 0.8,

    "xtick.major.width": 0.8,

    "ytick.major.width": 0.8
})


# Research-paper palette
COLORS = {

    "row_wise": "#1F4E79",       # deep academic blue

    "column_wise": "#2A9D8F",    # restrained teal

    "alternative": "#E76F51"     # warm coral
}


# ================================================================
# 15. CONFUSION MATRIX FUNCTION
# ================================================================

def plot_confusion_matrix(
    cm,
    title,
    filename
):

    fig, ax = plt.subplots(
        figsize=(7.0, 6.2)
    )

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.ax.tick_params(
        labelsize=9
    )

    ax.set(

        xticks=np.arange(
            NUM_CLASSES
        ),

        yticks=np.arange(
            NUM_CLASSES
        ),

        xticklabels=CLASS_NAMES,

        yticklabels=CLASS_NAMES,

        xlabel="Predicted class",

        ylabel="True class",

        title=title
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    threshold = cm.max() / 2.0

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            value = cm[i, j]

            ax.text(

                j,
                i,

                f"{value:,}",

                ha="center",

                va="center",

                fontsize=11,

                fontweight="bold",

                color=(
                    "white"
                    if value > threshold
                    else "#1F2937"
                )
            )

    fig.tight_layout()

    fig.savefig(
        filename,
        bbox_inches="tight"
    )

    plt.close(fig)


# ================================================================
# 16. INDIVIDUAL CONFUSION MATRICES
# ================================================================

for ordering, result in results.items():

    safe_name = ordering.replace(
        " ",
        "_"
    )

    plot_confusion_matrix(

        result["confusion_matrix"],

        result["condition"],

        os.path.join(
            OUTPUT_DIR,
            f"{safe_name}_confusion_matrix.png"
        )
    )


# ================================================================
# 17. COMBINED CONFUSION MATRICES
# ================================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5.3)
)

for ax, (ordering, result) in zip(
    axes,
    results.items()
):

    cm = result["confusion_matrix"]

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    ax.set(

        xticks=np.arange(
            NUM_CLASSES
        ),

        yticks=np.arange(
            NUM_CLASSES
        ),

        xticklabels=CLASS_NAMES,

        yticklabels=CLASS_NAMES,

        title=result["condition"]
    )

    ax.tick_params(
        axis="x",
        rotation=35
    )

    threshold = cm.max() / 2.0

    for i in range(
        NUM_CLASSES
    ):

        for j in range(
            NUM_CLASSES
        ):

            value = cm[i, j]

            ax.text(

                j,
                i,

                f"{value:,}",

                ha="center",

                va="center",

                fontsize=9,

                fontweight="bold",

                color=(
                    "white"
                    if value > threshold
                    else "#1F2937"
                )
            )

axes[0].set_ylabel(
    "True class"
)

axes[1].set_xlabel(
    "Predicted class"
)

fig.suptitle(
    "Confusion Matrices for Spatial Ordering Ablation",
    fontsize=16,
    fontweight="bold",
    y=1.02
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "all_orderings_confusion_matrices.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ================================================================
# 18. OVERALL PERFORMANCE COMPARISON
# ================================================================

labels = [
    "Row-wise\n(Original)",
    "Column-wise",
    "Alternative"
]

accuracy_values = [
    results["row_wise"]["accuracy"] * 100,
    results["column_wise"]["accuracy"] * 100,
    results["alternative"]["accuracy"] * 100
]

f1_values = [
    results["row_wise"]["weighted_f1"] * 100,
    results["column_wise"]["weighted_f1"] * 100,
    results["alternative"]["weighted_f1"] * 100
]

macro_f1_values = [
    results["row_wise"]["macro_f1"] * 100,
    results["column_wise"]["macro_f1"] * 100,
    results["alternative"]["macro_f1"] * 100
]


x = np.arange(
    len(labels)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars1 = ax.bar(
    x - width,
    accuracy_values,
    width,
    label="Accuracy",
    color=COLORS["row_wise"],
    edgecolor="white",
    linewidth=0.8
)

bars2 = ax.bar(
    x,
    f1_values,
    width,
    label="Weighted F1",
    color=COLORS["column_wise"],
    edgecolor="white",
    linewidth=0.8
)

bars3 = ax.bar(
    x + width,
    macro_f1_values,
    width,
    label="Macro F1",
    color=COLORS["alternative"],
    edgecolor="white",
    linewidth=0.8
)


ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_ylabel(
    "Performance (%)"
)

ax.set_title(
    "Performance Comparison Across Spatial Orderings",
    fontweight="bold"
)

ax.set_ylim(
    95,
    100.2
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)


def add_bar_labels(
    bars
):

    for bar in bars:

        height = bar.get_height()

        ax.text(

            bar.get_x()
            + bar.get_width() / 2,

            height + 0.05,

            f"{height:.2f}",

            ha="center",

            va="bottom",

            fontsize=8.5,

            fontweight="bold"
        )


add_bar_labels(bars1)
add_bar_labels(bars2)
add_bar_labels(bars3)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "overall_performance_comparison.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ================================================================
# 19. MACRO METRIC COMPARISON
# ================================================================

macro_metrics = [

    "Macro Precision",

    "Macro Recall",

    "Macro F1",

    "Macro ROC-AUC"
]

macro_data = {

    "Macro Precision": [
        results["row_wise"]["macro_precision"],
        results["column_wise"]["macro_precision"],
        results["alternative"]["macro_precision"]
    ],

    "Macro Recall": [
        results["row_wise"]["macro_recall"],
        results["column_wise"]["macro_recall"],
        results["alternative"]["macro_recall"]
    ],

    "Macro F1": [
        results["row_wise"]["macro_f1"],
        results["column_wise"]["macro_f1"],
        results["alternative"]["macro_f1"]
    ],

    "Macro ROC-AUC": [
        results["row_wise"]["macro_auc"],
        results["column_wise"]["macro_auc"],
        results["alternative"]["macro_auc"]
    ]
}


fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    len(labels)
)

width = 0.19

metric_colors = [
    "#1F4E79",
    "#2A9D8F",
    "#E76F51",
    "#6C5B7B"
]

for i, metric in enumerate(
    macro_metrics
):

    values = np.array(
        macro_data[metric]
    ) * 100

    bars = ax.bar(

        x
        + (i - 1.5) * width,

        values,

        width,

        label=metric,

        color=metric_colors[i],

        edgecolor="white",

        linewidth=0.7
    )


ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.set_ylabel(
    "Score (%)"
)

ax.set_title(
    "Macro-Level Performance Across Spatial Orderings",
    fontweight="bold"
)

ax.set_ylim(
    95,
    100.2
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25
)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.17)
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "macro_metrics_comparison.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ================================================================
# 20. CLASS-WISE F1 COMPARISON
# ================================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    NUM_CLASSES
)

width = 0.25

row_f1 = (
    results["row_wise"]["per_class_f1"]
    * 100
)

column_f1 = (
    results["column_wise"]["per_class_f1"]
    * 100
)

alternative_f1 = (
    results["alternative"]["per_class_f1"]
    * 100
)

ax.bar(
    x - width,
    row_f1,
    width,
    label="Row-wise (Original)",
    color=COLORS["row_wise"],
    edgecolor="white"
)

ax.bar(
    x,
    column_f1,
    width,
    label="Column-wise",
    color=COLORS["column_wise"],
    edgecolor="white"
)

ax.bar(
    x + width,
    alternative_f1,
    width,
    label="Alternative",
    color=COLORS["alternative"],
    edgecolor="white"
)

ax.set_xticks(x)

ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_ylabel(
    "F1-score (%)"
)

ax.set_title(
    "Class-wise F1-score Across Spatial Orderings",
    fontweight="bold"
)

ax.set_ylim(
    95,
    100.2
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "classwise_f1_comparison.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ================================================================
# 21. ROC CURVES
# ================================================================

roc_colors = [
    "#1F4E79",
    "#2A9D8F",
    "#E76F51"
]

fig, ax = plt.subplots(
    figsize=(8, 7)
)

for (
    (ordering, result),
    color
) in zip(
    results.items(),
    roc_colors
):

    y_true = result["y_true"]
    y_prob = result["y_prob"]

    y_true_onehot = np.eye(
        NUM_CLASSES
    )[y_true]

    # Micro-average ROC
    fpr, tpr, _ = roc_curve(
        y_true_onehot.ravel(),
        y_prob.ravel()
    )

    micro_auc = auc(
        fpr,
        tpr
    )

    ax.plot(

        fpr,

        tpr,

        linewidth=2.2,

        color=color,

        label=(
            f"{result['condition']} "
            f"(AUC = {result['macro_auc']:.4f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    color="#9CA3AF",
    label="Chance"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curves Across Spatial Orderings",
    fontweight="bold"
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.01
)

ax.grid(
    linestyle="--",
    linewidth=0.5,
    alpha=0.25
)

ax.legend(
    frameon=False,
    loc="lower right"
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves_all_orderings.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ================================================================
# 22. SAVE ALL RESULTS AS JSON
# ================================================================

json_results = {}

for ordering, result in results.items():

    json_results[ordering] = {

        "condition":
            result["condition"],

        "accuracy":
            float(result["accuracy"]),

        "weighted_precision":
            float(result["weighted_precision"]),

        "weighted_recall":
            float(result["weighted_recall"]),

        "weighted_f1":
            float(result["weighted_f1"]),

        "macro_precision":
            float(result["macro_precision"]),

        "macro_recall":
            float(result["macro_recall"]),

        "macro_f1":
            float(result["macro_f1"]),

        "macro_auc":
            float(result["macro_auc"]),

        "correct":
            int(result["correct"]),

        "incorrect":
            int(result["incorrect"]),

        "per_class_precision":
            result["per_class_precision"].tolist(),

        "per_class_recall":
            result["per_class_recall"].tolist(),

        "per_class_f1":
            result["per_class_f1"].tolist(),

        "per_class_auc":
            result["per_class_auc"].tolist(),

        "confusion_matrix":
            result["confusion_matrix"].tolist()
    }


json_path = os.path.join(
    OUTPUT_DIR,
    "spatial_ordering_complete_results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_results,
        f,
        indent=4
    )


# ================================================================
# 23. FINAL CONSOLE SUMMARY
# ================================================================

print()
print("=" * 80)
print("FINAL SPATIAL ORDERING ABLATION RESULTS")
print("=" * 80)

for ordering, result in results.items():

    print()
    print(result["condition"])

    print(
        f"  Accuracy       : "
        f"{result['accuracy'] * 100:.4f}%"
    )

    print(
        f"  Weighted F1    : "
        f"{result['weighted_f1']:.6f}"
    )

    print(
        f"  Macro Precision: "
        f"{result['macro_precision']:.6f}"
    )

    print(
        f"  Macro Recall   : "
        f"{result['macro_recall']:.6f}"
    )

    print(
        f"  Macro F1       : "
        f"{result['macro_f1']:.6f}"
    )

    print(
        f"  Macro ROC-AUC  : "
        f"{result['macro_auc']:.6f}"
    )

print()
print("=" * 80)
print("DELTA FROM ORIGINAL ROW-WISE ORDERING")
print("=" * 80)

for ordering, result in results.items():

    delta_acc = (
        result["accuracy"]
        - original["accuracy"]
    )

    delta_f1 = (
        result["macro_f1"]
        - original["macro_f1"]
    )

    delta_auc = (
        result["macro_auc"]
        - original["macro_auc"]
    )

    print()
    print(result["condition"])

    print(
        f"  Δ Accuracy   : "
        f"{delta_acc * 100:+.4f} percentage points"
    )

    print(
        f"  Δ Macro F1   : "
        f"{delta_f1:+.6f}"
    )

    print(
        f"  Δ Macro AUC  : "
        f"{delta_auc:+.6f}"
    )


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    f"Overall comparison:\n"
    f"{comparison_csv}"
)

print(
    f"\nPer-class metrics:\n"
    f"{per_class_csv}"
)

print(
    f"\nOutput directory:\n"
    f"{OUTPUT_DIR}"
)

print()
print("Evaluation completed successfully.")
print("=" * 80)