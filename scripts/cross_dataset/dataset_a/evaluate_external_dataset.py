# ============================================================
# CROSS-DATASET EXTERNAL VALIDATION
#
# Experiment 1:
#   Model trained on Dataset A -> Test on Dataset B
#
# Experiment 2:
#   Model trained on Dataset B -> Test on Dataset A
#
# NO TRAINING
# NO FINE-TUNING
# NO AUGMENTATION
# ============================================================

import os
import csv
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ============================================================
# 1. EDIT THESE FOUR PATHS
# ============================================================

# ------------------------------------------------------------
# Trained Dataset-A model
# ------------------------------------------------------------

MODEL_A_PATH = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\training_outputs_proposed_model_orgnl\training_outputs"
    r"\hybrid_cnn_bilstm_attention_final.pth"
)


# ------------------------------------------------------------
# Trained Dataset-B model
#
# This should be the BEST checkpoint produced by the B
# training code.
# ------------------------------------------------------------

MODEL_B_PATH = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B_training_results"
    r"\hybrid_cnn_bilstm_attention_best_B.pth"
)


# ------------------------------------------------------------
# Dataset A PREPROCESSED TEST SET
# ------------------------------------------------------------

TEST_A_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\preprocessed_testt"
)


# ------------------------------------------------------------
# Dataset B PREPROCESSED TEST SET
# ------------------------------------------------------------

TEST_B_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B"
    r"\preprocessed_test_B"
)


# ============================================================
# 2. OUTPUT DIRECTORY
# ============================================================

SAVE_ROOT = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\cross_dataset_external_validation"
)

os.makedirs(
    SAVE_ROOT,
    exist_ok=True
)


# ============================================================
# 3. BASIC SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

INPUT_SIZE = (256, 256)

NUM_CLASSES = 4

BATCH_SIZE = 16

NUM_WORKERS = 4

SEED = 42


# ============================================================
# 4. CLASS MAPPING
# ============================================================

# IMPORTANT:
#
# 0 = glioma
# 1 = meningioma
# 2 = notumor
# 3 = pituitary

EXPECTED_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

EXPECTED_CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

DISPLAY_NAMES = [
    "Glioma",
    "Meningioma",
    "No tumor",
    "Pituitary"
]


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 6. TEST TRANSFORM
# ============================================================

# MUST match the transform used during training.
#
# No augmentation is applied.

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
# 7. MODEL ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        use_pool=False
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

        if use_pool:

            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

        else:

            self.pool = nn.Identity()


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
        num_classes=NUM_CLASSES,
        lstm_hidden=128,
        lstm_layers=1
    ):

        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.hidden_size = lstm_hidden

        self.num_layers = lstm_layers

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

        x = self.cnn(x)

        b, c, h, w = x.size()

        x = x.view(
            b,
            h,
            c * w
        )

        lstm_out, _ = self.bilstm(x)

        context, attn_weights = (
            self.attention(lstm_out)
        )

        out = self.fc(context)

        return out, attn_weights


# ============================================================
# 8. LOAD CHECKPOINT SAFELY
# ============================================================

def load_model(checkpoint_path):

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Check checkpoint class mapping
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "class_names" in checkpoint:

            saved_classes = list(
                checkpoint["class_names"]
            )

            print(
                "Checkpoint classes:",
                saved_classes
            )

            if saved_classes != EXPECTED_CLASSES:

                raise RuntimeError(
                    "\nCHECKPOINT CLASS ORDER ERROR!\n"
                    f"Expected: {EXPECTED_CLASSES}\n"
                    f"Found:    {saved_classes}"
                )

        if "class_to_idx" in checkpoint:

            saved_mapping = checkpoint[
                "class_to_idx"
            ]

            print(
                "Checkpoint class mapping:",
                saved_mapping
            )

            if saved_mapping != EXPECTED_CLASS_TO_IDX:

                raise RuntimeError(
                    "\nCHECKPOINT CLASS MAPPING ERROR!\n"
                    f"Expected: "
                    f"{EXPECTED_CLASS_TO_IDX}\n"
                    f"Found: "
                    f"{saved_mapping}"
                )


    model = HybridCNNBiLSTMAttention(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    # --------------------------------------------------------
    # Support both:
    #
    # torch.save(model.state_dict())
    #
    # and:
    #
    # torch.save({
    #    "model_state_dict": ...
    # })
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    model.load_state_dict(
        state_dict
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# 9. LOAD EXTERNAL TEST DATASET
# ============================================================

def load_test_dataset(test_directory):

    dataset = datasets.ImageFolder(

        test_directory,

        transform=test_transform
    )


    print("\nTest dataset:")
    print(test_directory)

    print(
        "Detected classes:",
        dataset.classes
    )

    print(
        "Detected mapping:",
        dataset.class_to_idx
    )


    # --------------------------------------------------------
    # STRICT mapping verification
    # --------------------------------------------------------

    if dataset.classes != EXPECTED_CLASSES:

        raise RuntimeError(

            "\nTEST DATASET CLASS ORDER ERROR!\n"

            f"Expected:\n"
            f"{EXPECTED_CLASSES}\n\n"

            f"Found:\n"
            f"{dataset.classes}"

        )


    if (
        dataset.class_to_idx
        != EXPECTED_CLASS_TO_IDX
    ):

        raise RuntimeError(

            "\nTEST DATASET CLASS MAPPING ERROR!\n"

            f"Expected:\n"
            f"{EXPECTED_CLASS_TO_IDX}\n\n"

            f"Found:\n"
            f"{dataset.class_to_idx}"

        )


    print(
        "Test-set class mapping verification: PASS"
    )


    # --------------------------------------------------------
    # Class counts
    # --------------------------------------------------------

    print(
        f"\nTotal test images: {len(dataset):,}"
    )


    targets = np.array(
        dataset.targets
    )


    print("\nTest class distribution:")

    for idx, class_name in enumerate(
        EXPECTED_CLASSES
    ):

        count = int(
            np.sum(targets == idx)
        )

        print(
            f"{idx} = "
            f"{DISPLAY_NAMES[idx]:<12}: "
            f"{count:,}"
        )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=(
            NUM_WORKERS > 0
        )
    )


    return dataset, loader


# ============================================================
# 10. SPECIFICITY
# ============================================================

def calculate_specificity(cm):

    specificity = []

    for i in range(NUM_CLASSES):

        TP = cm[i, i]

        FN = cm[i, :].sum() - TP

        FP = cm[:, i].sum() - TP

        TN = cm.sum() - TP - FN - FP

        if (TN + FP) == 0:

            spec = np.nan

        else:

            spec = TN / (TN + FP)

        specificity.append(spec)

    return np.array(specificity)


# ============================================================
# 11. CONFUSION MATRIX FIGURE
# ============================================================

def save_confusion_matrix(
    cm,
    experiment_name,
    save_directory
):

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set(

        xticks=np.arange(NUM_CLASSES),

        yticks=np.arange(NUM_CLASSES),

        xticklabels=DISPLAY_NAMES,

        yticklabels=DISPLAY_NAMES,

        ylabel="True class",

        xlabel="Predicted class",

        title=(
            f"Confusion Matrix\n"
            f"{experiment_name}"
        )
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor"
    )


    threshold = (
        cm.max() / 2.0
        if cm.size
        else 0
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(

                j,
                i,
                str(cm[i, j]),

                ha="center",
                va="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )


    fig.tight_layout()


    path = os.path.join(

        save_directory,

        "confusion_matrix.png"
    )


    fig.savefig(

        path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close(fig)


# ============================================================
# 12. ROC CURVES
# ============================================================

def save_roc_curves(
    y_true,
    probabilities,
    experiment_name,
    save_directory
):

    y_binary = label_binarize(

        y_true,

        classes=np.arange(NUM_CLASSES)
    )


    fig, ax = plt.subplots(
        figsize=(8, 7)
    )


    roc_rows = []


    for i in range(NUM_CLASSES):

        # Need both positive and negative examples
        # for class-wise ROC.

        if len(
            np.unique(y_binary[:, i])
        ) < 2:

            print(
                f"WARNING: ROC unavailable for "
                f"{DISPLAY_NAMES[i]}."
            )

            continue


        fpr, tpr, _ = roc_curve(

            y_binary[:, i],

            probabilities[:, i]
        )


        class_auc = auc(
            fpr,
            tpr
        )


        ax.plot(

            fpr,
            tpr,

            label=(
                f"{DISPLAY_NAMES[i]} "
                f"(AUC={class_auc:.4f})"
            )
        )


        roc_rows.append({

            "class":
                DISPLAY_NAMES[i],

            "auc":
                class_auc
        })


    ax.plot(

        [0, 1],

        [0, 1],

        linestyle="--",

        label="Chance"
    )


    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        f"ROC Curves\n{experiment_name}"
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()


    fig.savefig(

        os.path.join(
            save_directory,
            "roc_curves.png"
        ),

        dpi=300,

        bbox_inches="tight"
    )


    plt.close(fig)


    pd.DataFrame(
        roc_rows
    ).to_csv(

        os.path.join(
            save_directory,
            "classwise_auc.csv"
        ),

        index=False
    )


# ============================================================
# 13. MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_external_dataset(

    model_path,

    test_directory,

    experiment_name,

    source_dataset,

    target_dataset

):

    print("\n")
    print("=" * 75)
    print(experiment_name)
    print("=" * 75)

    print(
        f"Training dataset : {source_dataset}"
    )

    print(
        f"External test    : {target_dataset}"
    )


    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    save_directory = os.path.join(

        SAVE_ROOT,

        experiment_name.replace(
            " ",
            "_"
        ).replace(
            "→",
            "to"
        )
    )


    os.makedirs(
        save_directory,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        model_path
    )


    # --------------------------------------------------------
    # Load external test data
    # --------------------------------------------------------

    dataset, loader = load_test_dataset(
        test_directory
    )


    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    all_true = []

    all_pred = []

    all_probabilities = []

    all_confidences = []


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    model.eval()


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs, _ = model(
                images
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            confidence, predictions = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )


            all_true.extend(
                labels.cpu().numpy()
            )


            all_pred.extend(
                predictions.cpu().numpy()
            )


            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


            all_confidences.extend(
                confidence.cpu().numpy()
            )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    y_true = np.asarray(
        all_true
    )

    y_pred = np.asarray(
        all_pred
    )

    probabilities = np.asarray(
        all_probabilities
    )

    confidences = np.asarray(
        all_confidences
    )


    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(

        y_true,
        y_pred,

        labels=np.arange(NUM_CLASSES)
    )


    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity = calculate_specificity(
        cm
    )


    macro_specificity = np.nanmean(
        specificity
    )


    # --------------------------------------------------------
    # Per-class precision/recall/F1
    # --------------------------------------------------------

    class_precision = precision_score(

        y_true,
        y_pred,

        labels=np.arange(NUM_CLASSES),

        average=None,

        zero_division=0
    )


    class_recall = recall_score(

        y_true,
        y_pred,

        labels=np.arange(NUM_CLASSES),

        average=None,

        zero_division=0
    )


    class_f1 = f1_score(

        y_true,
        y_pred,

        labels=np.arange(NUM_CLASSES),

        average=None,

        zero_division=0
    )


    # --------------------------------------------------------
    # AUC
    # --------------------------------------------------------

    y_binary = label_binarize(

        y_true,

        classes=np.arange(NUM_CLASSES)
    )


    class_auc_values = []


    for i in range(NUM_CLASSES):

        if len(
            np.unique(y_binary[:, i])
        ) < 2:

            class_auc_values.append(
                np.nan
            )

        else:

            class_auc_values.append(

                roc_auc_score(

                    y_binary[:, i],

                    probabilities[:, i]

                )

            )


    try:

        macro_auc = roc_auc_score(

            y_binary,

            probabilities,

            average="macro",

            multi_class="ovr"
        )

    except ValueError:

        macro_auc = np.nan


    try:

        weighted_auc = roc_auc_score(

            y_binary,

            probabilities,

            average="weighted",

            multi_class="ovr"
        )

    except ValueError:

        weighted_auc = np.nan


    # ========================================================
    # 14. SAVE PER-IMAGE PREDICTIONS
    # ========================================================

    prediction_rows = []


    # ImageFolder and shuffle=False preserve order.

    for index, (path, _) in enumerate(
        dataset.samples
    ):

        true_idx = int(
            y_true[index]
        )

        pred_idx = int(
            y_pred[index]
        )


        row = {

            "image":
                os.path.basename(path),

            "path":
                path,

            "true_index":
                true_idx,

            "true_class":
                DISPLAY_NAMES[true_idx],

            "predicted_index":
                pred_idx,

            "predicted_class":
                DISPLAY_NAMES[pred_idx],

            "confidence":
                float(confidences[index]),

            "correct":
                bool(
                    true_idx == pred_idx
                )

        }


        # Save probability for every class

        for class_idx in range(
            NUM_CLASSES
        ):

            row[
                f"probability_"
                f"{EXPECTED_CLASSES[class_idx]}"
            ] = float(
                probabilities[
                    index,
                    class_idx
                ]
            )


        prediction_rows.append(
            row
        )


    prediction_df = pd.DataFrame(
        prediction_rows
    )


    prediction_df.to_csv(

        os.path.join(
            save_directory,
            "per_image_predictions.csv"
        ),

        index=False
    )


    # --------------------------------------------------------
    # Save only misclassified images information
    # --------------------------------------------------------

    prediction_df[

        prediction_df["correct"] == False

    ].to_csv(

        os.path.join(
            save_directory,
            "misclassified_cases.csv"
        ),

        index=False
    )


    # ========================================================
    # 15. CLASS-WISE METRICS
    # ========================================================

    class_rows = []


    for i in range(NUM_CLASSES):

        support = int(
            np.sum(y_true == i)
        )


        class_rows.append({

            "class":
                DISPLAY_NAMES[i],

            "support":
                support,

            "precision":
                class_precision[i],

            "sensitivity_recall":
                class_recall[i],

            "specificity":
                specificity[i],

            "f1_score":
                class_f1[i],

            "auc_ovr":
                class_auc_values[i]

        })


    classwise_df = pd.DataFrame(
        class_rows
    )


    classwise_df.to_csv(

        os.path.join(
            save_directory,
            "classwise_metrics.csv"
        ),

        index=False
    )


    # ========================================================
    # 16. CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_true,
        y_pred,

        labels=np.arange(NUM_CLASSES),

        target_names=DISPLAY_NAMES,

        zero_division=0,

        digits=4
    )


    with open(

        os.path.join(
            save_directory,
            "classification_report.txt"
        ),

        "w",

        encoding="utf-8"

    ) as file:

        file.write(
            report
        )


    # ========================================================
    # 17. SAVE CONFUSION MATRIX CSV
    # ========================================================

    cm_df = pd.DataFrame(

        cm,

        index=[
            f"True_{name}"
            for name in DISPLAY_NAMES
        ],

        columns=[
            f"Pred_{name}"
            for name in DISPLAY_NAMES
        ]
    )


    cm_df.to_csv(

        os.path.join(
            save_directory,
            "confusion_matrix.csv"
        )
    )


    # ========================================================
    # 18. SAVE CONFUSION MATRIX FIGURE
    # ========================================================

    save_confusion_matrix(

        cm,

        experiment_name,

        save_directory
    )


    # ========================================================
    # 19. SAVE ROC CURVES
    # ========================================================

    save_roc_curves(

        y_true,

        probabilities,

        experiment_name,

        save_directory
    )


    # ========================================================
    # 20. OVERALL METRICS
    # ========================================================

    overall_results = {

        "experiment":
            experiment_name,

        "trained_on":
            source_dataset,

        "externally_tested_on":
            target_dataset,

        "number_of_test_images":
            len(y_true),

        "correct_predictions":
            int(
                np.sum(
                    y_true == y_pred
                )
            ),

        "incorrect_predictions":
            int(
                np.sum(
                    y_true != y_pred
                )
            ),

        "accuracy":
            accuracy,

        "macro_precision":
            macro_precision,

        "macro_recall_sensitivity":
            macro_recall,

        "macro_specificity":
            macro_specificity,

        "macro_f1":
            macro_f1,

        "weighted_precision":
            weighted_precision,

        "weighted_recall":
            weighted_recall,

        "weighted_f1":
            weighted_f1,

        "macro_auc_ovr":
            macro_auc,

        "weighted_auc_ovr":
            weighted_auc

    }


    pd.DataFrame(
        [overall_results]
    ).to_csv(

        os.path.join(
            save_directory,
            "overall_metrics.csv"
        ),

        index=False
    )


    # ========================================================
    # 21. PRINT RESULTS
    # ========================================================

    print("\n" + "-" * 75)

    print("EXTERNAL VALIDATION RESULTS")

    print("-" * 75)


    print(
        f"Test images        : "
        f"{len(y_true):,}"
    )

    print(
        f"Correct            : "
        f"{np.sum(y_true == y_pred):,}"
    )

    print(
        f"Misclassified      : "
        f"{np.sum(y_true != y_pred):,}"
    )


    print(
        f"\nAccuracy           : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision    : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Sensitivity  : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro Specificity  : "
        f"{macro_specificity:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Macro AUC (OvR)    : "
        f"{macro_auc:.4f}"
    )


    print("\nClass-wise results:")


    print(
        classwise_df.to_string(
            index=False
        )
    )


    print("\nConfusion matrix:")

    print(cm)


    print("\nClassification report:")

    print(report)


    print(
        "\nResults saved to:"
    )

    print(
        save_directory
    )


    return overall_results


# ============================================================
# 22. RUN BOTH CROSS-DATASET EXPERIMENTS
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 75)

    print(
        "BIDIRECTIONAL CROSS-DATASET "
        "EXTERNAL VALIDATION"
    )

    print("=" * 75)


    print(
        f"\nDevice: {DEVICE}"
    )


    # --------------------------------------------------------
    # Verify all required paths BEFORE starting
    # --------------------------------------------------------

    required_paths = {

        "Dataset A model":
            MODEL_A_PATH,

        "Dataset B model":
            MODEL_B_PATH,

        "Dataset A test":
            TEST_A_DIR,

        "Dataset B test":
            TEST_B_DIR
    }


    for description, path in required_paths.items():

        if not os.path.exists(path):

            raise FileNotFoundError(

                f"\n{description} was not found:\n"
                f"{path}"

            )


    print(
        "\nAll required paths found: PASS"
    )


    # ========================================================
    # EXPERIMENT 1
    #
    # MODEL A -> DATASET B
    # ========================================================

    result_A_to_B = evaluate_external_dataset(

        model_path=
            MODEL_A_PATH,

        test_directory=
            TEST_B_DIR,

        experiment_name=
            "A_to_B",

        source_dataset=
            "Dataset A",

        target_dataset=
            "Dataset B"

    )


    # ========================================================
    # EXPERIMENT 2
    #
    # MODEL B -> DATASET A
    # ========================================================

    result_B_to_A = evaluate_external_dataset(

        model_path=
            MODEL_B_PATH,

        test_directory=
            TEST_A_DIR,

        experiment_name=
            "B_to_A",

        source_dataset=
            "Dataset B",

        target_dataset=
            "Dataset A"

    )


    # ========================================================
    # 23. COMBINED COMPARISON TABLE
    # ========================================================

    comparison_df = pd.DataFrame([

        result_A_to_B,

        result_B_to_A

    ])


    comparison_path = os.path.join(

        SAVE_ROOT,

        "cross_dataset_comparison.csv"
    )


    comparison_df.to_csv(

        comparison_path,

        index=False
    )


    # --------------------------------------------------------
    # Thesis-friendly percentage table
    # --------------------------------------------------------

    thesis_table = pd.DataFrame({

        "Experiment": [
            "A → B",
            "B → A"
        ],

        "Accuracy (%)": [

            result_A_to_B[
                "accuracy"
            ] * 100,

            result_B_to_A[
                "accuracy"
            ] * 100

        ],

        "Precision (%)": [

            result_A_to_B[
                "macro_precision"
            ] * 100,

            result_B_to_A[
                "macro_precision"
            ] * 100

        ],

        "Sensitivity (%)": [

            result_A_to_B[
                "macro_recall_sensitivity"
            ] * 100,

            result_B_to_A[
                "macro_recall_sensitivity"
            ] * 100

        ],

        "Specificity (%)": [

            result_A_to_B[
                "macro_specificity"
            ] * 100,

            result_B_to_A[
                "macro_specificity"
            ] * 100

        ],

        "F1-score (%)": [

            result_A_to_B[
                "macro_f1"
            ] * 100,

            result_B_to_A[
                "macro_f1"
            ] * 100

        ],

        "AUC": [

            result_A_to_B[
                "macro_auc_ovr"
            ],

            result_B_to_A[
                "macro_auc_ovr"
            ]

        ]

    })


    thesis_table = thesis_table.round(2)


    thesis_table_path = os.path.join(

        SAVE_ROOT,

        "thesis_external_validation_table.csv"
    )


    thesis_table.to_csv(

        thesis_table_path,

        index=False
    )


    print("\n")
    print("=" * 75)

    print(
        "BOTH CROSS-DATASET EVALUATIONS COMPLETE"
    )

    print("=" * 75)


    print("\nFINAL COMPARISON:\n")

    print(
        thesis_table.to_string(
            index=False
        )
    )


    print(
        "\nComparison CSV:"
    )

    print(
        comparison_path
    )


    print(
        "\nThesis-ready table:"
    )

    print(
        thesis_table_path
    )


    print(
        "\nAll external-validation outputs:"
    )

    print(
        SAVE_ROOT
    )