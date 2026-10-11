# ============================================================
# DATASET B - 5-FOLD AUC RECOVERY
# ============================================================
#
# PURPOSE:
# Recover missing AUC values from EXISTING trained
# 5-fold cross-validation checkpoints.
#
# IMPORTANT:
# - NO TRAINING
# - NO RETRAINING
# - Uses existing best_model.pth files
# - Recreates the original CV splits
# - Uses the ORIGINAL MODEL ARCHITECTURE
# - Runs on CPU
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import random
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = r"C:\user\juli\thesis\Dataset\preprocessed_train"

CV_DIR = r"C:\user\juli\thesis\DatasetB_5Fold_CV"

OUTPUT_DIR = os.path.join(
    CV_DIR,
    "AUC_Recovery"
)

IMAGE_SIZE = (256, 256)

NUM_CLASSES = 4

BATCH_SIZE = 16

N_SPLITS = 5

RANDOM_SEED = 42

NUM_WORKERS = 0

# ------------------------------------------------------------
# FORCE CPU
# ------------------------------------------------------------
# We do NOT need the RTX 5090 for inference.
# CPU is intentionally used so this script can run locally.
# ------------------------------------------------------------

DEVICE = torch.device("cpu")


# ============================================================
# 3. CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}


# ============================================================
# 4. EXPECTED DATASET COUNTS
# ============================================================

EXPECTED_CLASS_COUNTS = {
    "glioma": 3018,
    "meningioma": 2183,
    "notumor": 1945,
    "pituitary": 2504
}

EXPECTED_TOTAL = 9650


# ============================================================
# 5. EXPECTED ORIGINAL FOLD ACCURACIES
# ============================================================
#
# These values come from your original CV results.
#
# They are used ONLY as a verification check.
#
# If recovered accuracy does not match these values,
# the recreated fold assignment may not match the original.
# ============================================================

EXPECTED_ACCURACY = {
    1: 0.965285,
    2: 0.969430,
    3: 0.965285,
    4: 0.973057,
    5: 0.973575
}


# ============================================================
# 6. EXPECTED EXISTING AUC VALUES
# ============================================================
#
# Folds 4 and 5 already had valid AUC values in your
# original results.
#
# These provide an important correctness check.
# ============================================================

EXPECTED_AUC = {
    4: 0.997647,
    5: 0.998473
}


# ============================================================
# 7. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


set_seed(RANDOM_SEED)


# ============================================================
# 8. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 9. PRINT HEADER
# ============================================================

print("=" * 70)
print("DATASET B - AUC RECOVERY")
print("=" * 70)

print()

print("Dataset path:")
print(DATA_DIR)

print()

print("Checkpoint path:")
print(CV_DIR)

print()

print("Output path:")
print(OUTPUT_DIR)

print()

print("Device:")
print(DEVICE)

print()

print("=" * 70)


# ============================================================
# 10. CHECK PATHS
# ============================================================

if not os.path.isdir(DATA_DIR):

    raise FileNotFoundError(
        f"\nDataset directory does not exist:\n{DATA_DIR}"
    )


if not os.path.isdir(CV_DIR):

    raise FileNotFoundError(
        f"\nCross-validation directory does not exist:\n{CV_DIR}"
    )


# ============================================================
# 11. TRANSFORMS
# ============================================================
#
# EXACTLY MATCHES ORIGINAL CV CODE
# ============================================================

transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 12. DATASET CLASS
# ============================================================
#
# IMPORTANT:
# We intentionally DO NOT sort os.listdir().
#
# The original training code used os.listdir() directly.
# Therefore the same ordering behavior is preserved here.
# ============================================================

class BrainTumorDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.root_dir = root_dir

        self.transform = transform

        self.samples = []

        self.class_to_idx = {
            "glioma": 0,
            "meningioma": 1,
            "notumor": 2,
            "pituitary": 3
        }

        # ----------------------------------------------------
        # SAME CLASS ORDER AS ORIGINAL CODE
        # ----------------------------------------------------

        for class_name, class_idx in self.class_to_idx.items():

            class_dir = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.isdir(class_dir):

                raise FileNotFoundError(
                    f"Class directory not found:\n{class_dir}"
                )

            # IMPORTANT:
            # Do NOT add sorted() here.
            for filename in os.listdir(class_dir):

                filepath = os.path.join(
                    class_dir,
                    filename
                )

                if os.path.isfile(filepath):

                    if filename.lower().endswith(
                        (
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".bmp",
                            ".tif",
                            ".tiff"
                        )
                    ):

                        self.samples.append(
                            (
                                filepath,
                                class_idx
                            )
                        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filepath, label = self.samples[index]

        image = Image.open(
            filepath
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# 13. LOAD DATASET
# ============================================================

print()
print("=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

dataset = BrainTumorDataset(
    DATA_DIR,
    transform=transform
)


# ============================================================
# 14. VERIFY TOTAL DATASET SIZE
# ============================================================

actual_total = len(dataset)

print()

print(
    f"Actual total images   : {actual_total}"
)

print(
    f"Expected total images : {EXPECTED_TOTAL}"
)


if actual_total != EXPECTED_TOTAL:

    raise RuntimeError(
        "\nDATASET SIZE MISMATCH.\n"
        f"Expected {EXPECTED_TOTAL} images, "
        f"but found {actual_total}.\n"
        "\n"
        "Do NOT continue because the fold reconstruction "
        "may no longer match the original experiment."
    )


# ============================================================
# 15. VERIFY CLASS COUNTS
# ============================================================

labels = np.array(
    [
        label
        for _, label in dataset.samples
    ]
)


print()
print("Class counts:")

dataset_verification_passed = True


for class_name, class_idx in CLASS_TO_IDX.items():

    actual_count = int(
        np.sum(labels == class_idx)
    )

    expected_count = EXPECTED_CLASS_COUNTS[
        class_name
    ]

    status = "OK"

    if actual_count != expected_count:

        status = "MISMATCH"

        dataset_verification_passed = False

    print(
        f"{class_name:<12} "
        f"Actual={actual_count:5d} "
        f"Expected={expected_count:5d} "
        f"[{status}]"
    )


if not dataset_verification_passed:

    raise RuntimeError(
        "\nCLASS COUNT MISMATCH.\n"
        "Do NOT continue."
    )


print()
print("Dataset verification PASSED.")


# ============================================================
# 16. RESIDUAL BLOCK
# ============================================================
#
# EXACT MODEL COMPONENT FROM ORIGINAL CV TRAINING CODE
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        pool=False
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

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),

                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

        self.pool = (

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

            if pool

            else nn.Identity()
        )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        out = self.pool(out)

        return out


# ============================================================
# 17. ORIGINAL HYBRID CNN + BiLSTM + ATTENTION MODEL
# ============================================================
#
# IMPORTANT:
# This version uses the ORIGINAL parameter names:
#
# self.cnn
# self.bilstm
# self.attention_linear
# self.attention_score
# self.fc
#
# This fixes the state_dict mismatch from the previous script.
# ============================================================

class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        num_classes=4
    ):

        super().__init__()

        # ====================================================
        # CNN FEATURE EXTRACTOR
        # ====================================================

        self.cnn = nn.Sequential(

            ResidualBlock(
                3,
                32,
                pool=True
            ),

            ResidualBlock(
                32,
                64,
                pool=True
            ),

            ResidualBlock(
                64,
                128,
                pool=True
            ),

            ResidualBlock(
                128,
                128,
                pool=True
            ),

            ResidualBlock(
                128,
                256,
                pool=False
            ),

            nn.AdaptiveAvgPool2d(
                (8, 8)
            )
        )


        # ====================================================
        # BiLSTM
        # ====================================================

        self.bilstm = nn.LSTM(

            input_size=256,

            hidden_size=128,

            num_layers=1,

            batch_first=True,

            bidirectional=True
        )


        # ====================================================
        # ADDITIVE ATTENTION
        # ====================================================

        self.attention_linear = nn.Linear(
            256,
            128
        )

        self.attention_score = nn.Linear(
            128,
            1
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.fc = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # ====================================================
        # CNN
        # ====================================================

        x = self.cnn(x)


        # ====================================================
        # CNN FEATURE MAP -> SEQUENCE
        # ====================================================
        #
        # CNN output:
        # batch x 256 x 8 x 8
        #
        # Convert to:
        # batch x 64 x 256
        # ====================================================

        x = x.permute(
            0,
            2,
            3,
            1
        )

        batch_size = x.size(0)

        x = x.reshape(
            batch_size,
            -1,
            256
        )


        # ====================================================
        # BiLSTM
        # ====================================================

        lstm_out, _ = self.bilstm(x)


        # ====================================================
        # ADDITIVE ATTENTION
        # ====================================================

        attention_hidden = torch.tanh(
            self.attention_linear(
                lstm_out
            )
        )

        attention_scores = self.attention_score(
            attention_hidden
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        context = torch.sum(
            attention_weights * lstm_out,
            dim=1
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        output = self.fc(
            context
        )

        return output


# ============================================================
# 18. MODEL LOADER
# ============================================================

def load_model(
    checkpoint_path
):

    print("Loading model...")

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            + checkpoint_path
        )


    # --------------------------------------------------------
    # Create EXACT original architecture
    # --------------------------------------------------------

    model = HybridCNNBiLSTMAttention(
        num_classes=NUM_CLASSES
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # The original training code saved the raw state_dict
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint,
        strict=True
    )


    model.to(
        DEVICE
    )

    model.eval()


    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# 19. VALIDATION / INFERENCE FUNCTION
# ============================================================

def run_inference(
    model,
    loader
):

    model.eval()

    all_labels = []

    all_predictions = []

    all_probabilities = []


    # --------------------------------------------------------
    # No gradients needed
    # --------------------------------------------------------

    with torch.no_grad():

        for images, labels_batch in tqdm(
            loader,
            desc="Inference",
            leave=False
        ):

            images = images.to(
                DEVICE
            )

            labels_batch = labels_batch.to(
                DEVICE
            )


            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(
                images
            )


            # ------------------------------------------------
            # Softmax probabilities
            # ------------------------------------------------

            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            # ------------------------------------------------
            # Predicted class
            # ------------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1
            )


            # ------------------------------------------------
            # Store
            # ------------------------------------------------

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# 20. RECREATE 5-FOLD SPLITS
# ============================================================

print()
print("=" * 70)
print("5-FOLD SPLITS RECREATED")
print("=" * 70)


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)


fold_splits = []


for fold, (
    train_indices,
    val_indices
) in enumerate(

    skf.split(
        np.zeros(
            len(labels)
        ),
        labels
    ),

    start=1
):

    fold_splits.append(
        (
            train_indices,
            val_indices
        )
    )

    print(
        f"Fold {fold}: "
        f"{len(val_indices)} validation images"
    )


# ============================================================
# 21. START AUC RECOVERY
# ============================================================

print()
print("=" * 70)
print("STARTING AUC RECOVERY")
print("=" * 70)


all_results = []


# ============================================================
# 22. FOLD LOOP
# ============================================================

for fold, (
    train_indices,
    val_indices
) in enumerate(
    fold_splits,
    start=1
):

    print()
    print()
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)


    # ========================================================
    # CHECKPOINT
    # ========================================================

    fold_dir = os.path.join(
        CV_DIR,
        f"fold_{fold}"
    )

    checkpoint_path = os.path.join(
        fold_dir,
        "best_model.pth"
    )


    print()
    print("Checkpoint:")
    print(checkpoint_path)


    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "\nCheckpoint missing:\n"
            + checkpoint_path
            + "\n\n"
            "Make sure all five fold checkpoints exist."
        )


    # ========================================================
    # VALIDATION DATASET
    # ========================================================

    val_subset = Subset(
        dataset,
        val_indices
    )


    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )


    print()
    print(
        f"Validation images: "
        f"{len(val_subset)}"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model(
        checkpoint_path
    )


    # ========================================================
    # RUN INFERENCE
    # ========================================================

    print()
    print("Running inference...")


    (
        final_labels,
        final_predictions,
        final_probabilities
    ) = run_inference(
        model,
        val_loader
    )


    # ========================================================
    # BASIC CHECK
    # ========================================================

    if len(final_labels) != len(val_indices):

        raise RuntimeError(
            f"\nFold {fold}: "
            "Number of predictions does not match "
            "validation set size."
        )


    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        final_labels,
        final_predictions
    )


    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(

            final_labels,

            final_predictions,

            labels=np.arange(
                NUM_CLASSES
            ),

            zero_division=0
        )
    )


    macro_precision = np.mean(
        precision
    )

    macro_recall = np.mean(
        recall
    )

    macro_f1 = np.mean(
        f1
    )


    # ========================================================
    # AUC
    # ========================================================
    #
    # EXACT SAME FORMULA AS ORIGINAL CV CODE
    # ========================================================

    y_true_bin = label_binarize(
        final_labels,
        classes=np.arange(
            NUM_CLASSES
        )
    )


    macro_auc = roc_auc_score(

        y_true_bin,

        final_probabilities,

        multi_class="ovr",

        average="macro"
    )


    # ========================================================
    # PER-CLASS AUC
    # ========================================================

    per_class_auc = []

    for class_idx in range(
        NUM_CLASSES
    ):

        class_auc = roc_auc_score(

            y_true_bin[:, class_idx],

            final_probabilities[:, class_idx]
        )

        per_class_auc.append(
            class_auc
        )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("-" * 70)
    print(f"FOLD {fold} RESULTS")
    print("-" * 70)

    print(
        f"Accuracy        : {accuracy:.6f}"
    )

    print(
        f"Macro Precision : {macro_precision:.6f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.6f}"
    )

    print(
        f"Macro F1        : {macro_f1:.6f}"
    )

    print(
        f"Macro AUC       : {macro_auc:.6f}"
    )


    # ========================================================
    # ACCURACY VERIFICATION
    # ========================================================

    expected_acc = EXPECTED_ACCURACY[
        fold
    ]

    accuracy_difference = abs(
        accuracy - expected_acc
    )


    print()
    print(
        "Original recorded accuracy "
        f": {expected_acc:.6f}"
    )

    print(
        "Recovered accuracy          "
        f": {accuracy:.6f}"
    )

    print(
        "Accuracy difference         "
        f": {accuracy_difference:.8f}"
    )


    # --------------------------------------------------------
    # Use tolerance because original CSV values were rounded.
    # --------------------------------------------------------

    if accuracy_difference <= 0.0001:

        print(
            "Accuracy verification      : PASS"
        )

    else:

        print(
            "Accuracy verification      : WARNING"
        )

        print(
            "The recreated validation "
            "predictions do not exactly match "
            "the original recorded accuracy."
        )


    # ========================================================
    # AUC VERIFICATION FOR FOLDS 4 AND 5
    # ========================================================

    if fold in EXPECTED_AUC:

        expected_auc = EXPECTED_AUC[
            fold
        ]

        auc_difference = abs(
            macro_auc - expected_auc
        )


        print()
        print(
            "Original recorded AUC       "
            f": {expected_auc:.6f}"
        )

        print(
            "Recovered AUC              "
            f": {macro_auc:.6f}"
        )

        print(
            "AUC difference             "
            f": {auc_difference:.8f}"
        )


        if auc_difference <= 0.00001:

            print(
                "AUC verification           : PASS"
            )

        else:

            print(
                "AUC verification           : WARNING"
            )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_row = {

        "Fold": fold,

        "Validation_Images": len(
            val_indices
        ),

        "Accuracy": accuracy,

        "Macro_Precision": macro_precision,

        "Macro_Recall": macro_recall,

        "Macro_F1": macro_f1,

        "Macro_AUC": macro_auc,

        "Original_Accuracy": expected_acc,

        "Accuracy_Difference": accuracy_difference
    }


    if fold in EXPECTED_AUC:

        result_row[
            "Original_AUC"
        ] = EXPECTED_AUC[fold]

        result_row[
            "AUC_Difference"
        ] = abs(
            macro_auc -
            EXPECTED_AUC[fold]
        )

    else:

        result_row[
            "Original_AUC"
        ] = np.nan

        result_row[
            "AUC_Difference"
        ] = np.nan


    all_results.append(
        result_row
    )


    # ========================================================
    # SAVE PER-CLASS AUC
    # ========================================================

    per_class_auc_path = os.path.join(

        OUTPUT_DIR,

        f"fold_{fold}_per_class_auc.csv"
    )


    per_class_auc_df = pd.DataFrame({

        "Class": CLASS_NAMES,

        "One_vs_Rest_AUC":
            per_class_auc

    })


    per_class_auc_df.to_csv(
        per_class_auc_path,
        index=False
    )


    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    del model

    del val_loader

    del val_subset

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 23. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# 24. SAVE PER-FOLD RESULTS
# ============================================================

per_fold_path = os.path.join(

    OUTPUT_DIR,

    "auc_recovery_per_fold.csv"
)


results_df.to_csv(
    per_fold_path,
    index=False
)


# ============================================================
# 25. CALCULATE 5-FOLD AUC MEAN AND SD
# ============================================================
#
# IMPORTANT:
# Unlike the old summary, this uses ALL FIVE recovered
# AUC values.
# ============================================================

auc_values = results_df[
    "Macro_AUC"
].astype(float)


auc_mean = auc_values.mean()

auc_std = auc_values.std(
    ddof=1
)


# ============================================================
# 26. ALSO CALCULATE OTHER METRIC MEANS
# ============================================================

accuracy_mean = results_df[
    "Accuracy"
].mean()

accuracy_std = results_df[
    "Accuracy"
].std(
    ddof=1
)


precision_mean = results_df[
    "Macro_Precision"
].mean()

precision_std = results_df[
    "Macro_Precision"
].std(
    ddof=1
)


recall_mean = results_df[
    "Macro_Recall"
].mean()

recall_std = results_df[
    "Macro_Recall"
].std(
    ddof=1
)


f1_mean = results_df[
    "Macro_F1"
].mean()

f1_std = results_df[
    "Macro_F1"
].std(
    ddof=1
)


# ============================================================
# 27. SUMMARY TABLE
# ============================================================

summary_df = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "Macro AUC"
    ],

    "Mean": [

        accuracy_mean,

        precision_mean,

        recall_mean,

        f1_mean,

        auc_mean
    ],

    "Std": [

        accuracy_std,

        precision_std,

        recall_std,

        f1_std,

        auc_std
    ]
})


# ============================================================
# 28. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(

    OUTPUT_DIR,

    "auc_recovery_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 29. FINAL REPORT
# ============================================================

print()
print()
print("=" * 70)
print("AUC RECOVERY COMPLETED")
print("=" * 70)


print()
print("PER-FOLD MACRO AUC")
print("-" * 70)


for _, row in results_df.iterrows():

    print(

        f"Fold {int(row['Fold'])}: "
        f"{row['Macro_AUC']:.6f}"

    )


print()
print("-" * 70)

print(
    f"5-Fold Mean Macro AUC : "
    f"{auc_mean:.6f}"
)

print(
    f"5-Fold SD             : "
    f"{auc_std:.6f}"
)


print()
print(
    f"5-Fold Macro AUC      : "
    f"{auc_mean:.6f} ± {auc_std:.6f}"
)


# ============================================================
# 30. PERCENTAGE FORM
# ============================================================

print()
print(
    f"5-Fold Macro AUC (%)  : "
    f"{auc_mean * 100:.2f}% ± "
    f"{auc_std * 100:.2f}%"
)


# ============================================================
# 31. PRINT OTHER METRICS
# ============================================================

print()
print("=" * 70)
print("5-FOLD METRIC SUMMARY")
print("=" * 70)

print()

print(
    f"Accuracy        : "
    f"{accuracy_mean:.6f} ± "
    f"{accuracy_std:.6f}"
)

print(
    f"Macro Precision : "
    f"{precision_mean:.6f} ± "
    f"{precision_std:.6f}"
)

print(
    f"Macro Recall    : "
    f"{recall_mean:.6f} ± "
    f"{recall_std:.6f}"
)

print(
    f"Macro F1        : "
    f"{f1_mean:.6f} ± "
    f"{f1_std:.6f}"
)

print(
    f"Macro AUC       : "
    f"{auc_mean:.6f} ± "
    f"{auc_std:.6f}"
)


# ============================================================
# 32. FINAL VALIDATION CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL CHECK")
print("=" * 70)


all_accuracy_checks_passed = True


for _, row in results_df.iterrows():

    fold = int(
        row["Fold"]
    )

    difference = row[
        "Accuracy_Difference"
    ]

    if difference > 0.0001:

        all_accuracy_checks_passed = False

        print(
            f"Fold {fold}: "
            f"ACCURACY CHECK FAILED"
        )

    else:

        print(
            f"Fold {fold}: "
            f"accuracy check PASSED"
        )


# ============================================================
# CHECK FOLD 4 AND 5 AUC
# ============================================================

auc_checks_passed = True


for fold in [4, 5]:

    recovered_auc = results_df.loc[
        results_df["Fold"] == fold,
        "Macro_AUC"
    ].iloc[0]


    expected_auc = EXPECTED_AUC[
        fold
    ]


    difference = abs(
        recovered_auc -
        expected_auc
    )


    if difference > 0.00001:

        auc_checks_passed = False

        print(
            f"Fold {fold}: "
            f"AUC CHECK FAILED"
        )

    else:

        print(
            f"Fold {fold}: "
            f"AUC check PASSED"
        )


# ============================================================
# 33. FINAL STATUS
# ============================================================

print()

if (
    all_accuracy_checks_passed
    and auc_checks_passed
):

    print(
        "OVERALL VERIFICATION: PASSED"
    )

    print()

    print(
        "The recovered AUC values are "
        "consistent with the original "
        "CV experiment."
    )

else:

    print(
        "OVERALL VERIFICATION: WARNING"
    )

    print()

    print(
        "Do NOT use the recovered 5-fold "
        "AUC in the manuscript yet."
    )

    print(
        "Inspect the verification warnings above."
    )


# ============================================================
# 34. OUTPUT FILE LOCATIONS
# ============================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print()

print(
    "Per-fold results:"
)

print(
    per_fold_path
)

print()

print(
    "Summary:"
)

print(
    summary_path
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)