# ============================================================
# 5-FOLD CROSS-VALIDATION FOR DATASET B
# Hybrid CNN + BiLSTM + Additive Attention
# ============================================================

import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_DIR = r"C:\user\juli\thesis\Dataset\preprocessed_train"

OUTPUT_DIR = r"C:\user\juli\thesis\DatasetB_5Fold_CV"

IMAGE_SIZE = (256, 256)

NUM_CLASSES = 4
BATCH_SIZE = 16

LEARNING_RATE = 1e-4

NUM_EPOCHS = 80

# Early stopping
MIN_EPOCHS = 30
PATIENCE = 8

N_SPLITS = 5

RANDOM_SEED = 42

# IMPORTANT FOR WINDOWS
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


# ============================================================
# 3. CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("Hybrid CNN + BiLSTM + Additive Attention")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Maximum epochs: {NUM_EPOCHS}")
print(f"Minimum epochs: {MIN_EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")
print(f"Number of folds: {N_SPLITS}")
print(f"Number of workers: {NUM_WORKERS}")

print("=" * 70)


# ============================================================
# 5. CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# 6. TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 7. DATASET CLASS
# ============================================================

class BrainTumorDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        self.class_to_idx = {
            "glioma": 0,
            "meningioma": 1,
            "notumor": 2,
            "pituitary": 3
        }

        for class_name, class_idx in self.class_to_idx.items():

            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            for filename in os.listdir(class_dir):

                filepath = os.path.join(class_dir, filename)

                if os.path.isfile(filepath):

                    if filename.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                    ):

                        self.samples.append(
                            (filepath, class_idx)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        filepath, label = self.samples[index]

        image = Image.open(filepath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# 8. LOAD DATASET
# ============================================================

dataset = BrainTumorDataset(
    DATA_DIR,
    transform=transform
)


print("\nDataset class distribution:")

labels = np.array([label for _, label in dataset.samples])

for class_idx, class_name in enumerate(CLASS_NAMES):

    count = np.sum(labels == class_idx)

    print(f"{class_name}: {count}")

print(f"Total: {len(dataset)}")

print("=" * 70)


# ============================================================
# 9. MODEL: RESIDUAL BLOCK
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

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        else:

            self.shortcut = nn.Identity()

        self.pool = (
            nn.MaxPool2d(kernel_size=2, stride=2)
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
# 10. HYBRID CNN + BiLSTM + ATTENTION MODEL
# ============================================================

class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # ----------------------------
        # CNN FEATURE EXTRACTOR
        # ----------------------------

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

            nn.AdaptiveAvgPool2d((8, 8))
        )


        # CNN output:
        # 256 x 8 x 8
        #
        # Convert spatial dimensions into sequence:
        # sequence length = 64
        # feature size = 256

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

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # Shape:
        # batch, channels, height, width

        # Convert to:
        # batch, sequence, features

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


        # BiLSTM

        lstm_out, _ = self.bilstm(x)


        # ====================================================
        # ADDITIVE ATTENTION
        # ====================================================

        attention_hidden = torch.tanh(
            self.attention_linear(lstm_out)
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


        # Classifier

        output = self.fc(context)

        return output


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels_batch in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels_batch = labels_batch.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels_batch.detach().cpu().numpy()
        )

        progress.set_postfix(
            loss=loss.item()
        )


    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# 12. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels_batch in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels_batch = labels_batch.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels_batch.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )


    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# 13. START 5-FOLD CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED
)


all_fold_metrics = []
all_per_class_metrics = []
all_confusion_matrices = []


# ============================================================
# 14. FOLD LOOP
# ============================================================

for fold, (train_indices, val_indices) in enumerate(
    skf.split(
        np.zeros(len(labels)),
        labels
    ),
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 70)

    fold_dir = os.path.join(
        OUTPUT_DIR,
        f"fold_{fold}"
    )

    os.makedirs(
        fold_dir,
        exist_ok=True
    )


    print(
        f"Training images: {len(train_indices)}"
    )

    print(
        f"Validation images: {len(val_indices)}"
    )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_subset = Subset(
        dataset,
        train_indices
    )

    val_subset = Subset(
        dataset,
        val_indices
    )


    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # NEW MODEL FOR EVERY FOLD
    # ========================================================

    model = HybridCNNBiLSTMAttention(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # TRACKING
    # ========================================================

    history = {

        "epoch": [],

        "train_loss": [],
        "train_accuracy": [],

        "val_loss": [],
        "val_accuracy": []
    }


    best_val_accuracy = -np.inf

    best_epoch = 0

    best_model_state = None

    epochs_without_improvement = 0


    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(1, NUM_EPOCHS + 1):

        print(
            f"\nFold {fold} | "
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )


        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )


        (
            val_loss,
            val_accuracy,
            val_labels,
            val_predictions,
            val_probabilities
        ) = validate(
            model,
            val_loader,
            criterion
        )


        history["epoch"].append(epoch)

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )


        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            torch.save(
                best_model_state,
                os.path.join(
                    fold_dir,
                    "best_model.pth"
                )
            )

            print(
                f"*** Best model saved "
                f"(Val Acc: {best_val_accuracy * 100:.2f}%)"
            )

        else:

            epochs_without_improvement += 1


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epoch >= MIN_EPOCHS
            and epochs_without_improvement >= PATIENCE
        ):

            print(
                f"\nEarly stopping triggered "
                f"at epoch {epoch}."
            )

            print(
                f"Best epoch: {best_epoch}"
            )

            print(
                f"Best validation accuracy: "
                f"{best_val_accuracy * 100:.2f}%"
            )

            break


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        os.path.join(
            fold_dir,
            "training_history.csv"
        ),
        index=False
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_model_state
    )


    # ========================================================
    # FINAL VALIDATION USING BEST MODEL
    # ========================================================

    (
        final_val_loss,
        final_val_accuracy,
        final_labels,
        final_predictions,
        final_probabilities
    ) = validate(
        model,
        val_loader,
        criterion
    )


    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            final_labels,
            final_predictions,
            labels=np.arange(NUM_CLASSES),
            zero_division=0
        )
    )


    # ========================================================
    # MACRO METRICS
    # ========================================================

    macro_precision = np.mean(precision)

    macro_recall = np.mean(recall)

    macro_f1 = np.mean(f1)


    # ========================================================
    # AUC
    # ========================================================

    try:

        y_true_bin = label_binarize(
            final_labels,
            classes=np.arange(NUM_CLASSES)
        )

        macro_auc = roc_auc_score(
            y_true_bin,
            final_probabilities,
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        macro_auc = np.nan


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        final_labels,
        final_predictions,
        labels=np.arange(NUM_CLASSES)
    )

    all_confusion_matrices.append(cm)


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        os.path.join(
            fold_dir,
            "confusion_matrix.csv"
        )
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        final_labels,
        final_predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )


    with open(
        os.path.join(
            fold_dir,
            "classification_report.txt"
        ),
        "w"
    ) as f:

        f.write(report)


    # ========================================================
    # PER-CLASS METRICS
    # ========================================================

    per_class_df = pd.DataFrame({

        "Class": CLASS_NAMES,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "Support": support
    })


    per_class_df.to_csv(
        os.path.join(
            fold_dir,
            "per_class_metrics.csv"
        ),
        index=False
    )


    # ========================================================
    # FOLD SUMMARY
    # ========================================================

    fold_result = {

        "Fold": fold,

        "Best_Epoch": best_epoch,

        "Epochs_Run": len(history["epoch"]),

        "Accuracy": final_val_accuracy,

        "Macro_Precision": macro_precision,

        "Macro_Recall": macro_recall,

        "Macro_F1": macro_f1,

        "Macro_AUC": macro_auc,

        "Validation_Loss": final_val_loss
    }


    all_fold_metrics.append(
        fold_result
    )


    # ========================================================
    # STORE PER-CLASS METRICS
    # ========================================================

    for i, class_name in enumerate(CLASS_NAMES):

        all_per_class_metrics.append({

            "Fold": fold,

            "Class": class_name,

            "Precision": precision[i],

            "Recall": recall[i],

            "F1_Score": f1[i],

            "Support": support[i]
        })


    # ========================================================
    # SAVE MODEL SUMMARY
    # ========================================================

    torch.save(
        best_model_state,
        os.path.join(
            fold_dir,
            "best_model.pth"
        )
    )


    # ========================================================
    # TRAINING CURVES
    # ========================================================

    epochs = history["epoch"]


    # Loss curve

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        f"Fold {fold} - Loss Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fold_dir,
            "loss_curve.png"
        ),
        dpi=300
    )

    plt.close()


    # Accuracy curve

    plt.figure(figsize=(8, 6))

    plt.plot(
        epochs,
        np.array(history["train_accuracy"]) * 100,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        np.array(history["val_accuracy"]) * 100,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title(
        f"Fold {fold} - Accuracy Curve"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            fold_dir,
            "accuracy_curve.png"
        ),
        dpi=300
    )

    plt.close()


    # ========================================================
    # FOLD COMPLETE
    # ========================================================

    print("\n" + "-" * 70)

    print(f"Fold {fold} completed.")

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Accuracy: "
        f"{final_val_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision: "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1: "
        f"{macro_f1:.4f}"
    )

    print(
        f"Macro AUC: "
        f"{macro_auc:.4f}"
    )

    print("-" * 70)


# ============================================================
# 15. SAVE ALL FOLD METRICS
# ============================================================

fold_metrics_df = pd.DataFrame(
    all_fold_metrics
)

fold_metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "fold_metrics.csv"
    ),
    index=False
)


# ============================================================
# 16. MEAN ± STANDARD DEVIATION
# ============================================================

metric_columns = [
    "Accuracy",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Macro_AUC"
]


summary_rows = []


for metric in metric_columns:

    values = fold_metrics_df[metric].dropna()

    summary_rows.append({

        "Metric": metric,

        "Mean": values.mean(),

        "Std": values.std(ddof=1),

        "Mean_PlusMinus_SD":
            f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
    })


cv_summary_df = pd.DataFrame(
    summary_rows
)


cv_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cv_summary_mean_std.csv"
    ),
    index=False
)


# ============================================================
# 17. PER-CLASS METRICS ACROSS ALL FOLDS
# ============================================================

all_per_class_df = pd.DataFrame(
    all_per_class_metrics
)


all_per_class_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_folds_per_class_metrics.csv"
    ),
    index=False
)


# ============================================================
# 18. PER-CLASS MEAN ± SD
# ============================================================

per_class_summary = []


for class_name in CLASS_NAMES:

    class_data = all_per_class_df[
        all_per_class_df["Class"] == class_name
    ]


    per_class_summary.append({

        "Class": class_name,

        "Precision_Mean":
            class_data["Precision"].mean(),

        "Precision_SD":
            class_data["Precision"].std(ddof=1),

        "Recall_Mean":
            class_data["Recall"].mean(),

        "Recall_SD":
            class_data["Recall"].std(ddof=1),

        "F1_Mean":
            class_data["F1_Score"].mean(),

        "F1_SD":
            class_data["F1_Score"].std(ddof=1)
    })


per_class_summary_df = pd.DataFrame(
    per_class_summary
)


per_class_summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "per_class_mean_std.csv"
    ),
    index=False
)


# ============================================================
# 19. SAVE ALL CONFUSION MATRICES
# ============================================================

all_cm_array = np.array(
    all_confusion_matrices
)


cm_rows = []


for fold_idx, cm in enumerate(
    all_cm_array,
    start=1
):

    for i, actual_class in enumerate(
        CLASS_NAMES
    ):

        for j, predicted_class in enumerate(
            CLASS_NAMES
        ):

            cm_rows.append({

                "Fold": fold_idx,

                "Actual": actual_class,

                "Predicted": predicted_class,

                "Count": cm[i, j]
            })


all_cm_df = pd.DataFrame(
    cm_rows
)


all_cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_folds_confusion_matrices.csv"
    ),
    index=False
)


# ============================================================
# 20. MEAN CONFUSION MATRIX
# ============================================================

mean_cm = np.mean(
    all_cm_array,
    axis=0
)


mean_cm_df = pd.DataFrame(
    mean_cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


mean_cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mean_confusion_matrix.csv"
    )
)


# ============================================================
# 21. FINAL CONSOLE SUMMARY
# ============================================================

print("\n\n")

print("=" * 70)

print("5-FOLD CROSS-VALIDATION COMPLETED")

print("=" * 70)


print("\nFold Results:")

print(
    fold_metrics_df.to_string(
        index=False
    )
)


print("\n")

print("=" * 70)

print("MEAN ± STANDARD DEVIATION")

print("=" * 70)


for _, row in cv_summary_df.iterrows():

    print(
        f"{row['Metric']}: "
        f"{row['Mean']:.4f} ± "
        f"{row['Std']:.4f}"
    )


print("\n")

print("=" * 70)

print("PER-CLASS MEAN ± SD")

print("=" * 70)


print(
    per_class_summary_df.to_string(
        index=False
    )
)


print("\n")

print("=" * 70)

print("OUTPUT FILES")

print("=" * 70)

print(
    f"All results saved to:\n"
    f"{OUTPUT_DIR}"
)


print("\nFiles generated:")

print("1. fold_metrics.csv")

print("2. cv_summary_mean_std.csv")

print("3. all_folds_per_class_metrics.csv")

print("4. per_class_mean_std.csv")

print("5. all_folds_confusion_matrices.csv")

print("6. mean_confusion_matrix.csv")

print("7. Each fold contains:")
print("   - best_model.pth")
print("   - training_history.csv")
print("   - classification_report.txt")
print("   - confusion_matrix.csv")
print("   - per_class_metrics.csv")
print("   - loss_curve.png")
print("   - accuracy_curve.png")

print("\n")

print("=" * 70)
print("DONE")
print("=" * 70)