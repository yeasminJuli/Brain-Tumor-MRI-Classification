# ============================================================
# DENSENET121 - 5-FOLD STRATIFIED CROSS-VALIDATION
# FINAL SIMPLE VERSION
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score
    , confusion_matrix
)

from tqdm.auto import tqdm


def save_combined_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    table = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    table.index.name = "Actual"
    table.columns.name = "Predicted"
    table.to_csv(os.path.join(SAVE_ROOT, "DENSENET121_COMBINED_5FOLD_CONFUSION_MATRIX.csv"))

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(title="DenseNet121 - Combined 5-Fold Confusion Matrix",
           xlabel="Predicted label", ylabel="Actual label",
           xticks=np.arange(NUM_CLASSES), yticks=np.arange(NUM_CLASSES),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_ROOT, "DENSENET121_COMBINED_5FOLD_CONFUSION_MATRIX.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\user\juli\thesis\original_file\othr folders\PREPROCESSED_TRAINN_AUGMENTED"

FIXED_FOLDS_FILE = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\CV_RESULTS_PROPOSED_MODEL\FIXED_5_FOLD_SPLITS.npz"
)

SAVE_ROOT = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\CV_RESULTS_DENSENET121_SAFE_RERUN"
)

SEED = 42
N_SPLITS = 5

NUM_CLASSES = 4
INPUT_SIZE = (256, 256)

BATCH_SIZE = 16
LEARNING_RATE = 1e-4

NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

NUM_WORKERS = 0

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
USE_AMP = DEVICE.type == "cuda"

print("=" * 70, flush=True)
print("DENSENET121 - 5-FOLD STRATIFIED CROSS-VALIDATION", flush=True)
print("=" * 70, flush=True)

print(f"Device        : {DEVICE}", flush=True)

if torch.cuda.is_available():
    print(
        f"GPU           : {torch.cuda.get_device_name(0)}",
        flush=True
    )

print(f"Dataset       : {DATA_DIR}", flush=True)
print(f"Fixed folds   : {FIXED_FOLDS_FILE}", flush=True)
print(f"Save folder   : {SAVE_ROOT}", flush=True)
print(f"Folds         : {N_SPLITS}", flush=True)
print(f"Epochs        : {NUM_EPOCHS}", flush=True)
print(f"Batch size    : {BATCH_SIZE}", flush=True)
print(f"Learning rate : {LEARNING_RATE}", flush=True)
print(f"Workers       : {NUM_WORKERS}", flush=True)
print(
    f"Early stopping: patience={EARLY_STOPPING_PATIENCE}",
    flush=True
)

print("=" * 70, flush=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

print("\nDataset information:", flush=True)
print(f"Total images: {len(dataset)}", flush=True)
print(f"Classes     : {dataset.classes}", flush=True)

# Check class order
if dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"\nClass order mismatch!\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found   : {dataset.classes}"
    )


# ============================================================
# LOAD FIXED FOLDS
# ============================================================

if not os.path.exists(FIXED_FOLDS_FILE):
    raise FileNotFoundError(
        f"\nFixed fold file not found:\n{FIXED_FOLDS_FILE}"
    )

fixed_folds = np.load(FIXED_FOLDS_FILE)

print("\nLoading existing fixed 5-fold splits...", flush=True)

folds = []

for fold_number in range(1, N_SPLITS + 1):

    train_key = f"fold_{fold_number}_train"
    val_key = f"fold_{fold_number}_validation"

    if train_key not in fixed_folds:
        raise KeyError(
            f"Missing key in fixed fold file: {train_key}"
        )

    if val_key not in fixed_folds:
        raise KeyError(
            f"Missing key in fixed fold file: {val_key}"
        )

    train_indices = fixed_folds[train_key].astype(int)
    val_indices = fixed_folds[val_key].astype(int)

    # Check overlap
    overlap = np.intersect1d(
        train_indices,
        val_indices
    )

    if len(overlap) != 0:
        raise ValueError(
            f"Fold {fold_number} has overlapping "
            f"train/validation samples!"
        )

    folds.append(
        (train_indices, val_indices)
    )

    print(
        f"Fold {fold_number}: "
        f"train={len(train_indices)}, "
        f"validation={len(val_indices)}",
        flush=True
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    print(
        "Loading pretrained DenseNet121...",
        flush=True
    )

    model = densenet121(
        weights=DenseNet121_Weights.IMAGENET1K_V1
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    print(
        "DenseNet121 loaded.",
        flush=True
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_class_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_micro = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_micro = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_micro = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    try:

        auc_macro = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )

        auc_micro = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="micro"
        )

    except ValueError:

        auc_macro = np.nan
        auc_micro = np.nan

    return {
        "accuracy": accuracy,
        "micro_accuracy": accuracy,
        "macro_class_accuracy": macro_class_accuracy,

        "precision_macro": precision_macro,
        "precision_micro": precision_micro,

        "recall_macro": recall_macro,
        "recall_micro": recall_micro,

        "f1_macro": f1_macro,
        "f1_micro": f1_micro,

        "auc_macro": auc_macro,
        "auc_micro": auc_micro
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold_number,
    train_indices,
    val_indices
):

    print("\n" + "=" * 70, flush=True)
    print(
        f"FOLD {fold_number}/{N_SPLITS}",
        flush=True
    )
    print("=" * 70, flush=True)

    set_seed(SEED + fold_number)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        foreach=False
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # --------------------------------------------------------
    # EARLY STOPPING VARIABLES
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_epoch = 0

    best_model_state = None

    patience_counter = 0

    history = []

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # ----------------------------------------------------
        # TRAINING LOOP WITH TQDM
        # ----------------------------------------------------

        train_progress = tqdm(
            train_loader,
            desc=(
                f"Fold {fold_number} | "
                f"Epoch {epoch:02d}/{NUM_EPOCHS}"
            ),
            unit="batch",
            dynamic_ncols=True,
            leave=True
        )

        for images, labels in train_progress:

            images = images.to(
                DEVICE,
                non_blocking=False
            )

            labels = labels.to(
                DEVICE,
                non_blocking=False
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            train_total += labels.size(0)

            train_progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss / train_total
        )

        train_accuracy = (
            train_correct / train_total
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_correct = 0
        val_total = 0

        val_loss_total = 0.0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=False
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=False
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=USE_AMP
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_total += (
                    loss.item() * images.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    (predictions == labels)
                    .sum()
                    .item()
                )

                val_total += labels.size(0)

        val_loss = (
            val_loss_total / val_total
        )

        val_accuracy = (
            val_correct / val_total
        )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}%",
            flush=True
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

            print(
                f"  -> New best validation accuracy: "
                f"{best_val_accuracy * 100:.4f}%",
                flush=True
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})",
                flush=True
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(
                f"\nEarly stopping triggered at "
                f"epoch {epoch}.",
                flush=True
            )

            break

    # ========================================================
    # LOAD BEST MODEL FROM RAM
    # ========================================================

    model.load_state_dict(
        best_model_state
    )

    model.to(DEVICE)

    model.eval()

    # ========================================================
    # FINAL VALIDATION USING BEST MODEL
    # ========================================================

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=False
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=USE_AMP
            ):
                outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )

    all_probabilities = np.array(
        all_probabilities
    )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
        all_probabilities
    )
    metrics["loss"] = float(-np.mean(np.log(np.clip(
        all_probabilities[np.arange(len(all_labels)), all_labels], 1e-12, 1.0
    ))))

    metrics["fold"] = fold_number
    metrics["best_epoch"] = best_epoch
    metrics["best_val_accuracy"] = best_val_accuracy
    metrics["validation_samples"] = len(val_indices)

    # ========================================================
    # SAVE FOLD RESULT
    # ========================================================

    os.makedirs(
        SAVE_ROOT,
        exist_ok=True
    )

    fold_result_path = os.path.join(
        SAVE_ROOT,
        f"fold_{fold_number}_metrics.csv"
    )

    pd.DataFrame([metrics]).to_csv(
        fold_result_path,
        index=False
    )

    prediction_frame = pd.DataFrame({
        "true_label": all_labels,
        "predicted_label": all_predictions,
    })
    for class_index, class_name in enumerate(CLASS_NAMES):
        prediction_frame[f"prob_{class_name}"] = all_probabilities[:, class_index]
    prediction_frame.to_csv(
        os.path.join(SAVE_ROOT, f"fold_{fold_number}_predictions.csv"), index=False
    )

    # ========================================================
    # PRINT FOLD RESULT
    # ========================================================

    print("\nFold result:", flush=True)

    print(
        f"Accuracy              : "
        f"{metrics['accuracy'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro Class Accuracy  : "
        f"{metrics['macro_class_accuracy'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro Precision       : "
        f"{metrics['precision_macro'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro Recall          : "
        f"{metrics['recall_macro'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro F1              : "
        f"{metrics['f1_macro'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro AUC             : "
        f"{metrics['auc_macro']:.6f}",
        flush=True
    )

    print(
        f"Best Epoch            : "
        f"{best_epoch}",
        flush=True
    )

    print(
        f"Best Validation Acc   : "
        f"{best_val_accuracy * 100:.4f}%",
        flush=True
    )

    del model, optimizer, scaler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, history


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        SAVE_ROOT,
        exist_ok=True
    )

    all_fold_results = []

    # ========================================================
    # RUN 5 FOLDS
    # ========================================================

    for fold_number in range(1, N_SPLITS + 1):

        train_indices, val_indices = folds[
            fold_number - 1
        ]

        metrics, history = train_fold(
            fold_number,
            train_indices,
            val_indices
        )

        all_fold_results.append(
            metrics
        )

    # ========================================================
    # ALL FOLD RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_fold_results
    )

    results_df = results_df.sort_values(
        "fold"
    )

    results_path = os.path.join(
        SAVE_ROOT,
        "DENSENET121_5FOLD_RESULTS.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    prediction_frames = [
        pd.read_csv(os.path.join(SAVE_ROOT, f"fold_{fold_number}_predictions.csv"))
        for fold_number in range(1, N_SPLITS + 1)
    ]
    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    combined_predictions.to_csv(
        os.path.join(SAVE_ROOT, "DENSENET121_ALL_5FOLD_OOF_PREDICTIONS.csv"), index=False
    )
    save_combined_confusion_matrix(
        combined_predictions["true_label"].to_numpy(),
        combined_predictions["predicted_label"].to_numpy(),
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    metric_columns = [
        "accuracy",
        "micro_accuracy",
        "macro_class_accuracy",
        "precision_macro",
        "precision_micro",
        "recall_macro",
        "recall_micro",
        "f1_macro",
        "f1_micro",
        "auc_macro",
        "auc_micro"
        , "loss", "best_epoch"
    ]

    summary_rows = []

    for metric in metric_columns:

        values = results_df[
            metric
        ].astype(float)

        mean_value = values.mean()
        std_value = values.std(
            ddof=1
        )

        se_value = (
            std_value /
            np.sqrt(len(values))
        )

        summary_rows.append({
            "metric": metric,
            "mean": mean_value,
            "std": std_value,
            "se": se_value,
            "ci95_lower": mean_value - 2.7764451051977987 * se_value,
            "ci95_upper": mean_value + 2.7764451051977987 * se_value
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        SAVE_ROOT,
        "DENSENET121_5FOLD_SUMMARY.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print("\n" + "=" * 70, flush=True)
    print(
        "DENSENET121 - FINAL 5-FOLD RESULTS",
        flush=True
    )
    print("=" * 70, flush=True)

    for metric in [
        "accuracy",
        "macro_class_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_macro"
    ]:

        row = summary_df[
            summary_df["metric"] == metric
        ].iloc[0]

        print(
            f"{metric:25s}: "
            f"{row['mean'] * 100:.4f}% "
            f"± {row['std'] * 100:.4f}%",
            flush=True
        )

    print("\nFold-wise results:", flush=True)

    print(
        results_df[
            [
                "fold",
                "best_epoch",
                "accuracy",
                "macro_class_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auc_macro"
            ]
        ].to_string(index=False),
        flush=True
    )

    print("\nResults saved to:", flush=True)
    print(SAVE_ROOT, flush=True)

    print("\nDone.", flush=True)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()
