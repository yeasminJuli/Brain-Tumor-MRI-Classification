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
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score
    , confusion_matrix
)

from tqdm.auto import tqdm


def save_combined_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    table = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    table.index.name = "Actual"
    table.columns.name = "Predicted"
    table.to_csv(os.path.join(SAVE_ROOT, "RESNET50_COMBINED_5FOLD_CONFUSION_MATRIX.csv"))

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="ResNet50 - Combined 5-Fold Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_ROOT, "RESNET50_COMBINED_5FOLD_CONFUSION_MATRIX.png"),
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
    r"\CV_RESULTS_RESNET50_SAFE_RERUN"
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
print("RESNET50 - 5-FOLD STRATIFIED CROSS-VALIDATION", flush=True)
print("=" * 70, flush=True)

print(f"Device           : {DEVICE}", flush=True)

if torch.cuda.is_available():
    print(
        f"GPU              : {torch.cuda.get_device_name(0)}",
        flush=True
    )

print(f"Dataset          : {DATA_DIR}", flush=True)
print(f"Fixed folds      : {FIXED_FOLDS_FILE}", flush=True)
print(f"Save folder      : {SAVE_ROOT}", flush=True)
print(f"Folds / Epochs   : {N_SPLITS} / {NUM_EPOCHS}", flush=True)
print(f"Batch / LR       : {BATCH_SIZE} / {LEARNING_RATE}", flush=True)
print(f"Workers          : {NUM_WORKERS}", flush=True)
print(f"Early stopping   : {EARLY_STOPPING_PATIENCE}", flush=True)
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


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

print(f"\nTotal images: {len(dataset)}", flush=True)
print(f"Classes: {dataset.classes}", flush=True)

if dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"\nClass order mismatch!\n"
        f"Expected: {CLASS_NAMES}\n"
        f"Found:    {dataset.classes}"
    )


# ============================================================
# LOAD FIXED 5-FOLD SPLITS
# ============================================================

if not os.path.exists(FIXED_FOLDS_FILE):
    raise FileNotFoundError(
        f"\nFixed fold file not found:\n{FIXED_FOLDS_FILE}\n"
        f"\nDo NOT generate new folds. "
        f"Use the FIXED_5_FOLD_SPLITS.npz created for the proposed model."
    )

print("\nLoading existing fixed folds...", flush=True)

fixed_folds = np.load(FIXED_FOLDS_FILE)

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

    if len(overlap) > 0:
        raise ValueError(
            f"Fold {fold_number} has overlapping train/validation indices."
        )

    folds.append(
        (
            train_indices,
            val_indices
        )
    )

    print(
        f"Fold {fold_number}: "
        f"train={len(train_indices)}, "
        f"validation={len(val_indices)}",
        flush=True
    )

print("Existing fixed folds loaded successfully.", flush=True)


# ============================================================
# MODEL
# ============================================================

def build_model():

    print(
        "\nLoading pretrained ResNet50...",
        flush=True
    )

    model = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    print(
        "ResNet50 loaded.",
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

    micro_accuracy = accuracy

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
    except ValueError:
        auc_macro = np.nan

    try:
        auc_micro = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="micro"
        )
    except ValueError:
        auc_micro = np.nan

    return {
        "accuracy": accuracy,
        "micro_accuracy": micro_accuracy,
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
        f"STARTING FOLD {fold_number}/{N_SPLITS}",
        flush=True
    )
    print("=" * 70, flush=True)

    # --------------------------------------------------------
    # Seed for this fold
    # --------------------------------------------------------

    set_seed(SEED + fold_number)

    # --------------------------------------------------------
    # Data
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
    # Model
    # --------------------------------------------------------

    model = build_model()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    #
    # foreach=False is intentional for stability with
    # Python 3.14 / newer PyTorch environments.
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        foreach=False
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # --------------------------------------------------------
    # Best model kept in RAM
    # --------------------------------------------------------

    best_val_accuracy = -1.0
    best_epoch = 0
    best_model_state = None

    patience_counter = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Fold {fold_number} | "
                f"Epoch {epoch:02d}/{NUM_EPOCHS}"
            ),
            unit="batch",
            dynamic_ncols=True,
            leave=True
        )

        for images, labels in progress_bar:

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

            # ----------------------------------------------
            # Training statistics
            # ----------------------------------------------

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%"
            )

        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_correct = 0
        val_total = 0

        val_predictions = []
        val_targets = []

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

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()

                val_predictions.extend(
                    predicted.cpu().numpy()
                )

                val_targets.extend(
                    labels.cpu().numpy()
                )

        val_accuracy = (
            val_correct / val_total
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}%",
            flush=True
        )

        # ====================================================
        # BEST MODEL + EARLY STOPPING
        # ====================================================

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
                f"{best_val_accuracy * 100:.4f}% "
                f"(Epoch {best_epoch})",
                flush=True
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement. "
                f"Patience: "
                f"{patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}",
                flush=True
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print(
                f"\nEarly stopping triggered at "
                f"Epoch {epoch}.",
                flush=True
            )

            break

    # ========================================================
    # LOAD BEST MODEL FROM RAM
    # ========================================================

    if best_model_state is None:

        raise RuntimeError(
            f"No best model was saved for Fold {fold_number}."
        )

    model.load_state_dict(
        best_model_state
    )

    model = model.to(DEVICE)

    # ========================================================
    # FINAL VALIDATION USING BEST MODEL
    # ========================================================

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

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

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

            y_prob.extend(
                probabilities.cpu().numpy()
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob
    )
    metrics["loss"] = float(-np.mean(np.log(np.clip(
        y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0
    ))))

    metrics["fold"] = fold_number
    metrics["best_epoch"] = best_epoch
    metrics["best_val_accuracy"] = best_val_accuracy
    metrics["validation_samples"] = len(val_indices)

    # ========================================================
    # SAVE FOLD METRICS
    # ========================================================

    fold_dir = os.path.join(
        SAVE_ROOT,
        f"Fold_{fold_number}"
    )

    os.makedirs(
        fold_dir,
        exist_ok=True
    )

    fold_metrics_path = os.path.join(
        fold_dir,
        f"fold_{fold_number}_metrics.csv"
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        fold_metrics_path,
        index=False
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    })

    for class_index, class_name in enumerate(CLASS_NAMES):

        predictions_df[
            f"prob_{class_name}"
        ] = y_prob[:, class_index]

    predictions_path = os.path.join(
        fold_dir,
        f"fold_{fold_number}_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # ========================================================
    # PRINT FOLD RESULT
    # ========================================================

    print("\n" + "-" * 70, flush=True)
    print(
        f"FOLD {fold_number} FINAL RESULT",
        flush=True
    )
    print("-" * 70, flush=True)

    print(
        f"Best Epoch              : {best_epoch}",
        flush=True
    )

    print(
        f"Accuracy                : "
        f"{metrics['accuracy'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro Class Accuracy    : "
        f"{metrics['macro_class_accuracy'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro Precision         : "
        f"{metrics['precision_macro'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro Recall            : "
        f"{metrics['recall_macro'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro F1                : "
        f"{metrics['f1_macro'] * 100:.4f}%",
        flush=True
    )

    print(
        f"Macro AUC               : "
        f"{metrics['auc_macro']:.6f}",
        flush=True
    )

    print("-" * 70, flush=True)

    del model, optimizer, scaler, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        SAVE_ROOT,
        exist_ok=True
    )

    all_results = []

    # ========================================================
    # RUN ALL 5 FOLDS
    # ========================================================

    for fold_number in range(
        1,
        N_SPLITS + 1
    ):

        train_indices, val_indices = folds[
            fold_number - 1
        ]

        result = train_fold(
            fold_number,
            train_indices,
            val_indices
        )

        all_results.append(
            result
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    # Put fold first
    preferred_columns = [
        "fold",
        "best_epoch",
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
        "auc_micro",
        "loss",
        "best_val_accuracy",
        "validation_samples"
    ]

    results_df = results_df[
        [
            column
            for column in preferred_columns
            if column in results_df.columns
        ]
    ]

    # ========================================================
    # SAVE 5-FOLD RESULTS
    # ========================================================

    results_path = os.path.join(
        SAVE_ROOT,
        "RESNET50_5FOLD_RESULTS.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    prediction_frames = []
    for fold_number in range(1, N_SPLITS + 1):
        prediction_frames.append(pd.read_csv(os.path.join(
            SAVE_ROOT, f"Fold_{fold_number}", f"fold_{fold_number}_predictions.csv"
        )))
    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    combined_predictions.to_csv(
        os.path.join(SAVE_ROOT, "RESNET50_ALL_5FOLD_OOF_PREDICTIONS.csv"), index=False
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

        if metric not in results_df.columns:
            continue

        values = results_df[
            metric
        ].dropna().values

        mean_value = np.mean(values)

        if len(values) > 1:

            sd_value = np.std(
                values,
                ddof=1
            )

            se_value = (
                sd_value /
                np.sqrt(len(values))
            )

        else:

            sd_value = np.nan
            se_value = np.nan

        summary_rows.append({
            "metric": metric,
            "mean": mean_value,
            "sd": sd_value,
            "se": se_value,
            "ci95_lower": mean_value - 2.7764451051977987 * se_value,
            "ci95_upper": mean_value + 2.7764451051977987 * se_value
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = os.path.join(
        SAVE_ROOT,
        "RESNET50_5FOLD_SUMMARY.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70, flush=True)
    print(
        "RESNET50 - FINAL 5-FOLD SUMMARY",
        flush=True
    )
    print("=" * 70, flush=True)

    for _, row in summary_df.iterrows():

        print(
            f"{row['metric']:25s} "
            f"Mean = {row['mean']:.6f} | "
            f"SD = {row['sd']:.6f} | "
            f"SE = {row['se']:.6f}",
            flush=True
        )

    print("\n" + "=" * 70, flush=True)
    print(
        "RESNET50 5-FOLD CROSS-VALIDATION COMPLETED",
        flush=True
    )
    print("=" * 70, flush=True)

    print(
        f"Results saved to:\n{results_path}",
        flush=True
    )

    print(
        f"Summary saved to:\n{summary_path}",
        flush=True
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
