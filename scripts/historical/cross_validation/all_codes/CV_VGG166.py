import os, json, time, random, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

DATA_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\PREPROCESSED_TRAINN_AUGMENTED"
)

FIXED_FOLD_SOURCE = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\CV_RESULTS_PROPOSED_MODEL"
)

FIXED_FOLDS_FILE = os.path.join(
    FIXED_FOLD_SOURCE,
    "FIXED_5_FOLD_SPLITS.npz"
)

SAVE_ROOT = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\CV_RESULTS_VGG16_RERUN"
)

SEED = 42
N_SPLITS = 5
NUM_CLASSES = 4
INPUT_SIZE = (256, 256)
BATCH_SIZE = 16
# VGG16 is substantially heavier than the other baselines.  Use two
# 8-image micro-batches so the effective optimization batch remains 16 while
# peak VRAM stays safe on a 12-GB RTX 4070 Super.
MICRO_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.0
NUM_WORKERS = 0
PRETRAINED = True
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
os.makedirs(SAVE_ROOT, exist_ok=True)

print("=" * 80)
print("VGG16 - 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 80)
print(f"Device           : {DEVICE}")
print(f"Dataset          : {DATA_DIR}")
print(f"Fixed folds      : {FIXED_FOLDS_FILE}")
print(f"Save folder      : {SAVE_ROOT}")
print(f"Folds / Epochs   : {N_SPLITS} / {NUM_EPOCHS}")
print(f"Batch / LR       : {BATCH_SIZE} / {LEARNING_RATE}")
print(f"Physical batch   : {MICRO_BATCH_SIZE} (gradient accumulation x{GRAD_ACCUM_STEPS})")
print(f"Workers          : {NUM_WORKERS}")
print(f"Pretrained       : {PRETRAINED}")
print(f"Early stopping   : {EARLY_STOPPING_PATIENCE}")
print("=" * 80)

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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
print(f"\nTotal images: {len(dataset)}")
print("Classes:", dataset.classes)

if dataset.classes != CLASS_NAMES:
    raise ValueError(f"Unexpected class order. Found {dataset.classes}, expected {CLASS_NAMES}.")

if not os.path.exists(FIXED_FOLDS_FILE):
    raise FileNotFoundError(
        f"FIXED_5_FOLD_SPLITS.npz not found: {FIXED_FOLDS_FILE}\n"
        "Do not create new folds for VGG16. Use the fixed fold file from the proposed-model experiment."
    )

def load_folds(path, n):
    with np.load(path, allow_pickle=False) as d:
        required = []
        for k in range(1, N_SPLITS + 1):
            required += [f"fold_{k}_train", f"fold_{k}_validation"]
        missing = [key for key in required if key not in d.files]
        if missing:
            raise KeyError(
                "Fixed fold file is missing required keys: "
                + ", ".join(missing)
                + f"\nAvailable keys: {list(d.files)}"
            )

        folds = []
        for k in range(1, N_SPLITS + 1):
            tr = d[f"fold_{k}_train"].astype(np.int64)
            va = d[f"fold_{k}_validation"].astype(np.int64)

            if len(tr) == 0 or len(va) == 0:
                raise ValueError(f"Fold {k} has an empty train or validation set.")
            if np.any(tr < 0) or np.any(tr >= n):
                raise ValueError(f"Invalid training index in fold {k}.")
            if np.any(va < 0) or np.any(va >= n):
                raise ValueError(f"Invalid validation index in fold {k}.")
            if len(np.unique(tr)) != len(tr):
                raise ValueError(f"Duplicate training indices in fold {k}.")
            if len(np.unique(va)) != len(va):
                raise ValueError(f"Duplicate validation indices in fold {k}.")
            if len(np.intersect1d(tr, va)) > 0:
                raise ValueError(f"Train/validation overlap in fold {k}.")
            folds.append((tr, va))
        return folds

folds = load_folds(FIXED_FOLDS_FILE, len(dataset))
print("\nExisting fixed folds loaded successfully.")
for i, (tr, va) in enumerate(folds, 1):
    print(f"Fold {i}: train={len(tr)}, validation={len(va)}")

def create_model():
    weights = VGG16_Weights.IMAGENET1K_V1 if PRETRAINED else None
    model = vgg16(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    return model.to(DEVICE)

def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def cpu_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def rng_save():
    out = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu().tolist()
    }
    if torch.cuda.is_available():
        out["cuda"] = [x.cpu().tolist() for x in torch.cuda.get_rng_state_all()]
    return out

def rng_restore(s):
    if not s:
        return
    try:
        random.setstate(s["python"])
        ns = s["numpy"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(torch.tensor(s["torch"], dtype=torch.uint8))
        if torch.cuda.is_available() and "cuda" in s:
            torch.cuda.set_rng_state_all([torch.tensor(x, dtype=torch.uint8) for x in s["cuda"]])
    except Exception as e:
        print(f"WARNING: RNG restore skipped: {e}")

def aucs(y_true, probabilities):
    y_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
    try:
        macro_auc = roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")
    except Exception:
        macro_auc = np.nan
    try:
        micro_auc = roc_auc_score(y_bin, probabilities, average="micro")
    except Exception:
        micro_auc = np.nan
    return macro_auc, micro_auc

def calculate_metrics(y_true, predictions, probabilities):
    macro_auc, micro_auc = aucs(y_true, probabilities)
    accuracy = accuracy_score(y_true, predictions)
    return {
        "accuracy": accuracy,
        "micro_accuracy": accuracy,
        "macro_class_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_macro": precision_score(y_true, predictions, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, predictions, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, predictions, average="macro", zero_division=0),
        "recall_micro": recall_score(y_true, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, predictions, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, predictions, average="micro", zero_division=0),
        "auc_macro": macro_auc,
        "auc_micro": micro_auc
    }

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    ys, preds, probs_list = [], [], []
    loss_sum, total = 0.0, 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            logits = model(x)
            loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        bs = y.size(0)
        loss_sum += loss.item() * bs
        total += bs
        ys.extend(y.cpu().numpy())
        preds.extend(pred.cpu().numpy())
        probs_list.append(probs.cpu().numpy())

    y_true = np.asarray(ys)
    y_pred = np.asarray(preds)
    y_prob = np.concatenate(probs_list, axis=0)
    result = calculate_metrics(y_true, y_pred, y_prob)
    result["loss"] = loss_sum / total
    return result, y_true, y_pred, y_prob

def train_fold(fold, train_idx, val_idx):
    fold_dir = os.path.join(SAVE_ROOT, f"Fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # This is temporary and is deleted immediately after fold evaluation.
    best_path = os.path.join(fold_dir, "TEMP_BEST_MODEL.pth")
    history_path = os.path.join(fold_dir, "training_history.csv")
    complete_path = os.path.join(fold_dir, "FOLD_COMPLETE.json")
    metric_path = os.path.join(fold_dir, "fold_metrics.csv")

    predictions_path = os.path.join(fold_dir, "raw_predictions.csv")
    if os.path.exists(complete_path) and os.path.exists(metric_path) and os.path.exists(predictions_path):
        print(f"\nFold {fold} already complete. Reusing saved metrics.")
        return pd.read_csv(metric_path).iloc[0].to_dict()

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=MICRO_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    patience_counter = 0

    fold_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch_number, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{NUM_EPOCHS}"), start=1
        ):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss / GRAD_ACCUM_STEPS).backward()
            if batch_number % GRAD_ACCUM_STEPS == 0 or batch_number == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = y.size(0)
            running_loss += loss.item() * bs
            total += bs
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = running_loss / total
        train_accuracy = correct / total
        val_metrics, _, _, _ = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_auc_macro": val_metrics["auc_macro"]
        })

        print(
            f"Epoch {epoch:03d} | Train Acc {train_accuracy*100:.2f}% | "
            f"Val Acc {val_metrics['accuracy']*100:.2f}% | Val F1 {val_metrics['f1_macro']:.5f}"
        )

        improved = val_metrics["accuracy"] > best_accuracy + MIN_DELTA

        if improved:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            atomic_torch_save(cpu_state_dict(model), best_path)
            print(f"  New best model: {best_accuracy*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement: patience {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        pd.DataFrame(history).to_csv(history_path, index=False)

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if not os.path.exists(best_path):
        raise RuntimeError(f"Temporary best weights were not created for Fold {fold}.")

    best_state = torch.load(best_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(best_state)
    del best_state
    final_metrics, y_true, y_pred, probabilities = evaluate(model, val_loader)

    result = {
        "fold": fold,
        "best_epoch": best_epoch,
        "accuracy": final_metrics["accuracy"],
        "micro_accuracy": final_metrics["micro_accuracy"],
        "macro_class_accuracy": final_metrics["macro_class_accuracy"],
        "precision_macro": final_metrics["precision_macro"],
        "precision_micro": final_metrics["precision_micro"],
        "recall_macro": final_metrics["recall_macro"],
        "recall_micro": final_metrics["recall_micro"],
        "f1_macro": final_metrics["f1_macro"],
        "f1_micro": final_metrics["f1_micro"],
        "auc_macro": final_metrics["auc_macro"],
        "auc_micro": final_metrics["auc_micro"],
        "loss": final_metrics["loss"],
        "runtime_minutes": (time.time() - fold_start) / 60
    }

    prediction_df = pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred
    })
    for i, class_name in enumerate(CLASS_NAMES):
        prediction_df[f"prob_{class_name}"] = probabilities[:, i]
    prediction_df.to_csv(predictions_path, index=False)

    np.savetxt(
        os.path.join(fold_dir, "confusion_matrix.csv"),
        confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES)),
        delimiter=",", fmt="%d"
    )

    report = classification_report(
        y_true, y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(fold_dir, "classification_report.csv")
    )

    pd.DataFrame([result]).to_csv(metric_path, index=False)

    save_json(
        {"fold": fold, "status": "complete", "final_metrics": result},
        complete_path
    )

    # Only metrics, predictions, histories and figures remain permanently.
    if os.path.exists(best_path):
        os.remove(best_path)
    del model, optimizer, scaler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nFOLD {fold} COMPLETE")
    return result

def main():
    experiment_start = time.time()
    results = []

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        results.append(train_fold(fold, train_idx, val_idx))
        pd.DataFrame(results).to_csv(
            os.path.join(SAVE_ROOT, "VGG16_5FOLD_RESULTS_PROGRESS.csv"),
            index=False
        )

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_ROOT, "VGG16_5FOLD_RESULTS.csv"), index=False)

    prediction_frames = []
    for fold in range(1, N_SPLITS + 1):
        path = os.path.join(SAVE_ROOT, f"Fold_{fold}", "raw_predictions.csv")
        frame = pd.read_csv(path)
        frame["fold"] = fold
        prediction_frames.append(frame)
    oof = pd.concat(prediction_frames, ignore_index=True)
    if len(oof) != len(dataset):
        raise RuntimeError(f"Expected {len(dataset)} OOF predictions, found {len(oof)}.")
    oof.to_csv(os.path.join(SAVE_ROOT, "VGG16_ALL_5FOLD_OOF_PREDICTIONS.csv"), index=False)
    cm = confusion_matrix(oof["true_label"], oof["predicted_label"], labels=np.arange(NUM_CLASSES))
    cm_table = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_table.index.name = "Actual"
    cm_table.columns.name = "Predicted"
    cm_table.to_csv(os.path.join(SAVE_ROOT, "VGG16_COMBINED_5FOLD_CONFUSION_MATRIX.csv"))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(title="VGG16 - Combined 5-Fold Confusion Matrix", xlabel="Predicted label", ylabel="Actual label")
    ax.set_xticks(np.arange(NUM_CLASSES), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(np.arange(NUM_CLASSES), CLASS_NAMES)
    threshold = cm.max() / 2.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            ax.text(column, row, str(cm[row, column]), ha="center", va="center", color="white" if cm[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_ROOT, "VGG16_COMBINED_5FOLD_CONFUSION_MATRIX.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metric_columns = [
        "accuracy", "micro_accuracy", "macro_class_accuracy",
        "precision_macro", "precision_micro",
        "recall_macro", "recall_micro",
        "f1_macro", "f1_micro",
        "auc_macro", "auc_micro"
    ]

    summary = pd.DataFrame([
        {
            "metric": column,
            "mean": pd.to_numeric(df[column], errors="coerce").mean(),
            "std": pd.to_numeric(df[column], errors="coerce").std(ddof=1)
        }
        for column in metric_columns
    ])

    summary.to_csv(
        os.path.join(SAVE_ROOT, "VGG16_5FOLD_SUMMARY.csv"),
        index=False
    )

    save_json(
        {
            "experiment": "VGG16 5-Fold Stratified CV",
            "dataset": DATA_DIR,
            "fixed_fold_file": FIXED_FOLDS_FILE,
            "seed": SEED,
            "folds": N_SPLITS,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "pretrained": PRETRAINED,
            "device": str(DEVICE),
            "runtime_minutes": (time.time() - experiment_start) / 60
        },
        os.path.join(SAVE_ROOT, "FINAL_EXPERIMENT_RECORD.json")
    )

    print("\n" + "=" * 80)
    print("VGG16 - FINAL 5-FOLD RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\nMEAN ± SD:")
    for _, row in summary.iterrows():
        print(f"{row['metric']:25s}: {row['mean']:.6f} ± {row['std']:.6f}")

    print("\nIMPORTANT:")
    print("Reuse the same FIXED_5_FOLD_SPLITS.npz for ResNet50, DenseNet121 and EfficientNet-B3.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
