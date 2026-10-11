"""Final evaluation of the proposed model on the untouched cleaned A test set."""

import os
import ast
import json
import time
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
A_ROOT = Path(r"C:/user/juli/thesis/original_file/othr folders")
A_MODEL = A_ROOT / "training_outputs_proposed_model_orgnl/training_outputs/hybrid_cnn_bilstm_attention_final.pth"
C_MODEL = ROOT / "training_outputs_C/hybrid_cnn_bilstm_attention_final.pth"
A_TEST = A_ROOT / "preprocessed_testt"
C_TEST = ROOT / "preprocessed_test_C"
RESULTS = ROOT
CANONICAL = ["glioma", "meningioma", "notumor", "pituitary"]
ALIASES = {"Glioma": "glioma", "Meningioma": "meningioma", "Healthy": "notumor",
           "Pituitary Macroadenoma": "pituitary", **{x: x for x in CANONICAL}}
INPUT_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_CLASSES = 4
# Import only model definitions, avoiding training-script filesystem side effects.
model_source = (Path(__file__).resolve().parents[3] / 'src' / 'models' / 'proposed.py').read_text(encoding="utf-8")
nodes = [n for n in ast.parse(model_source).body if isinstance(n, ast.ClassDef) and n.name in
         {"ResidualBlock", "CNNFeatureExtractor", "AdditiveAttention", "HybridCNNBiLSTMAttention"}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), "model_definitions", "exec"))


def canonical_dataset(folder, transform=None):
    dataset = datasets.ImageFolder(str(folder), transform=transform)
    mapped = [ALIASES[name] for name in dataset.classes]
    assert set(mapped) == set(CANONICAL)
    dataset.samples = [(path, CANONICAL.index(mapped[label])) for path, label in dataset.samples]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    dataset.classes = CANONICAL[:]
    dataset.class_to_idx = dict(zip(CANONICAL, range(4)))
    return dataset


def checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint if "model_state_dict" in checkpoint else {}
    names = metadata.get("class_names", CANONICAL)
    if [ALIASES[name] for name in names] != CANONICAL:
        raise ValueError(f"Unexpected checkpoint output order: {names}")
    return checkpoint.get("model_state_dict", checkpoint), metadata


def evaluate():
    os.makedirs(SAVE_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = canonical_dataset(TEST_DIR, transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    model_state, metadata = checkpoint_state(MODEL_PATH)
    class_names = CANONICAL
    model = HybridCNNBiLSTMAttention(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    labels_all, predictions_all, probabilities_all, paths_all = [], [], [], []
    offset = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating untouched test set", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits, _ = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(predictions.cpu().numpy())
            probabilities_all.extend(probabilities.cpu().numpy())
            paths_all.extend(path for path, _ in test_dataset.samples[offset:offset + labels.size(0)])
            offset += labels.size(0)

    y_true, y_pred, probs = np.asarray(labels_all), np.asarray(predictions_all), np.asarray(probabilities_all)
    y_binary = label_binarize(y_true, classes=np.arange(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {
        "test_loss": loss_sum / len(y_true), "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision, "macro_recall": recall, "macro_f1": f1,
        "macro_auc_ovr": roc_auc_score(y_binary, probs, average="macro", multi_class="ovr"),
        "best_training_epoch": metadata.get("epoch"),
        "best_validation_loss": metadata.get("val_loss"),
        "best_validation_accuracy": metadata.get("val_accuracy"),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(SAVE_DIR, "overall_metrics.csv"), index=False)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(SAVE_DIR, "classification_report.csv"))
    (Path(SAVE_DIR) / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, zero_division=0), encoding="utf-8")
    per_class = report_df.loc[class_names, ["precision", "recall", "f1-score", "support"]].copy()

    roc_data, per_class_auc = [], []
    for index, class_name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(y_binary[:, index], probs[:, index])
        score = auc(fpr, tpr)
        per_class_auc.append(score)
        roc_data.append((class_name, fpr, tpr, thresholds, score))
    pd.DataFrame({"class": class_names, "auc": per_class_auc}).to_csv(Path(SAVE_DIR) / "auc_metrics.csv", index=False)
    per_class["auc"] = per_class_auc
    counts = confusion_matrix(y_true, y_pred, labels=np.arange(4))
    tp = np.diag(counts)
    fn = counts.sum(axis=1) - tp
    fp = counts.sum(axis=0) - tp
    tn = counts.sum() - tp - fn - fp
    per_class["specificity"] = tn / (tn + fp)
    per_class["TP"] = tp
    per_class["FN"] = fn
    per_class["FP"] = fp
    per_class["TN"] = tn
    per_class.to_csv(os.path.join(SAVE_DIR, "per_class_metrics.csv"))

    prediction_data = {"path": paths_all, "y_true": y_true, "y_pred": y_pred}
    prediction_data["true_class"] = [class_names[i] for i in y_true]
    prediction_data["predicted_class"] = [class_names[i] for i in y_pred]
    prediction_data.update({f"probability_{name}": probs[:, i] for i, name in enumerate(class_names)})
    pd.DataFrame(prediction_data).to_csv(os.path.join(SAVE_DIR, "predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(SAVE_DIR, "confusion_matrix.csv"))
    plt.figure(figsize=(7.2, 6.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="white")
    plt.xlabel("Predicted label"); plt.ylabel("Actual label")
    plt.title(JOB_NAME + " - Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"), dpi=300); plt.close()

    x, width = np.arange(len(class_names)), 0.22
    plt.figure(figsize=(9, 6.2))
    plt.bar(x - width, per_class["precision"], width, label="Precision",
            color="#F0D0D5", edgecolor="#B97A86", linewidth=0.8)
    plt.bar(x, per_class["recall"], width, label="Recall",
            color="#A4D8E8", edgecolor="#528FA3", linewidth=0.8)
    plt.bar(x + width, per_class["f1-score"], width, label="F1-score",
            color="#A8D5A2", edgecolor="#5E9660", linewidth=0.8)
    plt.xticks(x, class_names); plt.xlabel("Class"); plt.ylabel("Score"); plt.ylim(0, 1.05)
    plt.title(JOB_NAME + " - Class-wise Metrics")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
               frameon=False, borderaxespad=0)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(os.path.join(SAVE_DIR, "classwise_metrics_bar_chart.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 6.2))
    roc_rows = []
    for class_name, fpr, tpr, thresholds, score in roc_data:
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={score:.4f})")
        roc_rows.extend({"class": class_name, "fpr": xv, "tpr": yv, "threshold": th}
                        for xv, yv, th in zip(fpr, tpr, thresholds))
    pd.DataFrame(roc_rows).to_csv(os.path.join(SAVE_DIR, "roc_curve_points.csv"), index=False)
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.xlim(0, 1); plt.ylim(0, 1.01); plt.title(JOB_NAME + " - One-vs-Rest ROC Curves")
    plt.legend(loc="lower right"); plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "roc_curves.png"), dpi=300); plt.close()

    print(f"Final evaluation completed on {len(test_dataset)} test images.")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("Results saved to:", SAVE_DIR)


def main():
    global TEST_DIR, MODEL_PATH, SAVE_DIR, JOB_NAME
    torch.set_num_threads(4)
    jobs = [("evaluation_results_C", C_MODEL, C_TEST, 228),
            ("evaluation_results_crosscheck_A_C", A_MODEL, C_TEST, 228),
            ("evaluation_results_crosscheck_C_A", C_MODEL, A_TEST, 1311)]
    for name, checkpoint, folder, expected in jobs:
        dataset = canonical_dataset(folder)
        assert len(dataset) == expected, (name, len(dataset))
        state, _ = checkpoint_state(checkpoint)
        HybridCNNBiLSTMAttention().load_state_dict(state, strict=True)
    print("Preflight passed: checkpoint architectures, class mappings and all test counts verified.", flush=True)
    import sys
    if "--check-only" in sys.argv:
        return
    log = ROOT / "training_log.txt"
    deadline = time.monotonic() + 4 * 60 * 60
    while True:
        status = log.read_text(encoding="utf-8-sig", errors="replace")
        if "Training complete." in status and "Total training time:" in status:
            break
        if "Traceback (most recent call last)" in status:
            raise RuntimeError("Training failed; evaluation will not run on an incomplete run.")
        if time.monotonic() > deadline:
            raise TimeoutError("Training completion was not confirmed within four hours.")
        print("Waiting for training to finish before evaluating the saved best checkpoint...", flush=True)
        time.sleep(30)
    RESULTS.mkdir(exist_ok=True)
    summary = []
    for JOB_NAME, MODEL_PATH, TEST_DIR, expected in jobs:
        SAVE_DIR = str(RESULTS / JOB_NAME)
        print(f"START: {JOB_NAME}", flush=True)
        evaluate()
        metrics = pd.read_csv(Path(SAVE_DIR) / "overall_metrics.csv").iloc[0].to_dict()
        summary.append({"evaluation": JOB_NAME, "test_images": expected, **metrics})
        metadata = {"checkpoint": str(MODEL_PATH), "checkpoint_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
                    "test_directory": str(TEST_DIR), "metrics": metrics, "class_order": CANONICAL, "folder_aliases": ALIASES,
                    "device": str(DEVICE), "torch_version": str(torch.__version__),
                    "notes": "C checkpoint selected on file-level validation after augmentation; augmented relatives overlap between training and validation. A/C cross-dataset image and patient overlap has not been audited. Cross-dataset results do not establish independent patient-level validation."}
        (Path(SAVE_DIR) / "evaluation_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(RESULTS / "evaluation_summary.csv", index=False)
    print("ALL THREE EVALUATIONS COMPLETE", flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

