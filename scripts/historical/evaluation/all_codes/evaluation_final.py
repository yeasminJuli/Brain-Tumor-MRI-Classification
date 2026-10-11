"""Final evaluation of the proposed model on the untouched cleaned A test set."""

import os
import numpy as np
import pandas as pd
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

from train_model_withCustomized_cnn import HybridCNNBiLSTMAttention, INPUT_SIZE, DEVICE

TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_test_A"
MODEL_PATH = r"C:\user\juli\thesis\original_file\othr folders\clean_proposed_training_results\hybrid_cnn_bilstm_attention_final.pth"
SAVE_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_proposed_evaluation_results_final"
BATCH_SIZE = 16
NUM_WORKERS = 4


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(),
                             persistent_workers=NUM_WORKERS > 0)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    # Support both formats used by the training files:
    # (1) a full checkpoint dictionary and (2) a plain model state_dict.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        metadata = checkpoint
    else:
        model_state = checkpoint
        metadata = {}
    class_names = metadata.get("class_names", test_dataset.classes)
    if list(class_names) != list(test_dataset.classes):
        raise RuntimeError(f"Checkpoint classes {class_names} do not match test classes {test_dataset.classes}.")
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
    per_class = report_df.loc[class_names, ["precision", "recall", "f1-score", "support"]].copy()

    roc_data, per_class_auc = [], []
    for index, class_name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(y_binary[:, index], probs[:, index])
        score = auc(fpr, tpr)
        per_class_auc.append(score)
        roc_data.append((class_name, fpr, tpr, thresholds, score))
    per_class["auc"] = per_class_auc
    per_class.to_csv(os.path.join(SAVE_DIR, "per_class_metrics.csv"))

    prediction_data = {"path": paths_all, "y_true": y_true, "y_pred": y_pred}
    prediction_data.update({f"probability_{name}": probs[:, i] for i, name in enumerate(class_names)})
    pd.DataFrame(prediction_data).to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(SAVE_DIR, "confusion_matrix_counts.csv"))
    plt.figure(figsize=(7.2, 6.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="white")
    plt.xlabel("Predicted label"); plt.ylabel("Actual label")
    plt.title("Proposed Model - Confusion Matrix")
    plt.tight_layout(); plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_blues.png"), dpi=300); plt.close()

    x, width = np.arange(len(class_names)), 0.22
    plt.figure(figsize=(9, 6.2))
    plt.bar(x - width, per_class["precision"], width, label="Precision",
            color="#F0D0D5", edgecolor="#B97A86", linewidth=0.8)
    plt.bar(x, per_class["recall"], width, label="Recall",
            color="#A4D8E8", edgecolor="#528FA3", linewidth=0.8)
    plt.bar(x + width, per_class["f1-score"], width, label="F1-score",
            color="#A8D5A2", edgecolor="#5E9660", linewidth=0.8)
    plt.xticks(x, class_names); plt.xlabel("Class"); plt.ylabel("Score"); plt.ylim(0, 1.05)
    plt.title("Proposed Model - Class-wise Metrics")
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
    plt.xlim(0, 1); plt.ylim(0, 1.01); plt.title("Proposed Model - One-vs-Rest ROC Curves")
    plt.legend(loc="lower right"); plt.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "roc_auc_curves.png"), dpi=300); plt.close()

    print(f"Final evaluation completed on {len(test_dataset)} untouched test images.")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("Results saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
