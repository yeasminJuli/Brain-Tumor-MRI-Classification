"""Evaluate the trained EfficientNet-B3 on the untouched cleaned A test set."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_test_A"
MODEL_PATH = r"C:\user\juli\thesis\original_file\othr folders\clean_efficientnet_b3_training_results\efficientnet_b3_best.pth"
SAVE_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_efficientnet_b3_evaluation_results"
IMAGE_SIZE = 300
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"EfficientNet-B3 checkpoint not found:\n{MODEL_PATH}\n"
            "Finish efficient.py training before running this evaluation."
        )
    os.makedirs(SAVE_DIR, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        metadata = checkpoint
    else:
        model_state = checkpoint
        metadata = {}

    class_names = metadata.get("class_names", test_dataset.classes)
    if list(class_names) != list(test_dataset.classes):
        raise RuntimeError(
            f"Checkpoint classes {class_names} do not match test classes "
            f"{test_dataset.classes}."
        )

    # weights=None avoids downloading ImageNet weights again during evaluation.
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(model_state)
    model = model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    labels_all, predictions_all, probabilities_all, paths_all = [], [], [], []
    offset = 0
    with torch.no_grad():
        for images, labels in tqdm(
            test_loader, desc="Evaluating EfficientNet-B3", unit="batch"
        ):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            labels_all.extend(labels.cpu().numpy())
            predictions_all.extend(predictions.cpu().numpy())
            probabilities_all.extend(probabilities.cpu().numpy())
            paths_all.extend(
                path for path, _ in test_dataset.samples[offset : offset + labels.size(0)]
            )
            offset += labels.size(0)

    y_true = np.asarray(labels_all)
    y_pred = np.asarray(predictions_all)
    probabilities = np.asarray(probabilities_all)
    y_binary = label_binarize(y_true, classes=np.arange(len(class_names)))

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    overall_metrics = {
        "test_loss": loss_sum / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_auc_ovr": roc_auc_score(
            y_binary, probabilities, average="macro", multi_class="ovr"
        ),
        "best_training_epoch": metadata.get("epoch"),
        "best_validation_loss": metadata.get("val_loss"),
        "best_validation_accuracy": metadata.get("val_accuracy"),
    }
    pd.DataFrame([overall_metrics]).to_csv(
        os.path.join(SAVE_DIR, "overall_metrics.csv"), index=False
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(os.path.join(SAVE_DIR, "classification_report.csv"))
    per_class = report_df.loc[
        class_names, ["precision", "recall", "f1-score", "support"]
    ].copy()

    roc_data = []
    per_class_auc = []
    for index, class_name in enumerate(class_names):
        fpr, tpr, thresholds = roc_curve(
            y_binary[:, index], probabilities[:, index]
        )
        class_auc = auc(fpr, tpr)
        per_class_auc.append(class_auc)
        roc_data.append((class_name, fpr, tpr, thresholds, class_auc))
    per_class["auc"] = per_class_auc
    per_class.to_csv(os.path.join(SAVE_DIR, "per_class_metrics.csv"))

    prediction_data = {"path": paths_all, "y_true": y_true, "y_pred": y_pred}
    prediction_data.update(
        {
            f"probability_{class_name}": probabilities[:, index]
            for index, class_name in enumerate(class_names)
        }
    )
    pd.DataFrame(prediction_data).to_csv(
        os.path.join(SAVE_DIR, "test_predictions.csv"), index=False
    )

    # Standard publication-style Blues confusion matrix with integer counts.
    matrix = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(class_names))
    )
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        os.path.join(SAVE_DIR, "confusion_matrix_counts.csv")
    )
    plt.figure(figsize=(7.2, 6.0))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("EfficientNet-B3 - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, "confusion_matrix_blues.png"), dpi=300
    )
    plt.close()

    # Pastel class-wise bar chart with the legend below, not over the bars.
    x = np.arange(len(class_names))
    width = 0.22
    plt.figure(figsize=(9, 6.2))
    plt.bar(
        x - width,
        per_class["precision"],
        width,
        label="Precision",
        color="#F0D0D5",
        edgecolor="#B97A86",
        linewidth=0.8,
    )
    plt.bar(
        x,
        per_class["recall"],
        width,
        label="Recall",
        color="#A4D8E8",
        edgecolor="#528FA3",
        linewidth=0.8,
    )
    plt.bar(
        x + width,
        per_class["f1-score"],
        width,
        label="F1-score",
        color="#A8D5A2",
        edgecolor="#5E9660",
        linewidth=0.8,
    )
    plt.xticks(x, class_names)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("EfficientNet-B3 - Class-wise Metrics")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        borderaxespad=0,
    )
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(
        os.path.join(SAVE_DIR, "classwise_metrics_bar_chart.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # One-vs-rest ROC curves with class AUC values.
    plt.figure(figsize=(7.5, 6.2))
    roc_rows = []
    for class_name, fpr, tpr, thresholds, class_auc in roc_data:
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} (AUC={class_auc:.4f})",
        )
        roc_rows.extend(
            {
                "class": class_name,
                "fpr": fpr_value,
                "tpr": tpr_value,
                "threshold": threshold,
            }
            for fpr_value, tpr_value, threshold in zip(fpr, tpr, thresholds)
        )
    pd.DataFrame(roc_rows).to_csv(
        os.path.join(SAVE_DIR, "roc_curve_points.csv"), index=False
    )
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1.01)
    plt.title("EfficientNet-B3 - One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "roc_auc_curves.png"), dpi=300)
    plt.close()

    print(
        f"EfficientNet-B3 evaluation completed on {len(test_dataset)} "
        "untouched test images."
    )
    for metric_name, value in overall_metrics.items():
        print(f"{metric_name}: {value}")
    print("Results saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
