import torch
import torch.nn as nn
import gc
import pandas as pd
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

# =========================
# PATHS
# =========================
TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_test_A"
MODEL_PATH = r"C:\user\juli\thesis\original_file\othr folders\clean_vgg16_training_results\vgg16_best.pth"
SAVE_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_vgg16_evaluation_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS (VGG16)
# =========================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

classes = test_dataset.classes
NUM_CLASSES = len(classes)

# =========================
# LOAD VGG16
# =========================
# Load on CPU first. The training checkpoint also contains a large optimizer
# state, which must not be copied onto the 4 GB GPU during evaluation.
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_state = checkpoint["model_state_dict"]
    metadata = {
        key: checkpoint.get(key)
        for key in ("epoch", "val_loss", "val_accuracy", "class_names")
    }
else:
    model_state = checkpoint
    metadata = {}
classes = metadata.get("class_names") or test_dataset.classes
if list(classes) != list(test_dataset.classes):
    raise RuntimeError("Checkpoint and test-set class mappings do not match.")
NUM_CLASSES = len(classes)
model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
model.load_state_dict(model_state)
del model_state, checkpoint
gc.collect()
model.to(device)
model.eval()

# =========================
# EVALUATION
# =========================
all_labels = []
all_preds = []
all_probs = []
all_paths = []
test_loss_sum = 0.0
criterion = nn.CrossEntropyLoss()
offset = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating VGG16", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss_sum += criterion(outputs, labels).item() * labels.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(path for path, _ in test_dataset.samples[offset:offset + labels.size(0)])
        offset += labels.size(0)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# =========================
# OVERALL MODEL ACCURACY
# =========================
overall_accuracy = accuracy_score(all_labels, all_preds)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0)
y_true_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
macro_auc = roc_auc_score(y_true_bin, all_probs, average="macro", multi_class="ovr")
overall_metrics = {
    "test_loss": test_loss_sum / len(all_labels), "accuracy": overall_accuracy,
    "macro_precision": macro_precision, "macro_recall": macro_recall,
    "macro_f1": macro_f1, "macro_auc_ovr": macro_auc,
    "best_training_epoch": metadata.get("epoch"),
    "best_validation_loss": metadata.get("val_loss"),
    "best_validation_accuracy": metadata.get("val_accuracy"),
}
pd.DataFrame([overall_metrics]).to_csv(os.path.join(SAVE_DIR, "overall_metrics.csv"), index=False)
print("\nOverall metrics:", overall_metrics)

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(all_labels, all_preds, target_names=classes)
with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
report_table = classification_report(
    all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0
)
pd.DataFrame(report_table).T.to_csv(
    os.path.join(SAVE_DIR, "classification_report.csv")
)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    os.path.join(SAVE_DIR, "confusion_matrix_counts.csv"))

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - VGG16")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_blues.png"), dpi=300)
plt.close()

# =========================
# ROC CURVE (Glioma solid, others dotted)
# =========================
plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    if classes[i].lower() == "glioma":
        plt.plot(fpr, tpr, linewidth=2,
                 label=f"{classes[i]} (AUC={roc_auc:.3f})")
    else:
        plt.plot(fpr, tpr, linestyle=":",
                 label=f"{classes[i]} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - VGG16")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_auc_curves.png"), dpi=300)
plt.close()

# =========================
# METRICS BAR CHART
# =========================
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_preds,
    labels=range(NUM_CLASSES),
    zero_division=0
)

x = np.arange(NUM_CLASSES)
width = 0.18

plt.figure(figsize=(9, 6.2))
plt.bar(x - width, precision, width, label="Precision", color="#F0D0D5", edgecolor="#B97A86")
plt.bar(x, recall, width, label="Recall", color="#A4F8D5", edgecolor="#528FA3")
plt.bar(x + width, f1, width, label="F1-score", color="#90EE90", edgecolor="#5E9660")

plt.xticks(x, classes)
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Precision, Recall, F1-score per Class (VGG16)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig(os.path.join(SAVE_DIR, "classwise_metrics_bar_chart.png"), dpi=300, bbox_inches="tight")
plt.close()

per_class_auc = []
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    per_class_auc.append(auc(fpr, tpr))
per_class_metrics = pd.DataFrame({
    "Class": classes, "Precision": precision, "Recall": recall,
    "F1-score": f1, "AUC": per_class_auc,
})
per_class_metrics.to_csv(os.path.join(SAVE_DIR, "per_class_metrics.csv"), index=False)
prediction_data = {"path": all_paths, "y_true": all_labels, "y_pred": all_preds}
prediction_data.update({f"probability_{name}": all_probs[:, i] for i, name in enumerate(classes)})
pd.DataFrame(prediction_data).to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)

print("\n🎉 VGG16 evaluation completed successfully!")
