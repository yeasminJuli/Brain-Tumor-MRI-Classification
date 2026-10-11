import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# =========================
# CONFIGURATION
# =========================

TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_test_A"
MODEL_PATH = r"C:\user\juli\thesis\original_file\othr folders\clean_resnet50_training_results\resnet50_best.pth"
OUTPUT_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_resnet50_evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORMS
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
# DATASET + LOADER
# =========================

test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes

# =========================
# LOAD MODEL
# =========================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    metadata = checkpoint
else:
    state_dict = checkpoint
    metadata = {}
class_names = metadata.get("class_names", test_dataset.classes)
if list(class_names) != list(test_dataset.classes):
    raise RuntimeError("Checkpoint and test-set class mappings do not match.")
NUM_CLASSES = len(class_names)
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# =========================
# EVALUATION
# =========================

all_labels = []
all_preds = []
all_probs = []
all_paths = []
test_loss_sum = 0.0
criterion = torch.nn.CrossEntropyLoss()
offset = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating ResNet50", unit="batch"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        test_loss_sum += criterion(outputs, labels).item() * labels.size(0)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_paths.extend(path for path, _ in test_dataset.samples[offset:offset + labels.size(0)])
        offset += labels.size(0)

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -----------------------
# Overall Accuracy
# -----------------------
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
pd.DataFrame([overall_metrics]).to_csv(os.path.join(OUTPUT_DIR, "overall_metrics.csv"), index=False)
print("\nOverall metrics:", overall_metrics)

# -----------------------
# Classification Report
# -----------------------
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)
print("✔ Classification Report Saved")

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("ResNet50 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix_counts.csv"))
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_blues.png"), dpi=300)
plt.close()
print("✔ Confusion Matrix Saved")

# -----------------------
# ROC Curve (1 vs Rest)
# -----------------------
plt.figure(figsize=(8, 6))

for i, cls in enumerate(class_names):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    if cls.lower() == "glioma":
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})", linestyle='-')  # solid
    else:
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})", linestyle='--')  # dotted

plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ResNet50 ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_auc_curves.png"), dpi=300)
plt.close()
print("✔ ROC Curve Saved")

# -----------------------
# Precision, Recall, F1 Bar Chart
# -----------------------
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=range(NUM_CLASSES))
x = np.arange(NUM_CLASSES)
width = 0.25

plt.figure(figsize=(9, 6.2))
plt.bar(x - width, precision, width, color="#F0D0D5", edgecolor="#B97A86", label="Precision")
plt.bar(x, recall, width, color="#A4F8D5", edgecolor="#528FA3", label="Recall")
plt.bar(x + width, f1, width, color="#90EE90", edgecolor="#5E9660", label="F1-score")
plt.xticks(x, class_names)
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Precision, Recall, F1 per Class")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig(os.path.join(OUTPUT_DIR, "classwise_metrics_bar_chart.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✔ Precision-Recall-F1 Bar Chart Saved")

# -----------------------
# Save All Metrics to CSV
# -----------------------
metrics_df = pd.DataFrame({
    "Class": class_names,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "AUC": [auc(*roc_curve(y_true_bin[:, i], all_probs[:, i])[:2]) for i in range(NUM_CLASSES)]
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "per_class_metrics.csv"), index=False)
prediction_data = {"path": all_paths, "y_true": all_labels, "y_pred": all_preds}
prediction_data.update({f"probability_{name}": all_probs[:, i] for i, name in enumerate(class_names)})
pd.DataFrame(prediction_data).to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
print("✔ Metrics CSV Saved")

print("\n🎉 ResNet50 Evaluation Completed Successfully!")
