import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_test_A"
MODEL_PATH = r"C:\user\juli\thesis\original_file\othr folders\clean_densenet121_training_results\densenet121_best.pth"
SAVE_DIR = r"C:\user\juli\thesis\original_file\othr folders\clean_densenet121_evaluation_results"

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 16

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
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
test_dataset = ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=torch.cuda.is_available())

# =========================
# MODEL
# =========================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_state = checkpoint["model_state_dict"]
    metadata = checkpoint
else:
    model_state = checkpoint
    metadata = {}
CLASS_NAMES = metadata.get("class_names", test_dataset.classes)
if list(CLASS_NAMES) != list(test_dataset.classes):
    raise RuntimeError("Checkpoint and test-set class mappings do not match.")
NUM_CLASSES = len(CLASS_NAMES)
model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.load_state_dict(model_state)
model.to(DEVICE)
model.eval()

# =========================
# INFERENCE
# =========================
all_labels = []
all_preds = []
all_probs = []
test_loss_sum = 0.0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        test_loss_sum += criterion(outputs, labels).item() * labels.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# =========================
# OVERALL ACCURACY
# =========================
overall_accuracy = accuracy_score(all_labels, all_preds)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="macro", zero_division=0)
y_true_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
macro_auc = roc_auc_score(y_true_bin, all_probs, average="macro", multi_class="ovr")
overall_metrics = {
    "test_loss": test_loss_sum / len(all_labels),
    "accuracy": overall_accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "macro_auc_ovr": macro_auc,
    "best_training_epoch": metadata.get("epoch"),
    "best_validation_loss": metadata.get("val_loss"),
    "best_validation_accuracy": metadata.get("val_accuracy"),
}
pd.DataFrame([overall_metrics]).to_csv(os.path.join(SAVE_DIR, "overall_metrics.csv"), index=False)
print("\nOverall metrics:", overall_metrics)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(all_labels, all_preds, labels=range(NUM_CLASSES))
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
    os.path.join(SAVE_DIR, "confusion_matrix_counts.csv"))
plt.figure(figsize=(7.2, 6.0))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, linecolor="white")
plt.title("DenseNet121 - Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_blues.png"), dpi=300)
plt.close()

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES,
    output_dict=True
)
summary_df = pd.DataFrame(report).transpose()
summary_df.to_csv(os.path.join(SAVE_DIR, "classification_report.csv"))

# =========================
# METRIC BAR CHART (ResNet50 Style)
# =========================
precision = summary_df.loc[CLASS_NAMES, 'precision']
recall = summary_df.loc[CLASS_NAMES, 'recall']
f1 = summary_df.loc[CLASS_NAMES, 'f1-score']
auc_scores = []
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    auc_scores.append(auc(fpr, tpr))
per_class_metrics = summary_df.loc[CLASS_NAMES, ['precision', 'recall', 'f1-score', 'support']].copy()
per_class_metrics['auc'] = auc_scores
per_class_metrics.to_csv(os.path.join(SAVE_DIR, "per_class_metrics.csv"))
prediction_data = {"y_true": all_labels, "y_pred": all_preds}
prediction_data.update({f"probability_{name}": all_probs[:, i] for i, name in enumerate(CLASS_NAMES)})
pd.DataFrame(prediction_data).to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)

x = np.arange(NUM_CLASSES)
width = 0.22

plt.figure(figsize=(9,6.2))

plt.bar(x - width, precision, width, label="Precision", color="#F0D0D5", edgecolor="#B97A86")
plt.bar(x, recall, width, label="Recall", color="#A4D8E8", edgecolor="#528FA3")
plt.bar(x + width, f1, width, label="F1-score", color="#A8D5A2", edgecolor="#5E9660")

plt.xticks(x, CLASS_NAMES)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("DenseNet121 - Class-wise Metrics")

plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=3,
    frameon=False
)

plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.savefig(os.path.join(SAVE_DIR, "classwise_metrics_bar_chart.png"), dpi=300, bbox_inches="tight")
plt.close()


# =========================
# ROC CURVE
# =========================
plt.figure(figsize=(7,6))

for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={roc_auc:.4f})")

plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (DenseNet-121)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_auc_curves.png"), dpi=300)
plt.close()

print("\nEvaluation complete. All results saved successfully.")
