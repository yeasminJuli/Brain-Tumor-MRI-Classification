#r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\traing_outputs_inceptionV3\best_model.pth" 
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import numpy as np
import os

# -------------------------------
# Paths
# -------------------------------
test_dir = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\preprocessed_testt"
model_path = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\traing_outputs_inceptionV3\best_model.pth"  # your trained model
output_dir = r"C:\Users\kaziFARZANA\Documents\THESIS\BRAIN\dataset\two\Brain Tumor data - ayesha\evaluation_outputs_inception"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Test Transform for InceptionV3
# -------------------------------
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Dataset + Loader
# -------------------------------
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)

# -------------------------------
# Load InceptionV3 with AuxLogits=True
# -------------------------------
model = models.inception_v3(weights=None, aux_logits=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Note: Do NOT redefine AuxLogits.fc, we will ignore its weights

# Load trained weights, ignore AuxLogits to avoid mismatch
state_dict = torch.load(model_path, map_location=device)
state_dict = {k: v for k, v in state_dict.items() if "AuxLogits" not in k}
model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.eval()

# -------------------------------
# Evaluation Loop
# -------------------------------
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)  # AuxLogits ignored automatically in eval mode
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -------------------------------
# Overall Accuracy
# -------------------------------
overall_accuracy = (all_preds == all_labels).mean() * 100
print(f"\n✔ Overall Accuracy: {overall_accuracy:.2f}%")
with open(os.path.join(output_dir, "overall_accuracy.txt"), "w") as f:
    f.write(f"Overall Test Accuracy: {overall_accuracy:.2f}%\n")

# -------------------------------
# Classification Report
# -------------------------------
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
report_str = classification_report(all_labels, all_preds, target_names=class_names)
with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report_str)
print("✔ Classification Report Saved")

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - InceptionV3")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()
print("✔ Confusion Matrix Saved")

# -------------------------------
# Matrix Bar Chart from Confusion Matrix
# -------------------------------
plt.figure(figsize=(8, 6))
cm_sum = cm.sum(axis=1)
cm_percentage = cm / cm_sum[:, np.newaxis] * 100
for i, class_name in enumerate(class_names):
    plt.bar(np.arange(num_classes) + i*0.2, cm_percentage[i], width=0.2, label=class_name)
plt.xticks(np.arange(num_classes) + 0.3, class_names)
plt.ylabel("Percentage (%)")
plt.xlabel("Predicted Class")
plt.title("Confusion Matrix - Bar Chart")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_bar.png"))
plt.close()
print("✔ Confusion Matrix Bar Chart Saved")

# -------------------------------
# ROC Curve (1 vs Rest)
# -------------------------------
plt.figure(figsize=(8, 6))
colors = ['blue', 'orange', 'green', 'red']
linestyles = ['-', '--', '--', '--']  # Glioma solid, rest dotted
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i],
             label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - InceptionV3")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()
print("✔ ROC Curve Saved")

# -------------------------------
# Slim Bar Chart (Precision, Recall, F1)
# -------------------------------
precision = [report_dict[c]['precision'] for c in class_names]
recall    = [report_dict[c]['recall'] for c in class_names]
f1        = [report_dict[c]['f1-score'] for c in class_names]

x = np.arange(num_classes)
width = 0.18

plt.figure(figsize=(8,5))
plt.bar(x - width, precision, width, color="#F0D0D5", label='Precision')
plt.bar(x, recall, width, color="#A4F8D5", label='Recall')
plt.bar(x + width, f1, width, color='#90EE90', label='F1-score')
plt.xticks(x, class_names)
plt.ylabel("Score")
plt.ylim(0,1.0)
plt.title("Precision, Recall, F1-score per Class")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_barchart.png"), bbox_inches='tight')
plt.close()
print("✔ Slim Precision-Recall-F1 Bar Chart Saved")

print("\n🎉 InceptionV3 Evaluation Completed Successfully!")
