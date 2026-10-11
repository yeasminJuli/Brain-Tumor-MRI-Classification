"""Leakage-safe standalone EfficientNet-B3 training on cleaned Dataset A.

Uses only cleaned original preprocessed images with a deterministic stratified
80/20 train/validation split. Final test evaluation is separate.
"""
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm

ORIGINAL_DIR = Path(r"C:\user\juli\thesis\original_file\othr folders\clean_deduplicated_datasets\preprocessed_train_A")
OUTPUT_DIR = Path(r"C:\user\juli\thesis\original_file\othr folders\clean_efficientnet_b3_training_results")
MODEL_PATH = OUTPUT_DIR / "efficientnet_b3_best.pth"
MODEL_NAME = "EfficientNet-B3"
NUM_CLASSES = 4
IMAGE_SIZE = 300
BATCH_SIZE = 16
MAX_EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-4
SEED = 42
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImagePathDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return transform(image), label


def build_datasets():
    originals = datasets.ImageFolder(ORIGINAL_DIR)

    indices = np.arange(len(originals.samples))
    labels = np.asarray(originals.targets)
    train_indices, val_indices = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=labels
    )
    training_samples = [originals.samples[i] for i in train_indices]
    validation_samples = [originals.samples[i] for i in val_indices]
    train_paths = {path for path, _ in training_samples}
    validation_paths = {path for path, _ in validation_samples}
    if train_paths & validation_paths:
        raise RuntimeError("Leakage detected between training and validation.")

    split_rows = []
    for i in train_indices:
        path, label = originals.samples[i]
        split_rows.append({"path": path, "class": originals.classes[label], "split": "train"})
    for i in val_indices:
        path, label = originals.samples[i]
        split_rows.append({"path": path, "class": originals.classes[label], "split": "validation"})
    pd.DataFrame(split_rows).to_csv(OUTPUT_DIR / "train_validation_split.csv", index=False)

    print(f"Original images: {len(originals):,}")
    print(f"Training originals: {len(train_indices):,}")
    print(f"Training images (no offline augmentation): {len(training_samples):,}")
    print(f"Untouched validation images: {len(validation_samples):,}")
    print("Leakage check: PASS - training and validation are disjoint")
    return ImagePathDataset(training_samples), ImagePathDataset(validation_samples), originals.classes


def build_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_dataset, val_dataset, class_names = build_datasets()
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       pin_memory=torch.cuda.is_available(),
                       persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    no_improvement = 0

    print(f"{MODEL_NAME}: device={DEVICE}, batch={BATCH_SIZE}, max epochs={MAX_EPOCHS}, patience={PATIENCE}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} Train", unit="batch"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_loss, train_accuracy = loss_sum / total, correct / total

        model.eval()
        loss_sum = correct = total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} Validation", unit="batch"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_sum += loss.item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_loss, val_accuracy = loss_sum / total, correct / total

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "val_loss": val_loss,
                        "val_accuracy": val_accuracy})
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, acc={train_accuracy:.4f}; "
              f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}")

        if val_loss < best_loss:
            best_loss, best_epoch, no_improvement = val_loss, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "class_names": class_names, "val_loss": val_loss,
                        "val_accuracy": val_accuracy, "seed": SEED,
                        "model_name": MODEL_NAME}, MODEL_PATH)
            print("  Best validation loss improved; checkpoint saved.")
        else:
            no_improvement += 1
            print(f"  No validation-loss improvement: {no_improvement}/{PATIENCE}")
            if no_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(frame.epoch, frame.train_loss, label="Train")
    axes[0].plot(frame.epoch, frame.val_loss, label="Validation")
    axes[0].set(title=f"{MODEL_NAME} Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.train_accuracy * 100, label="Train")
    axes[1].plot(frame.epoch, frame.val_accuracy * 100, label="Validation")
    axes[1].set(title=f"{MODEL_NAME} Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=300)
    plt.close(fig)
    print(f"Training complete. Best epoch={best_epoch}; best validation loss={best_loss:.6f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"Total training time: {(time.time() - started) / 60:.2f} minutes")
