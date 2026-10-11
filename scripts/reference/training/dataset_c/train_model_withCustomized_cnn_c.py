# final_hybrid_model_with_summary.py
# PyTorch implementation: CNN(with residual blocks) -> BiLSTM -> Additive Attention -> FC
# Saves per-epoch metrics to CSV, plots curves, and saves a layer-by-layer parameter summary.

import os
from pathlib import Path
import csv
import time
import re
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------
# Config / Hyperparameters
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (256, 256)          # user confirmed preprocessing -> 256x256
NUM_CLASSES = 4                  # glioma, meningioma, pituitary, no_tumor
BATCH_SIZE = 16
LR = 1e-4
NUM_EPOCHS = 60
PATIENCE = 7
MIN_EPOCHS_BEFORE_EARLY_STOPPING = 30
SEED = 42
NUM_WORKERS = 4
ROOT = Path(__file__).resolve().parent
SAVE_DIR = str(ROOT / "training_outputs_C")
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "hybrid_cnn_bilstm_attention_final.pth")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
SUMMARY_CSV = os.path.join(SAVE_DIR, "model_summary.csv")
SUMMARY_TXT = os.path.join(SAVE_DIR, "model_summary.txt")

# ------------------------------
# Transforms
# Note: you said preprocessing produced grayscale images (1 channel),
# but classification expects 3 channels (RGB). We replicate grayscale to 3 channels.
# ------------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # replicate to 3 channels
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class ImagePathDataset(Dataset):
    """Dataset backed by explicit (path, label) samples."""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.copy()
        return self.transform(image), label

# ------------------------------
# Model components
# ------------------------------
class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers; uses 1x1 conv for skip if channels differ.
    Optionally includes a MaxPool inside the block to reduce spatial size.
    """
    def __init__(self, in_channels, out_channels, use_pool=False):
        super().__init__()
        self.use_pool = use_pool
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # skip conv if channel dims change
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)

        out += identity
        out = self.relu(out)
        out = self.pool(out)
        return out

class CNNFeatureExtractor(nn.Module):
    """
    5 residual blocks:
      Block1: 3ch -> 32 (pool)
      Block2: 32 -> 64 (pool)
      Block3: 64 -> 128 (pool)
      Block4: 128 -> 128 (pool)
      Block5: 128 -> 256 (no pool here)
    Final adaptive pooling to (8,8)
    """
    def __init__(self):
        super().__init__()
        self.block1 = ResidualBlock(3, 32, use_pool=True)
        self.block2 = ResidualBlock(32, 64, use_pool=True)
        self.block3 = ResidualBlock(64, 128, use_pool=True)
        self.block4 = ResidualBlock(128, 128, use_pool=True)
        self.block5 = ResidualBlock(128, 256, use_pool=False)
        self.final_pool = nn.AdaptiveAvgPool2d((8, 8))  # final featuremap 256 x 8 x 8

    def forward(self, x):
        x = self.block1(x)   # spatial reduced by /2
        x = self.block2(x)   # /4
        x = self.block3(x)   # /8
        x = self.block4(x)   # /16
        x = self.block5(x)   # remains /16
        x = self.final_pool(x)
        return x

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 128)
        self.tanh = nn.Tanh()
        self.score = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch, seq_len, hidden*2)
        e = self.tanh(self.attn(lstm_outputs))      # (batch, seq_len, 128)
        scores = self.score(e)                     # (batch, seq_len, 1)
        alpha = self.softmax(scores)               # (batch, seq_len, 1)
        context = torch.sum(alpha * lstm_outputs, dim=1)  # (batch, hidden*2)
        return context, alpha

class HybridCNNBiLSTMAttention(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.cnn = CNNFeatureExtractor()
        self.hidden_size = lstm_hidden
        self.num_layers = lstm_layers
        # After CNN final_pool -> (batch, 256, 8, 8)
        c_channels = 256
        c_w = 8
        self.lstm_input_size = c_channels * c_w  # 256*8 = 2048
        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
        )
        self.attention = AdditiveAttention(self.hidden_size)
        # FC layers after attention context vector (hidden*2)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x -> CNN
        x = self.cnn(x)                        # (batch, 256, 8, 8)
        b, c, h, w = x.size()
        # Flatten along height to create sequence length = h
        # sequence_length = h (8), input_size = c * w (256*8)
        x = x.view(b, h, c * w)               # (batch, seq_len=h, input_size)
        lstm_out, _ = self.bilstm(x)          # (batch, seq_len, hidden*2)
        context, attn_weights = self.attention(lstm_out)  # context: (batch, hidden*2)
        out = self.fc(context)                # (batch, num_classes)
        return out, attn_weights

# ------------------------------
# Utility: Model summary (layer-by-layer)
# We'll register forward hooks to capture output shapes and compute params.
# ------------------------------
def summarize_model(model, input_size=(3, INPUT_SIZE[0], INPUT_SIZE[1]), device=DEVICE):
    model.to(device)
    summary = []
    hooks = []

    def register_hook(module):
        def hook(module, inp, out):
            class_name = module.__class__.__name__
            module_idx = len(summary)
            # try to get param count
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            # get output shape (handle tuple outputs)
            try:
                if isinstance(out, (list, tuple)):
                    out_shape = [tuple(o.size()) for o in out]
                else:
                    out_shape = tuple(out.size())
            except Exception:
                out_shape = str(type(out))
            summary.append({
                "idx": module_idx,
                "name": f"{class_name}",
                "type": class_name,
                "output_shape": out_shape,
                "params": params,
                "trainable": trainable
            })
        # only register for modules that are of interest
        if not isinstance(module, nn.ModuleList) and \
           not isinstance(module, nn.Sequential) and \
           module != model:
            hooks.append(module.register_forward_hook(hook))

    # register hooks recursively
    model.apply(register_hook)

    # do a forward pass with dummy input to activate hooks
    input_tensor = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        # Some models return tuple (out, attn). We only need shapes, so wrap accordingly.
        try:
            out = model(input_tensor)
        except TypeError:
            out = model.forward(input_tensor)

    # remove hooks
    for h in hooks:
        h.remove()

    # Build DataFrame
    df = pd.DataFrame(summary)
    # reorder columns
    if not df.empty:
        df = df[["idx", "name", "type", "output_shape", "params", "trainable"]]
        df = df.sort_values("idx").reset_index(drop=True)

    # totals
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params

    return df, total_params, trainable_params, non_trainable

# ------------------------------
# Training loop with saving history and plotting
# ------------------------------
def original_id(path):
    """Map x.png, x_aug1.png, ... to the same original image ID."""
    return re.sub(r"_aug\d+$", "", os.path.splitext(os.path.basename(path))[0])


def build_augmented_file_datasets(original_train_dir, augmented_dir):
    class_names = ["Glioma", "Meningioma", "Healthy", "Pituitary Macroadenoma"]
    dataset = datasets.ImageFolder(augmented_dir)
    assert set(dataset.classes) == set(class_names)
    samples = [(path, class_names.index(dataset.classes[label])) for path, label in dataset.samples]
    assert len(samples) == 4545
    indices = np.arange(len(samples))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.20, random_state=SEED, stratify=[label for _, label in samples])
    training_samples = [samples[i] for i in train_indices]
    validation_samples = [samples[i] for i in val_indices]
    train_groups = {(label, original_id(path)) for path, label in training_samples}
    val_groups = {(label, original_id(path)) for path, label in validation_samples}
    print(f"Protocol: stratified 80/20 FILE split AFTER augmentation; seed={SEED}", flush=True)
    print(f"Class indices: {dict(enumerate(class_names))}", flush=True)
    print(f"Training files: {len(training_samples)}; validation files: {len(validation_samples)}; test images remain reserved.", flush=True)
    print(f"Original-image groups shared between training and validation: {len(train_groups & val_groups)}. Validation is not independent at original-image level.", flush=True)
    rows = [{"path": path, "class": class_names[label], "split": split}
            for split, part in [("train", training_samples), ("validation", validation_samples)]
            for path, label in part]
    pd.DataFrame(rows).to_csv(os.path.join(SAVE_DIR, "train_validation_split.csv"), index=False)
    return ImagePathDataset(training_samples, train_transform), ImagePathDataset(validation_samples, test_transform), class_names




def train_model(original_train_dir, augmented_dir,
                num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                patience=PATIENCE,
                min_epochs=MIN_EPOCHS_BEFORE_EARLY_STOPPING,
                save_path=MODEL_SAVE_PATH):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_dataset, val_dataset, class_names = build_augmented_file_datasets(
        original_train_dir, augmented_dir
    )
    loader_options = dict(batch_size=batch_size, num_workers=NUM_WORKERS,
                          pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = HybridCNNBiLSTMAttention(num_classes=len(class_names)).to(DEVICE)
    model.eval()  # Summary must not update BatchNorm running statistics.
    df_summary, total_params, trainable_params, non_trainable = summarize_model(model)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(df_summary.to_string(index=False))
        f.write(f"\n\nTotal params: {total_params}\nTrainable params: {trainable_params}\nNon-trainable params: {non_trainable}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history, best_val_loss, best_epoch, epochs_without_improvement = [], float("inf"), 0, 0

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else chr(67)+chr(80)+chr(85)}", flush=True)
    print(f"Device: {DEVICE}; batch size: {batch_size}; maximum epochs: {num_epochs}; "
          f"minimum epochs before early stopping: {min_epochs}; patience: {patience}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = train_correct = train_total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} Train", unit="batch", disable=True):
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum = val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} Validation", unit="batch", disable=True):
                images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs, _ = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss, train_acc = train_loss_sum / train_total, train_correct / train_total
        val_loss, val_acc = val_loss_sum / val_total, val_correct / val_total
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "val_loss": val_loss, "val_accuracy": val_acc})
        pd.DataFrame(history).to_csv(HISTORY_CSV, index=False)
        print(f"Epoch {epoch}: train loss={train_loss:.4f}, acc={train_acc:.4f}; val loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            if epoch > min_epochs:
                epochs_without_improvement = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "class_names": class_names,
                        "val_loss": val_loss, "val_accuracy": val_acc, "seed": SEED}, save_path)
            print("  Best validation loss improved; checkpoint saved.")
        else:
            if epoch <= min_epochs:
                print(f"  No improvement, but early stopping is disabled through epoch {min_epochs}.")
            else:
                epochs_without_improvement += 1
                print(f"  No validation-loss improvement after minimum epoch: "
                      f"{epochs_without_improvement}/{patience}")
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    df_hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df_hist.epoch, df_hist.train_loss, label="Train")
    axes[0].plot(df_hist.epoch, df_hist.val_loss, label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss"); axes[0].legend()
    axes[1].plot(df_hist.epoch, 100 * df_hist.train_accuracy, label="Train")
    axes[1].plot(df_hist.epoch, 100 * df_hist.val_accuracy, label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=300); plt.close(fig)
    print(f"Training complete. Best epoch: {best_epoch}; best validation loss: {best_val_loss:.6f}")
    print("Best checkpoint and training outputs saved to:", SAVE_DIR)
    return model, df_hist, (total_params, trainable_params, non_trainable)

# ------------------------------
# If called as script, run training with given paths.
# Replace these with your actual directories.
# ------------------------------
if __name__ == "__main__":
    original_train_dir = str(ROOT / "preprocessed_train_C")
    augmented_dir = str(ROOT / "preprocessed_trainn_augmented_C")
    if os.path.exists(MODEL_SAVE_PATH):
        raise FileExistsError("Training checkpoint already exists; refusing to overwrite it.")
    start_time = time.time()
    model, history_df, params_info = train_model(
        original_train_dir, augmented_dir,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
        patience=PATIENCE, min_epochs=MIN_EPOCHS_BEFORE_EARLY_STOPPING,
        save_path=MODEL_SAVE_PATH)
    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed/60:.2f} minutes")
