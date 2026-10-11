# final_hybrid_model_with_summary_early_stopping.py
# PyTorch implementation:
# CNN (with residual blocks) -> BiLSTM -> Additive Attention -> FC
# Saves per-epoch metrics to CSV, plots curves,
# saves layer-by-layer parameter summary,
# uses early stopping after a minimum number of epochs,
# and saves the best validation model.

import os
import csv
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Config / Hyperparameters
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (256, 256)
NUM_CLASSES = 4

# Keep your batch size here.
# If you already changed it for your 4 GB GPU, keep that value.
BATCH_SIZE = 16

LR = 1e-4

# Maximum number of epochs
NUM_EPOCHS = 80

# ------------------------------------------------------------
# Early Stopping
# ------------------------------------------------------------

# Training will NOT stop before this epoch.
MIN_EPOCHS = 30

# After MIN_EPOCHS, if validation accuracy does not improve
# for PATIENCE consecutive epochs, training will stop.
PATIENCE = 8


# ============================================================
# Output paths
# ============================================================

SAVE_DIR = "./training_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Final/best model
MODEL_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "hybrid_cnn_bilstm_attention_best.pth"
)

# Optional checkpoint for the most recently completed epoch
LAST_CHECKPOINT_PATH = os.path.join(
    SAVE_DIR,
    "hybrid_cnn_bilstm_attention_last.pth"
)

HISTORY_CSV = os.path.join(
    SAVE_DIR,
    "training_history.csv"
)

SUMMARY_CSV = os.path.join(
    SAVE_DIR,
    "model_summary.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "model_summary.txt"
)

CURVES_PATH = os.path.join(
    SAVE_DIR,
    "training_curves.png"
)


# ============================================================
# Transforms
# ============================================================

# Your preprocessing produces grayscale images.
# We replicate grayscale into 3 channels because the model
# expects 3-channel input.

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# Model components
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_pool=False):
        super().__init__()

        self.use_pool = use_pool

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

            self.skip_bn = nn.BatchNorm2d(out_channels)

        else:
            self.skip_conv = None

        self.pool = (
            nn.MaxPool2d(kernel_size=2, stride=2)
            if use_pool
            else nn.Identity()
        )

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


# ============================================================
# CNN Feature Extractor
# ============================================================

class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = ResidualBlock(
            3, 32, use_pool=True
        )

        self.block2 = ResidualBlock(
            32, 64, use_pool=True
        )

        self.block3 = ResidualBlock(
            64, 128, use_pool=True
        )

        self.block4 = ResidualBlock(
            128, 128, use_pool=True
        )

        self.block5 = ResidualBlock(
            128, 256, use_pool=False
        )

        self.final_pool = nn.AdaptiveAvgPool2d(
            (8, 8)
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.final_pool(x)

        return x


# ============================================================
# Additive Attention
# ============================================================

class AdditiveAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(
            hidden_dim * 2,
            128
        )

        self.tanh = nn.Tanh()

        self.score = nn.Linear(
            128,
            1
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs):

        # lstm_outputs:
        # (batch, seq_len, hidden*2)

        e = self.tanh(
            self.attn(lstm_outputs)
        )

        scores = self.score(e)

        alpha = self.softmax(scores)

        context = torch.sum(
            alpha * lstm_outputs,
            dim=1
        )

        return context, alpha


# ============================================================
# Hybrid CNN + BiLSTM + Attention
# ============================================================

class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        lstm_hidden=128,
        lstm_layers=1
    ):

        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.hidden_size = lstm_hidden
        self.num_layers = lstm_layers

        # CNN output:
        # (batch, 256, 8, 8)

        c_channels = 256
        c_w = 8

        self.lstm_input_size = (
            c_channels * c_w
        )

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(
                0.3
                if self.num_layers > 1
                else 0.0
            )
        )

        self.attention = AdditiveAttention(
            self.hidden_size
        )

        self.fc = nn.Sequential(

            nn.Linear(
                self.hidden_size * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # x:
        # (batch, 256, 8, 8)

        b, c, h, w = x.size()

        # Convert feature map into sequence
        #
        # (batch, 256, 8, 8)
        # ->
        # (batch, 8, 2048)

        x = x.view(
            b,
            h,
            c * w
        )

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Attention
        context, attn_weights = self.attention(
            lstm_out
        )

        # Fully connected classifier
        out = self.fc(context)

        return out, attn_weights


# ============================================================
# Model Summary
# ============================================================

def summarize_model(
    model,
    input_size=(3, INPUT_SIZE[0], INPUT_SIZE[1]),
    device=DEVICE
):

    model.to(device)

    summary = []
    hooks = []

    def register_hook(module):

        def hook(module, inp, out):

            class_name = module.__class__.__name__

            module_idx = len(summary)

            params = sum(
                p.numel()
                for p in module.parameters()
            )

            trainable = sum(
                p.numel()
                for p in module.parameters()
                if p.requires_grad
            )

            try:

                if isinstance(out, (list, tuple)):

                    out_shape = [
                        tuple(o.size())
                        for o in out
                    ]

                else:

                    out_shape = tuple(
                        out.size()
                    )

            except Exception:

                out_shape = str(type(out))

            summary.append({

                "idx": module_idx,

                "name": class_name,

                "type": class_name,

                "output_shape": out_shape,

                "params": params,

                "trainable": trainable

            })

        if (
            not isinstance(module, nn.ModuleList)
            and
            not isinstance(module, nn.Sequential)
            and
            module != model
        ):

            hooks.append(
                module.register_forward_hook(hook)
            )

    model.apply(register_hook)

    input_tensor = torch.randn(
        1,
        *input_size
    ).to(device)

    with torch.no_grad():

        try:

            model(input_tensor)

        except TypeError:

            model.forward(input_tensor)

    for h in hooks:
        h.remove()

    df = pd.DataFrame(summary)

    if not df.empty:

        df = df[
            [
                "idx",
                "name",
                "type",
                "output_shape",
                "params",
                "trainable"
            ]
        ]

        df = df.sort_values(
            "idx"
        ).reset_index(drop=True)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable = (
        total_params - trainable_params
    )

    return (
        df,
        total_params,
        trainable_params,
        non_trainable
    )


# ============================================================
# Training
# ============================================================

def train_model(
    train_dir,
    test_dir,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    save_path=MODEL_SAVE_PATH
):

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("\n========================================")
    print("Dataset Information")
    print("========================================")

    print(
        f"Training images: {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(test_dataset)}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Maximum epochs: {num_epochs}"
    )

    print(
        f"Minimum epochs before early stopping: {MIN_EPOCHS}"
    )

    print(
        f"Early stopping patience: {PATIENCE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print("========================================\n")


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HybridCNNBiLSTMAttention(
        num_classes=NUM_CLASSES
    ).to(DEVICE)


    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    (
        df_summary,
        total_params,
        trainable_params,
        non_trainable
    ) = summarize_model(model)

    if not df_summary.empty:

        df_summary.to_csv(
            SUMMARY_CSV,
            index=False
        )

    with open(
        SUMMARY_TXT,
        "w"
    ) as f:

        f.write(
            "Layer-by-layer summary:\n"
        )

        if not df_summary.empty:

            f.write(
                df_summary.to_string(
                    index=False
                )
            )

            f.write("\n\n")

        f.write(
            f"Total params: {total_params}\n"
        )

        f.write(
            f"Trainable params: {trainable_params}\n"
        )

        f.write(
            f"Non-trainable params: {non_trainable}\n"
        )


    print(
        f"Total params: {total_params}"
    )

    print(
        f"Trainable params: {trainable_params}"
    )

    print(
        f"Non-trainable params: {non_trainable}"
    )

    print(
        f"Model summary saved to: "
        f"{SUMMARY_CSV} and {SUMMARY_TXT}"
    )


    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )


    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_val_acc = 0.0

    best_epoch = -1

    epochs_without_improvement = 0


    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(
        1,
        num_epochs + 1
    ):

        # ====================================================
        # Training
        # ====================================================

        model.train()

        running_loss = 0.0

        running_correct = 0

        running_total = 0

        train_iterator = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch}/{num_epochs} - Train"
            ),
            leave=False
        )

        for images, labels in train_iterator:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs, _ = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            _, preds = torch.max(
                outputs,
                1
            )

            running_correct += (
                (preds == labels)
                .sum()
                .item()
            )

            running_total += (
                labels.size(0)
            )


        train_loss = (
            running_loss
            / running_total
        )

        train_acc = (
            100.0
            * running_correct
            / running_total
        )


        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_running_loss = 0.0

        val_running_correct = 0

        val_running_total = 0

        with torch.no_grad():

            val_iterator = tqdm(
                test_loader,
                desc=(
                    f"Epoch {epoch}/{num_epochs} - Val"
                ),
                leave=False
            )

            for images, labels in val_iterator:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )

                outputs, _ = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_running_loss += (
                    loss.item()
                    * images.size(0)
                )

                _, preds = torch.max(
                    outputs,
                    1
                )

                val_running_correct += (
                    (preds == labels)
                    .sum()
                    .item()
                )

                val_running_total += (
                    labels.size(0)
                )


        val_loss = (
            val_running_loss
            / val_running_total
        )

        val_acc = (
            100.0
            * val_running_correct
            / val_running_total
        )


        # ====================================================
        # Save epoch history
        # ====================================================

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc

        })


        # Save CSV after every epoch
        df_hist = pd.DataFrame(history)

        df_hist.to_csv(
            HISTORY_CSV,
            index=False
        )


        # ====================================================
        # Check whether this is the best model
        # ====================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = epoch

            epochs_without_improvement = 0

            # ------------------------------------------------
            # SAVE BEST MODEL
            # ------------------------------------------------

            torch.save({

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "train_acc":
                    train_acc,

                "val_acc":
                    val_acc,

                "best_val_acc":
                    best_val_acc

            }, save_path)

            best_message = (
                f"  ✓ NEW BEST MODEL SAVED "
                f"(Val Acc: {val_acc:.2f}%)"
            )

        else:

            epochs_without_improvement += 1

            best_message = (
                f"  No improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )


        # ====================================================
        # Save latest checkpoint
        # ====================================================

        torch.save({

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "train_acc":
                train_acc,

            "val_acc":
                val_acc

        }, LAST_CHECKPOINT_PATH)


        # ====================================================
        # Print epoch summary
        # ====================================================

        print(
            f"Epoch [{epoch}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.2f}%"
        )

        print(best_message)


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        # IMPORTANT:
        # Early stopping is NOT allowed before MIN_EPOCHS.

        if epoch >= MIN_EPOCHS:

            if epochs_without_improvement >= PATIENCE:

                print("\n" + "=" * 60)

                print(
                    "EARLY STOPPING TRIGGERED"
                )

                print(
                    f"Training stopped at epoch: {epoch}"
                )

                print(
                    f"Best validation accuracy: "
                    f"{best_val_acc:.2f}%"
                )

                print(
                    f"Best epoch: {best_epoch}"
                )

                print("=" * 60 + "\n")

                break


    # ========================================================
    # Training finished
    # ========================================================

    print("\n========================================")
    print("Training finished")
    print("========================================")

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Total epochs completed: "
        f"{len(history)}"
    )

    print(
        f"History CSV: {HISTORY_CSV}"
    )

    print(
        f"Best model: {save_path}"
    )

    print(
        f"Last checkpoint: "
        f"{LAST_CHECKPOINT_PATH}"
    )

    print("========================================\n")


    # ========================================================
    # Plot training history
    # ========================================================

    df_hist = pd.DataFrame(history)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        df_hist["epoch"],
        df_hist["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        df_hist["epoch"],
        df_hist["val_loss"],
        label="Val Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.title("Loss Curve")


    plt.subplot(1, 2, 2)

    plt.plot(
        df_hist["epoch"],
        df_hist["train_acc"],
        label="Train Acc"
    )

    plt.plot(
        df_hist["epoch"],
        df_hist["val_acc"],
        label="Val Acc"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.legend()

    plt.title("Accuracy Curve")

    plt.tight_layout()

    plt.savefig(
        CURVES_PATH,
        dpi=300
    )

    plt.show()


    return (
        model,
        df_hist,
        (
            total_params,
            trainable_params,
            non_trainable
        )
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # Replace with your real directories

    train_dir = (
        r"C:\user\juli\thesis\Dataset"
        r"\preprocessed_train"
    )

    test_dir = (
        r"C:\user\juli\thesis\Dataset"
        r"\preprocessed_test"
    )


    start_time = time.time()


    model, history_df, params_info = train_model(

        train_dir,

        test_dir,

        num_epochs=NUM_EPOCHS,

        batch_size=BATCH_SIZE,

        lr=LR,

        save_path=MODEL_SAVE_PATH
    )


    elapsed = (
        time.time() - start_time
    )


    print(
        f"Total training time: "
        f"{elapsed / 60:.2f} minutes"
    )