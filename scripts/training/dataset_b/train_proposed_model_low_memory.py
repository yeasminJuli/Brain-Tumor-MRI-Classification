# ============================================================
# FINAL HYBRID CNN + BiLSTM + ATTENTION MODEL
# 4 GB GPU CROSS-DATASET TESTING VERSION
#
# IMPORTANT:
# - Architecture unchanged
# - Input size unchanged: 256x256
# - Preprocessing unchanged
# - Learning rate unchanged
# - Batch size reduced from 70 -> 16
# - AMP added for lower GPU memory usage
# - Early stopping cannot happen before epoch 40
# ============================================================

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (256, 256)
NUM_CLASSES = 4

# Changed only for 4 GB GPU
BATCH_SIZE = 16

LR = 1e-4

# Maximum number of epochs
NUM_EPOCHS = 80

# Early stopping settings
# Training will NEVER stop before epoch 40.
MIN_EPOCHS = 40
EARLY_STOP_PATIENCE = 10

SAVE_DIR = "./training_outputs"

MODEL_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "hybrid_cnn_bilstm_attention_final.pth"
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

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# MIXED PRECISION
# ============================================================

# AMP reduces GPU memory usage when CUDA is available.
# It does NOT change the model architecture.
USE_AMP = torch.cuda.is_available()


# ============================================================
# TRANSFORMS
# ============================================================

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
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        pool=False
    ):
        super().__init__()

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
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        ) if pool else nn.Identity()


    def forward(self, x):

        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        out = self.pool(out)

        return out


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = ResidualBlock(
            3,
            32,
            pool=True
        )

        self.block2 = ResidualBlock(
            32,
            64,
            pool=True
        )

        self.block3 = ResidualBlock(
            64,
            128,
            pool=True
        )

        self.block4 = ResidualBlock(
            128,
            128,
            pool=True
        )

        self.block5 = ResidualBlock(
            128,
            256,
            pool=False
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (8, 8)
        )


    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.adaptive_pool(x)

        return x


# ============================================================
# ADDITIVE ATTENTION
# ============================================================

class AdditiveAttention(nn.Module):

    def __init__(
        self,
        input_dim,
        attention_dim=128
    ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(
                input_dim,
                attention_dim
            ),
            nn.Tanh(),
            nn.Linear(
                attention_dim,
                1
            )
        )


    def forward(self, x):

        # x:
        # batch x sequence x features

        scores = self.attention(x)

        # batch x sequence
        scores = scores.squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        )

        # batch x sequence x 1
        weights = weights.unsqueeze(-1)

        context = torch.sum(
            x * weights,
            dim=1
        )

        return context, weights


# ============================================================
# HYBRID CNN + BiLSTM + ATTENTION
# ============================================================

class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        num_classes=4,
        hidden_dim=128,
        num_layers=1
    ):
        super().__init__()

        # CNN
        self.cnn = CNNFeatureExtractor()

        # CNN output:
        # batch x 256 x 8 x 8

        # Each row becomes one sequence element.
        # Feature dimension = 256 * 8 = 2048
        self.lstm_input_size = 256 * 8

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # BiLSTM output:
        # hidden_dim * 2 = 256

        self.attention = AdditiveAttention(
            input_dim=hidden_dim * 2,
            attention_dim=128
        )

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_dim * 2,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

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
        # batch x 256 x 8 x 8

        batch_size = x.size(0)

        # Rearrange into sequence
        #
        # batch x 8 x (256*8)
        x = x.permute(
            0,
            3,
            1,
            2
        )

        x = x.reshape(
            batch_size,
            8,
            256 * 8
        )

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        context, attention_weights = self.attention(
            lstm_out
        )

        # Classification
        output = self.fc(context)

        return output, attention_weights


# ============================================================
# MODEL SUMMARY
# ============================================================

def model_summary(model):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    non_trainable_params = (
        total_params - trainable_params
    )

    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    print(
        f"Total params: {total_params:,}"
    )

    print(
        f"Trainable params: {trainable_params:,}"
    )

    print(
        f"Non-trainable params: {non_trainable_params:,}"
    )

    print("=" * 60)

    summary_data = pd.DataFrame({
        "Metric": [
            "Total Parameters",
            "Trainable Parameters",
            "Non-trainable Parameters"
        ],
        "Value": [
            total_params,
            trainable_params,
            non_trainable_params
        ]
    })

    summary_data.to_csv(
        SUMMARY_CSV,
        index=False
    )

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("MODEL SUMMARY\n")
        f.write("=" * 60 + "\n")

        f.write(
            f"Total params: {total_params:,}\n"
        )

        f.write(
            f"Trainable params: {trainable_params:,}\n"
        )

        f.write(
            f"Non-trainable params: "
            f"{non_trainable_params:,}\n"
        )

    return total_params


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    train_dir,
    test_dir,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    save_path=MODEL_SAVE_PATH
):

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Maximum epochs: {num_epochs}"
    )

    print(
        f"Minimum epochs: {MIN_EPOCHS}"
    )

    print(
        f"Early stopping patience: "
        f"{EARLY_STOP_PATIENCE}"
    )

    print(
        f"AMP enabled: {USE_AMP}"
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = datasets.ImageFolder(
        train_dir,
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=test_transform
    )

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation/Test images: "
        f"{len(test_dataset)}"
    )

    print(
        "\nClasses:",
        train_dataset.classes
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

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


    # ========================================================
    # MODEL
    # ========================================================

    model = HybridCNNBiLSTMAttention(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    model_summary(model)


    # ========================================================
    # LOSS + OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )


    # ========================================================
    # AMP SCALER
    # ========================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = []

    best_val_acc = 0.0
    best_epoch = -1

    epochs_without_improvement = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        num_epochs + 1
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )


            # ------------------------------------------------
            # MIXED PRECISION FORWARD
            # ------------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=USE_AMP
            ):

                outputs, _ = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )


            # ------------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            running_loss += (
                loss.item()
                * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        train_loss = (
            running_loss / total
        )

        train_acc = (
            100.0 * correct / total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=USE_AMP
                ):

                    outputs, _ = model(
                        images
                    )

                    loss = criterion(
                        outputs,
                        labels
                    )


                val_running_loss += (
                    loss.item()
                    * images.size(0)
                )

                _, predicted = torch.max(
                    outputs,
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted == labels
                ).sum().item()


        val_loss = (
            val_running_loss /
            val_total
        )

        val_acc = (
            100.0 *
            val_correct /
            val_total
        )


        # ====================================================
        # EPOCH TIME
        # ====================================================

        epoch_time = (
            time.time() - epoch_start
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history.append({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "Epoch Time": epoch_time
        })

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
            HISTORY_CSV,
            index=False
        )


        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "val_accuracy":
                val_acc,
            "train_accuracy":
                train_acc
        }

        torch.save(
            checkpoint,
            save_path
        )


        # ====================================================
        # BEST VALIDATION ACCURACY
        # ====================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Epoch [{epoch}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {val_acc:.2f}%  "
            f"Time: {epoch_time:.1f}s"
        )


        # ====================================================
        # EARLY STOPPING
        #
        # IMPORTANT:
        # This condition can NEVER trigger before epoch 40.
        # ====================================================

        if (
            epoch >= MIN_EPOCHS
            and
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}: no validation-accuracy "
                f"improvement for "
                f"{EARLY_STOP_PATIENCE} consecutive "
                f"epochs."
            )

            break


    # ========================================================
    # TRAINING FINISHED
    # ========================================================

    print("\n" + "=" * 60)

    print(
        f"Training finished."
    )

    print(
        f"Best val acc: "
        f"{best_val_acc:.2f}% "
        f"at epoch {best_epoch}"
    )

    print("=" * 60)


    # ========================================================
    # PLOTS
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(
        1,
        2,
        1
    )

    plt.plot(
        history_df["Epoch"],
        history_df["Train Loss"],
        label="Train Loss"
    )

    plt.plot(
        history_df["Epoch"],
        history_df["Val Loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()


    plt.subplot(
        1,
        2,
        2
    )

    plt.plot(
        history_df["Epoch"],
        history_df["Train Accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history_df["Epoch"],
        history_df["Val Accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "training_curves.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_dir = (
        r"C:\user\juli\thesis\Dataset"
        r"\augmented_train"
    )

    test_dir = (
        r"C:\user\juli\thesis\Dataset"
        r"\preprocessed_test"
    )

    train_model(
        train_dir=train_dir,
        test_dir=test_dir,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        save_path=MODEL_SAVE_PATH
    )