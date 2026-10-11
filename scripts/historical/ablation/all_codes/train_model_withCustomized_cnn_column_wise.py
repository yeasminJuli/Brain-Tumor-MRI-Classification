# ============================================================
# TWO-CONDITION SPATIAL ORDERING ABLATION
# Conditions:
#   1. Column-wise Ordering
#   2. Alternative Ordering (Boustrophedon / Zigzag)
#
# Original Row-wise Ordering is NOT retrained here.
# The original model result is already available separately.
#
# CNN:
#   Input: 3 x 256 x 256
#   Residual Blocks
#   Final feature map: 256 x 8 x 8
#
# Sequence representation:
#   Column-wise:
#       sequence length = 8
#       input size = 8 x 256 = 2048
#
#   Alternative Ordering:
#       sequence length = 8
#       input size = 8 x 256 = 2048
#
# Test dataset is NOT used for training/validation.
# ============================================================


import os
import time
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (256, 256)
NUM_CLASSES = 4

BATCH_SIZE = 16
LEARNING_RATE = 1e-4

MAX_EPOCHS = 80
MIN_EPOCHS = 40
PATIENCE = 5

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42

# Windows + limited VRAM stability
NUM_WORKERS = 0
PIN_MEMORY = True


# ============================================================
# 2. DATASET PATHS
# ============================================================

TRAIN_DIR = r"C:\user\juli\thesis\original_file\othr folders\PREPROCESSED_TRAINN_AUGMENTED"

TEST_DIR = r"C:\user\juli\thesis\original_file\othr folders\preprocessed_testt"


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = r"C:\user\juli\thesis\original_file\othr folders\training_outputs_proposed_column_zigzag"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. REPRODUCIBILITY
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 5. TRANSFORMS
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
# 6. RESIDUAL BLOCK
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

        # Spatial downsampling
        if use_pool:

            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

        else:

            self.pool = nn.Identity()


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
# 7. CNN FEATURE EXTRACTOR
# ============================================================

class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        # Block 1: 3 -> 32, spatial /2
        self.block1 = ResidualBlock(
            3,
            32,
            use_pool=True
        )

        # Block 2: 32 -> 64, spatial /2
        self.block2 = ResidualBlock(
            32,
            64,
            use_pool=True
        )

        # Block 3: 64 -> 128, spatial /2
        self.block3 = ResidualBlock(
            64,
            128,
            use_pool=True
        )

        # Block 4: 128 -> 128, spatial /2
        self.block4 = ResidualBlock(
            128,
            128,
            use_pool=True
        )

        # Block 5: 128 -> 256, no pooling
        self.block5 = ResidualBlock(
            128,
            256,
            use_pool=False
        )

        # Final feature map:
        # 256 x 8 x 8
        self.final_pool = nn.AdaptiveAvgPool2d(
            (8, 8)
        )


    def forward(self, x):

        x = self.block1(x)       # /2
        x = self.block2(x)       # /4
        x = self.block3(x)       # /8
        x = self.block4(x)       # /16
        x = self.block5(x)       # remains /16

        x = self.final_pool(x)   # 256 x 8 x 8

        return x


# ============================================================
# 8. ADDITIVE ATTENTION
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

        self.softmax = nn.Softmax(
            dim=1
        )


    def forward(self, lstm_outputs):

        # lstm_outputs:
        # (batch, sequence_length, hidden_dim*2)

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
# 9. HYBRID CNN + BiLSTM + ATTENTION MODEL
# ============================================================

class HybridCNNBiLSTMAttention(nn.Module):

    def __init__(
        self,
        ordering,
        num_classes=NUM_CLASSES,
        lstm_hidden=128,
        lstm_layers=1
    ):

        super().__init__()

        self.ordering = ordering

        self.cnn = CNNFeatureExtractor()

        self.hidden_size = lstm_hidden
        self.num_layers = lstm_layers

        # CNN output:
        # (batch, 256, 8, 8)

        self.c_channels = 256
        self.c_height = 8
        self.c_width = 8

        # Both orderings create:
        # sequence length = 8
        # input size = 8 * 256 = 2048

        self.lstm_input_size = (
            self.c_channels * self.c_width
        )

        self.bilstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if self.num_layers > 1 else 0.0
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


    # --------------------------------------------------------
    # COLUMN-WISE ORDERING
    # --------------------------------------------------------

    def column_wise_ordering(self, x):

        # Input:
        # (B, C, H, W)
        #
        # C = 256
        # H = 8
        # W = 8
        #
        # Each spatial column becomes one sequence position.
        #
        # Output:
        # (B, W, C*H)
        # = (B, 8, 2048)

        b, c, h, w = x.size()

        x = x.permute(
            0,
            3,
            1,
            2
        ).contiguous()

        x = x.view(
            b,
            w,
            c * h
        )

        return x


    # --------------------------------------------------------
    # ALTERNATIVE ORDERING
    # BOustrophedon / ZIGZAG
    # --------------------------------------------------------

    def alternative_ordering(self, x):

        # Input:
        # (B, C, H, W)
        #
        # Each spatial row is traversed alternately:
        #
        # Row 1: left -> right
        # Row 2: right -> left
        # Row 3: left -> right
        # Row 4: right -> left
        # ...
        #
        # Output:
        # (B, H, C*W)
        # = (B, 8, 2048)

        b, c, h, w = x.size()

        sequence_rows = []

        for row_index in range(h):

            # Select one spatial row
            row = x[:, :, row_index, :]

            # row shape:
            # (B, C, W)

            # Reverse the spatial-column direction
            # on alternate rows.

            if row_index % 2 == 1:

                row = torch.flip(
                    row,
                    dims=[2]
                )

            # Convert:
            # (B, C, W)
            # ->
            # (B, C*W)

            row = row.contiguous().view(
                b,
                c * w
            )

            sequence_rows.append(row)

        # Stack rows along sequence dimension

        x = torch.stack(
            sequence_rows,
            dim=1
        )

        # Final:
        # (B, H, C*W)
        # = (B, 8, 2048)

        return x


    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(self, x):

        # CNN feature extraction
        x = self.cnn(x)

        # x:
        # (B, 256, 8, 8)

        if self.ordering == "column_wise":

            x = self.column_wise_ordering(x)

        elif self.ordering == "alternative":

            x = self.alternative_ordering(x)

        else:

            raise ValueError(
                f"Unknown ordering: {self.ordering}"
            )

        # x:
        # (B, 8, 2048)

        lstm_out, _ = self.bilstm(x)

        # lstm_out:
        # (B, 8, 256)

        context, attn_weights = self.attention(
            lstm_out
        )

        # context:
        # (B, 256)

        out = self.fc(context)

        return out, attn_weights


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

def summarize_model(
    model,
    input_size=(3, 256, 256),
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

                out_shape = str(
                    type(out)
                )

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
            and not isinstance(module, nn.Sequential)
            and module != model
        ):

            hooks.append(
                module.register_forward_hook(hook)
            )


    model.apply(register_hook)


    # Dummy forward pass

    input_tensor = torch.randn(
        1,
        *input_size
    ).to(device)


    model.eval()

    with torch.no_grad():

        model(input_tensor)


    # Remove hooks

    for hook in hooks:

        hook.remove()


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
        total_params -
        trainable_params
    )


    return (
        df,
        total_params,
        trainable_params,
        non_trainable
    )


# ============================================================
# 11. TRAINING FUNCTION
# ============================================================

def train_one_condition(
    ordering,
    train_dataset,
    val_dataset
):

    print()
    print("=" * 70)
    print(f"STARTING ABLATION: {ordering}")
    print("=" * 70)


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    condition_dir = os.path.join(
        OUTPUT_DIR,
        ordering
    )

    os.makedirs(
        condition_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(RANDOM_SEED)


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HybridCNNBiLSTMAttention(
        ordering=ordering,
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


    summary_csv = os.path.join(
        condition_dir,
        f"model_summary_{ordering}.csv"
    )

    summary_txt = os.path.join(
        condition_dir,
        f"model_summary_{ordering}.txt"
    )


    if not df_summary.empty:

        df_summary.to_csv(
            summary_csv,
            index=False
        )


    with open(
        summary_txt,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Ordering: {ordering}\n\n"
        )

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


    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = DEVICE.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []

    best_val_acc = 0.0

    best_epoch = -1

    best_state = None

    no_improvement_epochs = 0


    history_csv = os.path.join(
        condition_dir,
        f"training_history_{ordering}.csv"
    )


    best_model_path = os.path.join(
        condition_dir,
        f"best_{ordering}.pth"
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_loss = 0.0

        running_correct = 0

        running_total = 0


        train_iterator = tqdm(
            train_loader,
            desc=(
                f"{ordering} | "
                f"Epoch {epoch}/{MAX_EPOCHS}"
            ),
            leave=False
        )


        for images, labels in train_iterator:

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


            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                outputs, _ = model(images)

                loss = criterion(
                    outputs,
                    labels
                )


            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()


            running_loss += (
                loss.item() *
                images.size(0)
            )


            preds = torch.argmax(
                outputs,
                dim=1
            )


            running_correct += (
                preds == labels
            ).sum().item()


            running_total += (
                labels.size(0)
            )


        train_loss = (
            running_loss /
            running_total
        )

        train_acc = (
            100.0 *
            running_correct /
            running_total
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_running_loss = 0.0

        val_running_correct = 0

        val_running_total = 0


        with torch.no_grad():

            val_iterator = tqdm(
                val_loader,
                desc=(
                    f"{ordering} | "
                    f"Epoch {epoch}/{MAX_EPOCHS} "
                    f"- Validation"
                ),
                leave=False
            )


            for images, labels in val_iterator:

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
                    enabled=use_amp
                ):

                    outputs, _ = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )


                val_running_loss += (
                    loss.item() *
                    images.size(0)
                )


                preds = torch.argmax(
                    outputs,
                    dim=1
                )


                val_running_correct += (
                    preds == labels
                ).sum().item()


                val_running_total += (
                    labels.size(0)
                )


        val_loss = (
            val_running_loss /
            val_running_total
        )

        val_acc = (
            100.0 *
            val_running_correct /
            val_running_total
        )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_acc": train_acc,

            "val_loss": val_loss,

            "val_acc": val_acc
        })


        df_history = pd.DataFrame(
            history
        )


        df_history.to_csv(
            history_csv,
            index=False
        )


        # ====================================================
        # CHECK BEST MODEL
        # ====================================================

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
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

                    "ordering":
                        ordering
                },
                best_model_path
            )

            no_improvement_epochs = 0

            best_marker = " <-- BEST"

        else:

            no_improvement_epochs += 1

            best_marker = ""


        # ====================================================
        # PRINT EPOCH RESULT
        # ====================================================

        print(
            f"{ordering} | "
            f"Epoch {epoch}/{MAX_EPOCHS}:"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.2f}%"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{val_acc:.2f}%{best_marker}"
        )

        print(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.2f}%"
        )

        print(
            f"No improvement epochs: "
            f"{no_improvement_epochs}"
        )


        # ====================================================
        # EARLY STOPPING
        #
        # IMPORTANT:
        # Never stop before MIN_EPOCHS.
        # ====================================================

        if (
            epoch >= MIN_EPOCHS
            and
            no_improvement_epochs >= PATIENCE
        ):

            print()
            print(
                f"Early stopping triggered for "
                f"{ordering}."
            )

            print(
                f"Training stopped at epoch "
                f"{epoch}."
            )

            break


    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_epochs_run = len(
        history
    )


    print()
    print("-" * 70)

    print(
        f"FINISHED: {ordering}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.2f}%"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Total epochs run: "
        f"{total_epochs_run}"
    )

    print(
        f"Best model saved to:"
    )

    print(
        best_model_path
    )

    print("-" * 70)


    # ========================================================
    # SAVE FINAL SUMMARY
    # ========================================================

    summary_txt_path = os.path.join(
        condition_dir,
        f"result_summary_{ordering}.txt"
    )


    with open(
        summary_txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Spatial Ordering Ablation Result\n"
        )

        f.write(
            "================================\n\n"
        )

        f.write(
            f"Ordering: {ordering}\n"
        )

        f.write(
            f"Best validation accuracy: "
            f"{best_val_acc:.4f}%\n"
        )

        f.write(
            f"Best epoch: {best_epoch}\n"
        )

        f.write(
            f"Total epochs run: "
            f"{total_epochs_run}\n"
        )

        f.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        f.write(
            f"Maximum epochs: "
            f"{MAX_EPOCHS}\n"
        )

        f.write(
            f"Minimum epochs: "
            f"{MIN_EPOCHS}\n"
        )

        f.write(
            f"Early stopping patience: "
            f"{PATIENCE}\n"
        )

        f.write(
            "\nTest dataset was NOT used "
            "for training or validation.\n"
        )


    # ========================================================
    # PLOT CURVES
    # ========================================================

    df_history = pd.DataFrame(
        history
    )


    # Loss curve

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        df_history["epoch"],
        df_history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        df_history["epoch"],
        df_history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        f"{ordering} - Loss Curve"
    )

    plt.legend()

    plt.tight_layout()

    loss_plot_path = os.path.join(
        condition_dir,
        f"{ordering}_loss_curve.png"
    )

    plt.savefig(
        loss_plot_path,
        dpi=300
    )

    plt.close()


    # Accuracy curve

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        df_history["epoch"],
        df_history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        df_history["epoch"],
        df_history["val_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.title(
        f"{ordering} - Accuracy Curve"
    )

    plt.legend()

    plt.tight_layout()

    accuracy_plot_path = os.path.join(
        condition_dir,
        f"{ordering}_accuracy_curve.png"
    )

    plt.savefig(
        accuracy_plot_path,
        dpi=300
    )

    plt.close()


    return {
        "ordering": ordering,

        "best_val_acc":
            best_val_acc,

        "best_epoch":
            best_epoch,

        "epochs_run":
            total_epochs_run,

        "best_model_path":
            best_model_path
    }


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    start_time = time.time()

    print()
    print("=" * 70)
    print("TWO-CONDITION SPATIAL ORDERING ABLATION STUDY")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Maximum epochs: {MAX_EPOCHS}"
    )

    print(
        f"Minimum epochs: {MIN_EPOCHS}"
    )

    print(
        f"Early stopping patience: {PATIENCE}"
    )

    print(
        f"Validation ratio: "
        f"{VALIDATION_RATIO}"
    )

    print()
    print(
        "Conditions being trained:"
    )

    print(
        "1. Column-wise Ordering"
    )

    print(
        "2. Alternative Ordering "
        "(Boustrophedon / Zigzag)"
    )

    print()
    print(
        "Original Row-wise Ordering is NOT retrained."
    )

    print(
        "Its existing result will be used separately "
        "for the final three-condition comparison."
    )

    print()


    # ========================================================
    # SET SEED
    # ========================================================

    set_seed(
        RANDOM_SEED
    )


    # ========================================================
    # LOAD FULL TRAINING DATASET
    # ========================================================

    full_dataset = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )


    print(
        f"Total images in training folder: "
        f"{len(full_dataset)}"
    )


    print(
        "Class mapping:"
    )

    for class_name, class_index in (
        full_dataset.class_to_idx.items()
    ):

        print(
            f"{class_index} -> {class_name}"
        )


    # ========================================================
    # LOAD TEST DATASET ONLY FOR CLASS-MAPPING CHECK
    # ========================================================

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )


    print(
        f"Test images: "
        f"{len(test_dataset)}"
    )


    print(
        "Test class mapping:"
    )

    for class_name, class_index in (
        test_dataset.class_to_idx.items()
    ):

        print(
            f"{class_index} -> {class_name}"
        )


    if (
        full_dataset.class_to_idx
        !=
        test_dataset.class_to_idx
    ):

        raise ValueError(
            "ERROR: Training and test "
            "class mappings do not match."
        )


    print(
        "Test class mapping matches."
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    total_size = len(
        full_dataset
    )

    validation_size = int(
        total_size *
        VALIDATION_RATIO
    )

    training_size = (
        total_size -
        validation_size
    )


    split_generator = torch.Generator()

    split_generator.manual_seed(
        RANDOM_SEED
    )


    train_dataset, val_dataset = random_split(
        full_dataset,
        [
            training_size,
            validation_size
        ],
        generator=split_generator
    )


    print()
    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        "IMPORTANT: Test dataset is NOT used."
    )


    # ========================================================
    # TWO CONDITIONS ONLY
    # ========================================================

    ORDERINGS = [
        "column_wise",
        "alternative"
    ]


    results = []


    # ========================================================
    # RUN EACH CONDITION
    # ========================================================

    for ordering in ORDERINGS:

        result = train_one_condition(
            ordering=ordering,
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        results.append(
            result
        )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        results
    )


    comparison_csv = os.path.join(
        OUTPUT_DIR,
        "two_condition_comparison.csv"
    )


    comparison_df.to_csv(
        comparison_csv,
        index=False
    )


    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("TWO-CONDITION ABLATION RESULTS")
    print("=" * 70)


    for result in results:

        print(
            f"{result['ordering']}: "
            f"{result['best_val_acc']:.2f}% "
            f"(best epoch = "
            f"{result['best_epoch']})"
        )


    print()
    print(
        f"Comparison CSV saved to:"
    )

    print(
        comparison_csv
    )


    # ========================================================
    # REMINDER FOR FINAL THREE-CONDITION TABLE
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL REVIEWER COMPARISON")
    print("=" * 70)

    print(
        "Row-wise Ordering (Original): "
        "use your existing 99.42% result"
    )

    print(
        "Column-wise Ordering: "
        "use the result from this run"
    )

    print(
        "Alternative Ordering: "
        "use the result from this run"
    )


    elapsed = (
        time.time() -
        start_time
    )


    print()
    print(
        f"Total experiment time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        "ALL TWO CONDITIONS COMPLETED."
    )