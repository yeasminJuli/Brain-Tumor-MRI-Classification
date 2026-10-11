# ============================================================
# DATASET B TRAINING
# Hybrid CNN -> BiLSTM -> Additive Attention -> FC
#
# IMPORTANT:
# - Dataset B has already been preprocessed.
# - NO augmented folder is used.
# - preprocessed_train_B is split:
#       80% training
#       20% validation
# - preprocessed_test_B is NOT used during training.
# - Best model is selected using validation loss.
# ============================================================

import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (256, 256)

NUM_CLASSES = 4

BATCH_SIZE = 16

LR = 1e-4

NUM_EPOCHS = 60

PATIENCE = 7

MIN_EPOCHS_BEFORE_EARLY_STOPPING = 30

SEED = 42

NUM_WORKERS = 4


# ------------------------------------------------------------
# Dataset B preprocessed TRAINING folder
# ------------------------------------------------------------

DATASET_B_TRAIN_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B\preprocessed_train_B"
)


# ------------------------------------------------------------
# Dataset B preprocessed TEST folder
#
# IMPORTANT:
# This path is recorded only for clarity.
# It is NOT loaded or used anywhere during training.
# ------------------------------------------------------------

DATASET_B_TEST_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B\preprocessed_test_B"
)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

SAVE_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B_training_results"
)

os.makedirs(SAVE_DIR, exist_ok=True)


MODEL_SAVE_PATH = os.path.join(
    SAVE_DIR,
    "hybrid_cnn_bilstm_attention_best_B.pth"
)

HISTORY_CSV = os.path.join(
    SAVE_DIR,
    "training_history_B.csv"
)

SUMMARY_CSV = os.path.join(
    SAVE_DIR,
    "model_summary_B.csv"
)

SUMMARY_TXT = os.path.join(
    SAVE_DIR,
    "model_summary_B.txt"
)

SPLIT_CSV = os.path.join(
    SAVE_DIR,
    "train_validation_split_B.csv"
)

TRAINING_CURVES_PATH = os.path.join(
    SAVE_DIR,
    "training_curves_B.png"
)


# ============================================================
# 2. EXPECTED CLASS MAPPING
# ============================================================

# ImageFolder sorts class folder names alphabetically.
#
# REQUIRED:
# 0 = glioma
# 1 = meningioma
# 2 = notumor
# 3 = pituitary

EXPECTED_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

EXPECTED_CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Improve reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 4. TRANSFORMS
# ============================================================

# Dataset B preprocessing already generated grayscale images.
#
# The network expects 3 channels.
# Therefore grayscale is replicated to 3 channels.
#
# NO DATA AUGMENTATION IS APPLIED HERE.


train_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        INPUT_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


validation_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        INPUT_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# ============================================================
# 5. DATASET CLASS
# ============================================================

class ImagePathDataset(Dataset):

    """
    Dataset backed by explicit (image_path, label) samples.
    """

    def __init__(self, samples, transform):

        self.samples = samples
        self.transform = transform


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        path, label = self.samples[index]

        with Image.open(path) as image:

            image = image.copy()

        image = self.transform(image)

        return image, label


# ============================================================
# 6. BUILD DATASET B TRAIN / VALIDATION SPLIT
# ============================================================

def build_dataset_b_split(train_directory):

    print("\n" + "=" * 70)

    print("LOADING DATASET B")

    print("=" * 70)


    # --------------------------------------------------------
    # Load Dataset B training directory
    # --------------------------------------------------------

    full_dataset = datasets.ImageFolder(
        train_directory
    )


    # --------------------------------------------------------
    # Verify class mapping
    # --------------------------------------------------------

    print("\nDetected classes:")

    print(full_dataset.classes)


    print("\nDetected class mapping:")

    print(full_dataset.class_to_idx)


    if full_dataset.classes != EXPECTED_CLASSES:

        raise RuntimeError(

            "\nCLASS ORDER ERROR!\n\n"

            f"Expected classes:\n"
            f"{EXPECTED_CLASSES}\n\n"

            f"Found classes:\n"
            f"{full_dataset.classes}\n"

        )


    if full_dataset.class_to_idx != EXPECTED_CLASS_TO_IDX:

        raise RuntimeError(

            "\nCLASS MAPPING ERROR!\n\n"

            f"Expected mapping:\n"
            f"{EXPECTED_CLASS_TO_IDX}\n\n"

            f"Found mapping:\n"
            f"{full_dataset.class_to_idx}\n"

        )


    print("\nClass mapping verification: PASS")


    # --------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------

    total_images = len(full_dataset)

    print(f"\nTotal Dataset B training-folder images: {total_images:,}")


    # --------------------------------------------------------
    # Print class counts BEFORE splitting
    # --------------------------------------------------------

    labels = np.asarray(
        full_dataset.targets
    )

    print("\nOriginal Dataset B class distribution:")

    for class_index, class_name in enumerate(full_dataset.classes):

        count = np.sum(labels == class_index)

        print(
            f"{class_index} = "
            f"{class_name:<12} : "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Generate indices
    # --------------------------------------------------------

    indices = np.arange(
        total_images
    )


    # --------------------------------------------------------
    # STRATIFIED 80 / 20 split
    # --------------------------------------------------------

    train_indices, val_indices = train_test_split(

        indices,

        test_size=0.20,

        random_state=SEED,

        stratify=labels

    )


    # --------------------------------------------------------
    # Build explicit sample lists
    # --------------------------------------------------------

    training_samples = [

        full_dataset.samples[i]

        for i in train_indices

    ]


    validation_samples = [

        full_dataset.samples[i]

        for i in val_indices

    ]


    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_paths = {

        os.path.abspath(path)

        for path, _ in training_samples

    }


    val_paths = {

        os.path.abspath(path)

        for path, _ in validation_samples

    }


    overlap = train_paths.intersection(
        val_paths
    )


    if overlap:

        raise RuntimeError(

            f"DATA LEAKAGE DETECTED: "
            f"{len(overlap)} images occur in both "
            f"training and validation."

        )


    print("\nTrain-validation leakage check: PASS")


    # --------------------------------------------------------
    # Print split sizes
    # --------------------------------------------------------

    print(
        f"\nTraining images (80%): "
        f"{len(training_samples):,}"
    )

    print(
        f"Validation images (20%): "
        f"{len(validation_samples):,}"
    )


    # --------------------------------------------------------
    # Print training distribution
    # --------------------------------------------------------

    train_labels = np.array([

        label

        for _, label in training_samples

    ])


    print("\nTraining class distribution:")

    for class_index, class_name in enumerate(full_dataset.classes):

        count = np.sum(
            train_labels == class_index
        )

        print(
            f"{class_index} = "
            f"{class_name:<12} : "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Print validation distribution
    # --------------------------------------------------------

    val_labels = np.array([

        label

        for _, label in validation_samples

    ])


    print("\nValidation class distribution:")

    for class_index, class_name in enumerate(full_dataset.classes):

        count = np.sum(
            val_labels == class_index
        )

        print(
            f"{class_index} = "
            f"{class_name:<12} : "
            f"{count:,}"
        )


    # --------------------------------------------------------
    # Save exact split to CSV
    # --------------------------------------------------------

    split_rows = []


    for i in train_indices:

        path, label = full_dataset.samples[i]

        split_rows.append({

            "path": path,

            "class_index": label,

            "class_name":
                full_dataset.classes[label],

            "split": "train"

        })


    for i in val_indices:

        path, label = full_dataset.samples[i]

        split_rows.append({

            "path": path,

            "class_index": label,

            "class_name":
                full_dataset.classes[label],

            "split": "validation"

        })


    split_dataframe = pd.DataFrame(
        split_rows
    )


    split_dataframe.to_csv(
        SPLIT_CSV,
        index=False
    )


    print(
        "\nTrain-validation split saved to:"
    )

    print(SPLIT_CSV)


    # --------------------------------------------------------
    # Final datasets
    # --------------------------------------------------------

    train_dataset = ImagePathDataset(

        training_samples,

        train_transform

    )


    validation_dataset = ImagePathDataset(

        validation_samples,

        validation_transform

    )


    return (

        train_dataset,

        validation_dataset,

        full_dataset.classes,

        full_dataset.class_to_idx

    )


# ============================================================
# 7. RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        use_pool=False
    ):

        super().__init__()

        self.use_pool = use_pool


        self.conv1 = nn.Conv2d(

            in_channels,

            out_channels,

            kernel_size=3,

            padding=1,

            bias=False

        )


        self.bn1 = nn.BatchNorm2d(
            out_channels
        )


        self.relu = nn.ReLU(
            inplace=True
        )


        self.conv2 = nn.Conv2d(

            out_channels,

            out_channels,

            kernel_size=3,

            padding=1,

            bias=False

        )


        self.bn2 = nn.BatchNorm2d(
            out_channels
        )


        # ----------------------------------------------------
        # Residual skip connection
        # ----------------------------------------------------

        if in_channels != out_channels:

            self.skip_conv = nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=1,

                bias=False

            )

            self.skip_bn = nn.BatchNorm2d(
                out_channels
            )

        else:

            self.skip_conv = None


        # ----------------------------------------------------
        # Optional spatial downsampling
        # ----------------------------------------------------

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

            identity = self.skip_conv(
                identity
            )

            identity = self.skip_bn(
                identity
            )


        out += identity

        out = self.relu(out)

        out = self.pool(out)


        return out


# ============================================================
# 8. CNN FEATURE EXTRACTOR
# ============================================================

class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()


        self.block1 = ResidualBlock(
            3,
            32,
            use_pool=True
        )


        self.block2 = ResidualBlock(
            32,
            64,
            use_pool=True
        )


        self.block3 = ResidualBlock(
            64,
            128,
            use_pool=True
        )


        self.block4 = ResidualBlock(
            128,
            128,
            use_pool=True
        )


        self.block5 = ResidualBlock(
            128,
            256,
            use_pool=False
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
# 9. ADDITIVE ATTENTION
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
        # batch x sequence_length x hidden*2

        e = self.tanh(

            self.attn(
                lstm_outputs
            )

        )


        scores = self.score(e)


        alpha = self.softmax(
            scores
        )


        context = torch.sum(

            alpha * lstm_outputs,

            dim=1

        )


        return context, alpha


# ============================================================
# 10. HYBRID CNN-BILSTM-ATTENTION MODEL
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
        #
        # batch x 256 x 8 x 8
        #
        # Sequence length = 8
        # LSTM input size = 256 x 8 = 2048


        self.lstm_input_size = 256 * 8


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


        # Shape:
        # batch x channels x height x width

        b, c, h, w = x.size()


        # Convert CNN feature map into sequence

        x = x.view(

            b,

            h,

            c * w

        )


        # BiLSTM

        lstm_out, _ = self.bilstm(
            x
        )


        # Attention

        context, attn_weights = self.attention(
            lstm_out
        )


        # Classification

        out = self.fc(
            context
        )


        return out, attn_weights


# ============================================================
# 11. MODEL SUMMARY
# ============================================================

def summarize_model(

    model,

    input_size=(
        3,
        INPUT_SIZE[0],
        INPUT_SIZE[1]
    ),

    device=DEVICE

):

    model.to(device)

    summary = []

    hooks = []


    def register_hook(module):

        def hook(module, inp, out):

            class_name = (
                module.__class__.__name__
            )


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

                if isinstance(
                    out,
                    (list, tuple)
                ):

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

                "output_shape":
                    out_shape,

                "params":
                    params,

                "trainable":
                    trainable

            })


        if (
            not isinstance(
                module,
                nn.ModuleList
            )

            and not isinstance(
                module,
                nn.Sequential
            )

            and module != model
        ):

            hooks.append(

                module.register_forward_hook(
                    hook
                )

            )


    model.apply(
        register_hook
    )


    input_tensor = torch.randn(

        1,

        *input_size

    ).to(device)


    model.eval()


    with torch.no_grad():

        model(
            input_tensor
        )


    for h in hooks:

        h.remove()


    df = pd.DataFrame(
        summary
    )


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
        ).reset_index(
            drop=True
        )


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

        total_params
        -
        trainable_params

    )


    return (

        df,

        total_params,

        trainable_params,

        non_trainable

    )


# ============================================================
# 12. TRAINING FUNCTION
# ============================================================

def train_model(

    train_directory,

    num_epochs=NUM_EPOCHS,

    batch_size=BATCH_SIZE,

    lr=LR,

    patience=PATIENCE,

    min_epochs=MIN_EPOCHS_BEFORE_EARLY_STOPPING,

    save_path=MODEL_SAVE_PATH

):

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    (
        train_dataset,

        val_dataset,

        class_names,

        class_to_idx

    ) = build_dataset_b_split(
        train_directory
    )


    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    loader_options = dict(

        batch_size=batch_size,

        num_workers=NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=(
            NUM_WORKERS > 0
        )

    )


    # Deterministic shuffle generator
    generator = torch.Generator()

    generator.manual_seed(SEED)


    train_loader = DataLoader(

        train_dataset,

        shuffle=True,

        generator=generator,

        **loader_options

    )


    val_loader = DataLoader(

        val_dataset,

        shuffle=False,

        **loader_options

    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HybridCNNBiLSTMAttention(

        num_classes=len(
            class_names
        )

    ).to(DEVICE)


    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    (
        df_summary,

        total_params,

        trainable_params,

        non_trainable

    ) = summarize_model(
        model
    )


    df_summary.to_csv(

        SUMMARY_CSV,

        index=False

    )


    with open(

        SUMMARY_TXT,

        "w",

        encoding="utf-8"

    ) as f:

        f.write(

            df_summary.to_string(
                index=False
            )

        )


        f.write(

            f"\n\n"
            f"Total params: {total_params:,}\n"
            f"Trainable params: {trainable_params:,}\n"
            f"Non-trainable params: {non_trainable:,}\n"

        )


    print("\n" + "=" * 70)

    print("MODEL")

    print("=" * 70)


    print(
        f"Total parameters: "
        f"{total_params:,}"
    )


    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )


    print(
        f"Non-trainable parameters: "
        f"{non_trainable:,}"
    )


    # IMPORTANT:
    # summarize_model() temporarily sets model.eval().
    # Put it back into training mode before training starts.

    model.train()


    # --------------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()


    optimizer = optim.Adam(

        model.parameters(),

        lr=lr

    )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []


    best_val_loss = float(
        "inf"
    )


    best_epoch = 0


    epochs_without_improvement = 0


    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("TRAINING CONFIGURATION")

    print("=" * 70)


    print(
        f"Device                : {DEVICE}"
    )

    print(
        f"Batch size            : {batch_size}"
    )

    print(
        f"Learning rate         : {lr}"
    )

    print(
        f"Maximum epochs        : {num_epochs}"
    )

    print(
        f"Minimum epochs        : {min_epochs}"
    )

    print(
        f"Early-stop patience   : {patience}"
    )

    print(
        f"Random seed           : {SEED}"
    )

    print(
        "Data augmentation      : NO"
    )

    print(
        "Validation split       : 20%"
    )

    print(
        "Checkpoint criterion   : validation loss"
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        num_epochs + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()


        train_loss_sum = 0.0

        train_correct = 0

        train_total = 0


        for images, labels in tqdm(

            train_loader,

            desc=(
                f"Epoch "
                f"{epoch}/{num_epochs} "
                f"Train"
            ),

            unit="batch"

        ):

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


            outputs, _ = model(
                images
            )


            loss = criterion(

                outputs,

                labels

            )


            loss.backward()


            optimizer.step()


            train_loss_sum += (

                loss.item()
                *
                labels.size(0)

            )


            predictions = outputs.argmax(
                dim=1
            )


            train_correct += (

                predictions == labels

            ).sum().item()


            train_total += labels.size(
                0
            )


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()


        val_loss_sum = 0.0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, labels in tqdm(

                val_loader,

                desc=(
                    f"Epoch "
                    f"{epoch}/{num_epochs} "
                    f"Validation"
                ),

                unit="batch"

            ):

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )


                outputs, _ = model(
                    images
                )


                loss = criterion(

                    outputs,

                    labels

                )


                val_loss_sum += (

                    loss.item()
                    *
                    labels.size(0)

                )


                predictions = outputs.argmax(
                    dim=1
                )


                val_correct += (

                    predictions == labels

                ).sum().item()


                val_total += labels.size(
                    0
                )


        # ====================================================
        # EPOCH METRICS
        # ====================================================

        train_loss = (

            train_loss_sum
            /
            train_total

        )


        train_accuracy = (

            train_correct
            /
            train_total

        )


        val_loss = (

            val_loss_sum
            /
            val_total

        )


        val_accuracy = (

            val_correct
            /
            val_total

        )


        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy

        })


        # Save after every epoch
        pd.DataFrame(
            history
        ).to_csv(

            HISTORY_CSV,

            index=False

        )


        print(

            f"\nEpoch {epoch:02d}: "

            f"Train Loss = "
            f"{train_loss:.4f} | "

            f"Train Acc = "
            f"{train_accuracy * 100:.2f}% | "

            f"Val Loss = "
            f"{val_loss:.4f} | "

            f"Val Acc = "
            f"{val_accuracy * 100:.2f}%"

        )


        # ====================================================
        # BEST MODEL CHECKPOINT
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch


            # Reset patience counter
            # once minimum epoch requirement is passed.

            if epoch > min_epochs:

                epochs_without_improvement = 0


            torch.save({

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "class_names":
                    class_names,

                "class_to_idx":
                    class_to_idx,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "seed":
                    SEED,

                "input_size":
                    INPUT_SIZE,

                "batch_size":
                    batch_size,

                "learning_rate":
                    lr,

                "augmentation":
                    False,

                "training_dataset":
                    "Dataset B"

            }, save_path)


            print(
                "Best validation loss improved."
            )

            print(
                "Checkpoint saved:"
            )

            print(
                save_path
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        else:

            if epoch <= min_epochs:

                print(

                    "No validation-loss improvement, "
                    "but early stopping is disabled "
                    f"through epoch {min_epochs}."

                )


            else:

                epochs_without_improvement += 1


                print(

                    "No validation-loss improvement: "
                    f"{epochs_without_improvement}/"
                    f"{patience}"

                )


                if (
                    epochs_without_improvement
                    >= patience
                ):

                    print(

                        f"\nEarly stopping at "
                        f"epoch {epoch}."

                    )

                    break


    # ========================================================
    # SAVE TRAINING CURVES
    # ========================================================

    history_df = pd.DataFrame(
        history
    )


    fig, axes = plt.subplots(

        1,

        2,

        figsize=(12, 5)

    )


    # Loss
    axes[0].plot(

        history_df["epoch"],

        history_df["train_loss"],

        label="Train"

    )


    axes[0].plot(

        history_df["epoch"],

        history_df["val_loss"],

        label="Validation"

    )


    axes[0].set_title(
        "Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()


    # Accuracy
    axes[1].plot(

        history_df["epoch"],

        100
        *
        history_df["train_accuracy"],

        label="Train"

    )


    axes[1].plot(

        history_df["epoch"],

        100
        *
        history_df["val_accuracy"],

        label="Validation"

    )


    axes[1].set_title(
        "Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy (%)"
    )

    axes[1].legend()


    fig.tight_layout()


    fig.savefig(

        TRAINING_CURVES_PATH,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close(
        fig
    )


    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\n" + "=" * 70)

    print("TRAINING COMPLETE")

    print("=" * 70)


    print(
        f"Best epoch          : "
        f"{best_epoch}"
    )


    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


    print(
        "\nBest model:"
    )

    print(
        MODEL_SAVE_PATH
    )


    print(
        "\nTraining history:"
    )

    print(
        HISTORY_CSV
    )


    print(
        "\nTraining curves:"
    )

    print(
        TRAINING_CURVES_PATH
    )


    print(
        "\nExact train-validation split:"
    )

    print(
        SPLIT_CSV
    )


    return (

        model,

        history_df,

        (
            total_params,
            trainable_params,
            non_trainable
        )

    )


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)

    print(
        "DATASET B TRAINING"
    )

    print("=" * 70)


    print(
        "\nDataset B training directory:"
    )

    print(
        DATASET_B_TRAIN_DIR
    )


    print(
        "\nDataset B test directory "
        "(NOT USED DURING TRAINING):"
    )

    print(
        DATASET_B_TEST_DIR
    )


    # --------------------------------------------------------
    # Verify paths
    # --------------------------------------------------------

    if not os.path.isdir(
        DATASET_B_TRAIN_DIR
    ):

        raise FileNotFoundError(

            "Dataset B training directory "
            "does not exist:\n"
            f"{DATASET_B_TRAIN_DIR}"

        )


    # We only warn about the test folder.
    # Training does NOT depend on it.

    if not os.path.isdir(
        DATASET_B_TEST_DIR
    ):

        print(

            "\nWARNING: Dataset B test directory "
            "was not found."

        )

        print(

            "This does NOT affect training because "
            "the test set is intentionally not used."

        )


    start_time = time.time()


    model, history_df, params_info = train_model(

        train_directory=
            DATASET_B_TRAIN_DIR,

        num_epochs=
            NUM_EPOCHS,

        batch_size=
            BATCH_SIZE,

        lr=
            LR,

        patience=
            PATIENCE,

        min_epochs=
            MIN_EPOCHS_BEFORE_EARLY_STOPPING,

        save_path=
            MODEL_SAVE_PATH

    )


    elapsed = (
        time.time()
        -
        start_time
    )


    print(

        f"\nTotal training time: "
        f"{elapsed / 60:.2f} minutes"

    )