# ============================================================
# PROPOSED MODEL
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================
#
# Architecture:
# CNN Residual Blocks -> BiLSTM -> Additive Attention -> FC
#
# Dataset:
# C:\user\juli\thesis\original_file\othr folders\
# PREPROCESSED_TRAINN_AUGMENTED
#
# IMPORTANT:
# 1. Same fixed 5 folds will later be used for all baselines.
# 2. No random augmentation is applied during CV.
# 3. Validation data is never used for training in that fold.
# 4. Every fold saves all important metrics and predictions.
# 5. Every fold saves its own checkpoint and history.
# 6. Fold indices are permanently saved.
# 7. Macro and micro metrics are saved.
# 8. Per-class metrics are saved.
# 9. Confusion matrices are saved.
# 10. Raw probabilities and predictions are saved.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import time
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    balanced_accuracy_score
)

from sklearn.preprocessing import label_binarize

from tqdm import tqdm


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\PREPROCESSED_TRAINN_AUGMENTED"
)

SAVE_ROOT = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\CV_RESULTS_PROPOSED_MODEL"
)


# Create output directory
os.makedirs(
    SAVE_ROOT,
    exist_ok=True
)


# ============================================================
# MODEL / TRAINING SETTINGS
# ============================================================

SEED = 42

N_SPLITS = 5

NUM_CLASSES = 4

INPUT_SIZE = (256, 256)

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

NUM_EPOCHS = 80

# Early stopping is based on validation accuracy.
# If validation accuracy does not improve for this many consecutive epochs,
# training for that fold stops and the best checkpoint is evaluated.
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.0

# IMPORTANT:
# Windows multiprocessing caused your previous error.
# Using 0 workers is safest for this experiment.
NUM_WORKERS = 0


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 80)
print("PROPOSED MODEL - 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 80)

print(f"Device       : {DEVICE}")
print(f"Dataset      : {DATA_DIR}")
print(f"Save folder  : {SAVE_ROOT}")
print(f"Seed         : {SEED}")
print(f"Folds        : {N_SPLITS}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Workers      : {NUM_WORKERS}")
print(f"Early stop   : patience={EARLY_STOPPING_PATIENCE}, monitor=val_accuracy")

print("=" * 80)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 4. TRANSFORM
# ============================================================
#
# Your dataset is already preprocessed and augmented.
#
# Therefore, DO NOT add RandomHorizontalFlip,
# RandomRotation, etc. here.
#
# Otherwise validation images could receive random
# transformations.
#
# ============================================================

base_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        INPUT_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ============================================================
# 5. RESIDUAL BLOCK
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

        # First convolution
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

        # Second convolution
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

        # Skip connection when channel count changes
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

            self.skip_bn = None

        # Optional pooling
        if use_pool:

            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

        else:

            self.pool = nn.Identity()


    def forward(self, x):

        identity = x

        # Conv 1
        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        # Conv 2
        out = self.conv2(out)

        out = self.bn2(out)

        # Skip connection
        if self.skip_conv is not None:

            identity = self.skip_conv(
                identity
            )

            identity = self.skip_bn(
                identity
            )

        # Residual addition
        out = out + identity

        out = self.relu(out)

        # Pooling
        out = self.pool(out)

        return out


# ============================================================
# 6. CNN FEATURE EXTRACTOR
# ============================================================

class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        # Block 1
        self.block1 = ResidualBlock(
            3,
            32,
            use_pool=True
        )

        # Block 2
        self.block2 = ResidualBlock(
            32,
            64,
            use_pool=True
        )

        # Block 3
        self.block3 = ResidualBlock(
            64,
            128,
            use_pool=True
        )

        # Block 4
        self.block4 = ResidualBlock(
            128,
            128,
            use_pool=True
        )

        # Block 5
        self.block5 = ResidualBlock(
            128,
            256,
            use_pool=False
        )

        # Final adaptive pooling
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
# 7. ADDITIVE ATTENTION
# ============================================================

class AdditiveAttention(nn.Module):

    def __init__(
        self,
        hidden_dim
    ):

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


    def forward(
        self,
        lstm_outputs
    ):

        # lstm_outputs:
        # (batch, sequence_length, hidden_dim*2)

        e = self.tanh(
            self.attn(
                lstm_outputs
            )
        )

        scores = self.score(
            e
        )

        alpha = self.softmax(
            scores
        )

        context = torch.sum(
            alpha * lstm_outputs,
            dim=1
        )

        return context, alpha


# ============================================================
# 8. PROPOSED HYBRID MODEL
# ============================================================

class HybridCNNBiLSTMAttention(
    nn.Module
):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        lstm_hidden=128,
        lstm_layers=1
    ):

        super().__init__()

        # CNN
        self.cnn = CNNFeatureExtractor()

        self.hidden_size = (
            lstm_hidden
        )

        self.num_layers = (
            lstm_layers
        )

        # CNN output:
        # (batch, 256, 8, 8)

        c_channels = 256

        c_w = 8

        self.lstm_input_size = (
            c_channels * c_w
        )

        # BiLSTM
        self.bilstm = nn.LSTM(

            input_size=
                self.lstm_input_size,

            hidden_size=
                self.hidden_size,

            num_layers=
                self.num_layers,

            batch_first=True,

            bidirectional=True,

            dropout=(
                0.3
                if self.num_layers > 1
                else 0.0
            )
        )

        # Attention
        self.attention = AdditiveAttention(
            self.hidden_size
        )

        # Fully connected classifier
        self.fc = nn.Sequential(

            nn.Linear(
                self.hidden_size * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # CNN feature extraction
        x = self.cnn(x)

        # Expected:
        # batch, 256, 8, 8

        b, c, h, w = x.size()

        # Reshape feature map into sequence
        #
        # sequence length = h = 8
        #
        # input size = c*w
        #             = 256*8
        #             = 2048

        x = x.view(
            b,
            h,
            c * w
        )

        # BiLSTM
        lstm_out, _ = self.bilstm(
            x
        )

        # Additive attention
        context, attention_weights = (
            self.attention(
                lstm_out
            )
        )

        # Classification
        out = self.fc(
            context
        )

        return out, attention_weights


# ============================================================
# 9. COUNT PARAMETERS
# ============================================================

def count_parameters(model):

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
        total_params
        - trainable_params
    )

    return (
        total_params,
        trainable_params,
        non_trainable_params
    )


# ============================================================
# 10. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=base_transform
)

print(
    f"Total images: {len(dataset)}"
)

print("\nClass mapping:")

for class_name, class_index in (
    dataset.class_to_idx.items()
):

    print(
        f"  {class_index}: {class_name}"
    )


# ============================================================
# 11. VERIFY CLASS STRUCTURE
# ============================================================

if len(dataset.classes) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes "
        f"but found {len(dataset.classes)}: "
        f"{dataset.classes}"
    )


# ============================================================
# 12. GET LABELS
# ============================================================

targets = np.array(
    dataset.targets
)


# ============================================================
# 13. CLASS DISTRIBUTION
# ============================================================

class_distribution = {}

for class_index, class_name in enumerate(
    dataset.classes
):

    count = int(
        np.sum(
            targets == class_index
        )
    )

    class_distribution[
        class_name
    ] = count


print("\nClass distribution:")

for class_name, count in (
    class_distribution.items()
):

    print(
        f"  {class_name}: {count}"
    )


# ============================================================
# 14. SAVE EXPERIMENT CONFIGURATION
# ============================================================

experiment_config = {

    "experiment":
        "Proposed Model 5-Fold Stratified CV",

    "architecture":
        "CNN Residual Blocks -> BiLSTM -> Additive Attention -> FC",

    "dataset_path":
        DATA_DIR,

    "total_images":
        int(len(dataset)),

    "classes":
        dataset.classes,

    "class_to_idx":
        dataset.class_to_idx,

    "class_distribution":
        class_distribution,

    "num_classes":
        NUM_CLASSES,

    "input_size":
        list(INPUT_SIZE),

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "num_epochs":
        NUM_EPOCHS,

    "n_splits":
        N_SPLITS,

    "seed":
        SEED,

    "num_workers":
        NUM_WORKERS,

    "device":
        str(DEVICE),

    "early_stopping":
        True,

    "early_stopping_patience":
        EARLY_STOPPING_PATIENCE,

    "early_stopping_monitor":
        "val_accuracy",

    "min_delta":
        MIN_DELTA,

    "checkpoint_format":
        "model_history_rng_lists_no_optimizer_v3",

    "optimizer_state_saved":
        False
}


with open(
    os.path.join(
        SAVE_ROOT,
        "experiment_configuration.json"
    ),
    "w"
) as f:

    json.dump(
        experiment_config,
        f,
        indent=4
    )


# ============================================================
# 15. CREATE OR REUSE FIXED STRATIFIED 5-FOLD SPLITS
# ============================================================

FIXED_FOLDS_FILE = os.path.join(
    SAVE_ROOT,
    "FIXED_5_FOLD_SPLITS.npz"
)


if os.path.exists(FIXED_FOLDS_FILE):

    print("\nExisting fixed 5-fold split file found.")
    print("Reusing it to protect the original fold assignments:")
    print(FIXED_FOLDS_FILE)

    loaded_folds = np.load(
        FIXED_FOLDS_FILE,
        allow_pickle=False
    )

    fold_indices = {}

    for fold_number in range(1, N_SPLITS + 1):

        train_key = f"fold_{fold_number}_train"
        val_key = f"fold_{fold_number}_validation"

        if train_key not in loaded_folds.files or val_key not in loaded_folds.files:
            raise RuntimeError(
                f"Existing fixed-fold file is missing {train_key} or {val_key}."
            )

        fold_indices[train_key] = loaded_folds[train_key].astype(np.int64)
        fold_indices[val_key] = loaded_folds[val_key].astype(np.int64)

        train_set = set(fold_indices[train_key].tolist())
        val_set = set(fold_indices[val_key].tolist())

        if len(train_set) != len(fold_indices[train_key]):
            raise RuntimeError(
                f"Duplicate training indices detected in Fold {fold_number}."
            )

        if len(val_set) != len(fold_indices[val_key]):
            raise RuntimeError(
                f"Duplicate validation indices detected in Fold {fold_number}."
            )

        if train_set.intersection(val_set):
            raise RuntimeError(
                f"Train/validation overlap detected in Fold {fold_number}."
            )

        if len(fold_indices[train_key]) + len(fold_indices[val_key]) != len(dataset):
            raise RuntimeError(
                f"Fold {fold_number} does not match the current dataset size "
                f"({len(dataset)})."
            )

        if np.any(fold_indices[train_key] < 0) or np.any(fold_indices[train_key] >= len(dataset)):
            raise RuntimeError(
                f"Invalid training index detected in Fold {fold_number}."
            )

        if np.any(fold_indices[val_key] < 0) or np.any(fold_indices[val_key] >= len(dataset)):
            raise RuntimeError(
                f"Invalid validation index detected in Fold {fold_number}."
            )

        print(
            f"Fold {fold_number}: "
            f"Train = {len(fold_indices[train_key])}, "
            f"Validation = {len(fold_indices[val_key])}"
        )

else:

    print("\nCreating fixed 5-fold splits...")

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_indices = {}

    for fold_number, (
        train_indices,
        validation_indices
    ) in enumerate(
        skf.split(
            np.zeros(len(targets)),
            targets
        ),
        start=1
    ):

        fold_indices[f"fold_{fold_number}_train"] = train_indices.astype(np.int64)
        fold_indices[f"fold_{fold_number}_validation"] = validation_indices.astype(np.int64)

        print(
            f"Fold {fold_number}: "
            f"Train = {len(train_indices)}, "
            f"Validation = {len(validation_indices)}"
        )

    np.savez_compressed(
        FIXED_FOLDS_FILE,
        **fold_indices
    )

    print("\nFixed folds created and saved to:")
    print(FIXED_FOLDS_FILE)


# ============================================================
# 17. ALSO SAVE HUMAN-READABLE FOLD ASSIGNMENTS
# ============================================================

fold_assignment_rows = []


for fold_number in range(
    1,
    N_SPLITS + 1
):

    train_indices = fold_indices[
        f"fold_{fold_number}_train"
    ]

    validation_indices = fold_indices[
        f"fold_{fold_number}_validation"
    ]


    for index in train_indices:

        fold_assignment_rows.append({

            "sample_index":
                int(index),

            "fold":
                fold_number,

            "split":
                "train"
        })


    for index in validation_indices:

        fold_assignment_rows.append({

            "sample_index":
                int(index),

            "fold":
                fold_number,

            "split":
                "validation"
        })


fold_assignment_df = pd.DataFrame(
    fold_assignment_rows
)


fold_assignment_df.to_csv(

    os.path.join(
        SAVE_ROOT,
        "ALL_FOLD_SAMPLE_ASSIGNMENTS.csv"
    ),

    index=False
)


# ============================================================
# 18. METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probs
):

    metrics = {}


    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    metrics[
        "accuracy"
    ] = accuracy


    # --------------------------------------------------------
    # Micro accuracy
    #
    # For single-label multiclass classification:
    # micro accuracy = overall accuracy.
    # --------------------------------------------------------

    metrics[
        "micro_accuracy"
    ] = accuracy_score(
        y_true,
        y_pred
    )


    # --------------------------------------------------------
    # Macro-class accuracy
    #
    # Mean of class-wise recalls.
    # Equivalent to balanced accuracy.
    # --------------------------------------------------------

    metrics[
        "macro_class_accuracy"
    ] = balanced_accuracy_score(
        y_true,
        y_pred
    )


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    metrics[
        "macro_precision"
    ] = precision_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0
    )


    metrics[
        "micro_precision"
    ] = precision_score(

        y_true,

        y_pred,

        average="micro",

        zero_division=0
    )


    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    metrics[
        "macro_recall"
    ] = recall_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0
    )


    metrics[
        "micro_recall"
    ] = recall_score(

        y_true,

        y_pred,

        average="micro",

        zero_division=0
    )


    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    metrics[
        "macro_f1"
    ] = f1_score(

        y_true,

        y_pred,

        average="macro",

        zero_division=0
    )


    metrics[
        "micro_f1"
    ] = f1_score(

        y_true,

        y_pred,

        average="micro",

        zero_division=0
    )


    # --------------------------------------------------------
    # Multiclass ROC-AUC
    # One-vs-rest
    # --------------------------------------------------------

    y_true_bin = label_binarize(

        y_true,

        classes=np.arange(
            NUM_CLASSES
        )
    )


    try:

        macro_auc = roc_auc_score(

            y_true_bin,

            y_probs,

            average="macro",

            multi_class="ovr"
        )

    except Exception:

        macro_auc = np.nan


    try:

        micro_auc = roc_auc_score(

            y_true_bin,

            y_probs,

            average="micro",

            multi_class="ovr"
        )

    except Exception:

        micro_auc = np.nan


    metrics[
        "macro_auc"
    ] = macro_auc


    metrics[
        "micro_auc"
    ] = micro_auc


    return metrics


# ============================================================
# ============================================================
# 19. CHECKPOINT / RESUME HELPERS
# ============================================================

def atomic_torch_save(obj, path):
    """Safely write a PyTorch checkpoint and clean failed temporary files."""
    tmp_path = path + ".tmp"

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def cpu_state_dict(model):
    """Return a detached CPU copy of model parameters for robust serialization."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def save_json_atomic(obj, path):
    """Safely write JSON without leaving a partial file."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
    os.replace(tmp_path, path)


def hash_array(arr):
    """Stable hash used to verify that a resumed fold uses identical indices."""
    import hashlib
    return hashlib.sha256(np.asarray(arr, dtype=np.int64).tobytes()).hexdigest()


def _torch_rng_to_list(rng_tensor):
    """Convert a torch RNG ByteTensor to plain Python integers.

    This avoids serializing RNG tensors directly, which is especially useful
    on the current Python 3.14 / PyTorch environment.
    """
    return rng_tensor.detach().cpu().to(torch.uint8).tolist()


def _list_to_torch_rng(values):
    """Reconstruct a torch RNG ByteTensor from a saved integer list."""
    return torch.tensor(values, dtype=torch.uint8, device="cpu")


def get_rng_state(train_generator=None):
    # Store RNG tensors as ordinary lists rather than tensors.
    # This prevents RNG-state serialization/deserialization problems.
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": _torch_rng_to_list(torch.get_rng_state()),
    }

    if torch.cuda.is_available():
        state["cuda"] = [
            _torch_rng_to_list(x)
            for x in torch.cuda.get_rng_state_all()
        ]

    if train_generator is not None:
        state["train_generator"] = _torch_rng_to_list(
            train_generator.get_state()
        )

    return state


def restore_rng_state(state, train_generator=None):
    """Restore RNG states when the saved representation is valid.

    Legacy checkpoints may contain RNG tensors that were serialized by the
    previous implementation. If one of those states is malformed, we skip
    that particular RNG restore rather than aborting the entire experiment.
    """
    if not state:
        return

    try:
        if "python" in state:
            random.setstate(state["python"])
    except Exception as exc:
        print(f"WARNING: Could not restore Python RNG state: {exc}")

    try:
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except Exception as exc:
        print(f"WARNING: Could not restore NumPy RNG state: {exc}")

    # Torch RNG: support both the new list format and a valid legacy ByteTensor.
    try:
        torch_state = state.get("torch")
        if torch_state is not None:
            if isinstance(torch_state, torch.Tensor):
                if torch_state.dtype == torch.uint8:
                    torch.set_rng_state(torch_state.cpu())
                else:
                    print(
                        "WARNING: Legacy torch RNG state has invalid dtype "
                        f"{torch_state.dtype}; skipping torch RNG restore."
                    )
            elif isinstance(torch_state, (list, tuple)):
                torch.set_rng_state(_list_to_torch_rng(torch_state))
            else:
                print(
                    "WARNING: Unrecognized torch RNG state format; "
                    "skipping torch RNG restore."
                )
    except Exception as exc:
        print(f"WARNING: Could not restore torch RNG state: {exc}")

    # CUDA RNG: support both the new list format and legacy tensors.
    if torch.cuda.is_available() and "cuda" in state:
        try:
            cuda_states = state["cuda"]
            restored_cuda = []
            for item in cuda_states:
                if isinstance(item, torch.Tensor):
                    if item.dtype == torch.uint8:
                        restored_cuda.append(item.cpu())
                    else:
                        raise TypeError(
                            f"invalid CUDA RNG dtype {item.dtype}"
                        )
                elif isinstance(item, (list, tuple)):
                    restored_cuda.append(_list_to_torch_rng(item))
                else:
                    raise TypeError(
                        f"unrecognized CUDA RNG state type {type(item)}"
                    )
            torch.cuda.set_rng_state_all(restored_cuda)
        except Exception as exc:
            print(f"WARNING: Could not restore CUDA RNG state: {exc}")

    if train_generator is not None and "train_generator" in state:
        try:
            generator_state = state["train_generator"]
            if isinstance(generator_state, torch.Tensor):
                if generator_state.dtype == torch.uint8:
                    train_generator.set_state(generator_state.cpu())
                else:
                    raise TypeError(
                        f"invalid DataLoader generator dtype {generator_state.dtype}"
                    )
            elif isinstance(generator_state, (list, tuple)):
                train_generator.set_state(_list_to_torch_rng(generator_state))
            else:
                raise TypeError(
                    f"unrecognized DataLoader generator state type {type(generator_state)}"
                )
        except Exception as exc:
            print(
                "WARNING: Could not restore DataLoader generator RNG state: "
                f"{exc}"
            )


def load_checkpoint(path):
    """Load a checkpoint and safely discard any legacy optimizer state.

    Older checkpoints from the failed run may still contain
    ``optimizer_state_dict``. We never restore or re-save that object.
    """
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    # A legacy checkpoint may contain the optimizer state. Remove it
    # immediately so it cannot be accidentally propagated into a new
    # checkpoint or consumed during resume.
    checkpoint.pop("optimizer_state_dict", None)

    return checkpoint


# ============================================================
# 20. TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    fold_number,
    train_indices,
    validation_indices
):

    print("\n")
    print("=" * 80)
    print(f"STARTING FOLD {fold_number}/{N_SPLITS}")
    print("=" * 80)

    fold_dir = os.path.join(
        SAVE_ROOT,
        f"Fold_{fold_number}"
    )

    checkpoint_dir = os.path.join(
        fold_dir,
        "checkpoints"
    )

    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_checkpoint_path = os.path.join(
        checkpoint_dir,
        "LATEST_CHECKPOINT.pth"
    )

    best_checkpoint_path = os.path.join(
        fold_dir,
        "BEST_MODEL.pth"
    )

    progress_path = os.path.join(
        fold_dir,
        "fold_progress.json"
    )

    complete_path = os.path.join(
        fold_dir,
        "FOLD_COMPLETE.json"
    )

    # --------------------------------------------------------
    # Save fold indices and validation samples
    # --------------------------------------------------------
    np.save(
        os.path.join(fold_dir, "train_indices.npy"),
        train_indices
    )

    np.save(
        os.path.join(fold_dir, "validation_indices.npy"),
        validation_indices
    )

    validation_labels = targets[validation_indices]

    pd.DataFrame({
        "sample_index": validation_indices,
        "true_label": validation_labels,
        "true_class": [dataset.classes[x] for x in validation_labels]
    }).to_csv(
        os.path.join(fold_dir, "validation_samples.csv"),
        index=False
    )

    # --------------------------------------------------------
    # Dataset subsets
    # --------------------------------------------------------
    train_subset = Subset(dataset, train_indices)
    validation_subset = Subset(dataset, validation_indices)

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED + fold_number)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Reproducible model initialization
    # --------------------------------------------------------
    set_seed(SEED + fold_number)

    model = HybridCNNBiLSTMAttention(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    total_params, trainable_params, non_trainable_params = count_parameters(model)

    parameter_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }

    with open(
        os.path.join(fold_dir, "parameter_info.json"),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(parameter_info, f, indent=4)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------
    history = []
    best_val_accuracy = -1.0
    best_epoch = -1
    best_model_state = None
    early_stopping_counter = 0
    start_epoch = 1
    resumed = False

    # --------------------------------------------------------
    # RESUME IF A LATEST CHECKPOINT EXISTS
    # --------------------------------------------------------
    if os.path.exists(latest_checkpoint_path):
        print("\nLatest checkpoint found.")
        print(f"Resuming Fold {fold_number} from:")
        print(latest_checkpoint_path)

        checkpoint = load_checkpoint(latest_checkpoint_path)

        # Basic compatibility checks
        if checkpoint.get("fold") != fold_number:
            raise RuntimeError(
                f"Checkpoint fold mismatch: expected Fold {fold_number}, "
                f"found Fold {checkpoint.get('fold')}."
            )

        if checkpoint.get("data_dir") != DATA_DIR:
            raise RuntimeError(
                "Checkpoint dataset path does not match the current DATA_DIR."
            )

        if checkpoint.get("train_indices_hash") != hash_array(train_indices):
            raise RuntimeError(
                f"Fold {fold_number} training indices do not match the checkpoint."
            )

        if checkpoint.get("validation_indices_hash") != hash_array(validation_indices):
            raise RuntimeError(
                f"Fold {fold_number} validation indices do not match the checkpoint."
            )

        model.load_state_dict(checkpoint["model_state_dict"])

        # The optimizer state is intentionally not restored. The previous
        # checkpoint implementation stored Adam's optimizer tensors and
        # caused torch.save() to fail on the current environment. A fresh
        # Adam optimizer is created above and training resumes from the
        # restored model weights.

        history = checkpoint.get("history", [])
        best_val_accuracy = checkpoint.get("best_val_accuracy", -1.0)
        best_epoch = checkpoint.get("best_epoch", -1)
        early_stopping_counter = checkpoint.get("early_stopping_counter", 0)

        if checkpoint.get("best_model_state_dict") is not None:
            best_model_state = checkpoint["best_model_state_dict"]
        elif os.path.exists(best_checkpoint_path):
            best_checkpoint = load_checkpoint(best_checkpoint_path)
            best_model_state = best_checkpoint["model_state_dict"]

        last_completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = last_completed_epoch + 1
        resumed = True

        restore_rng_state(
            checkpoint.get("rng_state"),
            train_generator
        )

        print(
            f"Resume point: epoch {last_completed_epoch} completed. "
            f"Next epoch = {start_epoch}."
        )

        if early_stopping_counter > 0:
            print(
                f"Early-stopping counter restored: "
                f"{early_stopping_counter}/{EARLY_STOPPING_PATIENCE}"
            )

    else:
        print(f"No checkpoint found for Fold {fold_number}. Starting from epoch 1.")

    # If all requested epochs were already completed, skip training and use
    # the saved best model.
    if start_epoch > NUM_EPOCHS:
        print(
            f"Fold {fold_number} already reached the configured maximum "
            f"of {NUM_EPOCHS} epochs."
        )

    # ========================================================
    # EPOCH LOOP
    # ========================================================
    if start_epoch <= NUM_EPOCHS:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):

            # ====================================================
            # TRAINING
            # ====================================================
            model.train()

            running_train_loss = 0.0
            running_train_correct = 0
            running_train_total = 0

            train_bar = tqdm(
                train_loader,
                desc=(
                    f"Fold {fold_number} "
                    f"Epoch {epoch}/{NUM_EPOCHS} TRAIN"
                ),
                leave=False
            )

            for images, labels in train_bar:
                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                outputs, _ = model(images)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_train_loss += (
                    loss.item() * images.size(0)
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                running_train_correct += (
                    (predictions == labels).sum().item()
                )

                running_train_total += labels.size(0)

            train_loss = (
                running_train_loss / running_train_total
            )

            train_accuracy = (
                running_train_correct / running_train_total
            )

            # ====================================================
            # VALIDATION
            # ====================================================
            model.eval()

            running_val_loss = 0.0
            running_val_correct = 0
            running_val_total = 0

            with torch.no_grad():
                val_bar = tqdm(
                    validation_loader,
                    desc=(
                        f"Fold {fold_number} "
                        f"Epoch {epoch}/{NUM_EPOCHS} VAL"
                    ),
                    leave=False
                )

                for images, labels in val_bar:
                    images = images.to(
                        DEVICE,
                        non_blocking=True
                    )

                    labels = labels.to(
                        DEVICE,
                        non_blocking=True
                    )

                    outputs, _ = model(images)

                    loss = criterion(outputs, labels)

                    running_val_loss += (
                        loss.item() * images.size(0)
                    )

                    predictions = torch.argmax(
                        outputs,
                        dim=1
                    )

                    running_val_correct += (
                        (predictions == labels).sum().item()
                    )

                    running_val_total += labels.size(0)

            val_loss = (
                running_val_loss / running_val_total
            )

            val_accuracy = (
                running_val_correct / running_val_total
            )

            # ====================================================
            # UPDATE BEST MODEL
            # ====================================================
            improved = val_accuracy > (
                best_val_accuracy + MIN_DELTA
            )

            if improved:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                best_model_state = cpu_state_dict(model)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # ====================================================
            # SAVE HISTORY AFTER EVERY EPOCH
            # ====================================================
            epoch_record = {
                "fold": fold_number,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "is_best": bool(improved),
                "early_stopping_counter": early_stopping_counter
            }

            history.append(epoch_record)

            history_df = pd.DataFrame(history)

            history_df.to_csv(
                os.path.join(
                    fold_dir,
                    "training_history.csv"
                ),
                index=False
            )

            # ====================================================
            # SAVE BEST MODEL
            # ====================================================
            if improved:
                best_checkpoint = {
                    "checkpoint_type": "best_model",
                    "checkpoint_format": "model_history_rng_only_v2",
                    "fold": fold_number,
                    "epoch": epoch,
                    "model_state_dict": cpu_state_dict(model),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                    "optimizer_state_saved": False,
                    "best_val_accuracy": best_val_accuracy,
                    "best_epoch": best_epoch,
                    "seed": SEED + fold_number,
                    "class_names": CLASS_NAMES,
                    "architecture": (
                        "CNN Residual Blocks -> "
                        "BiLSTM -> Additive Attention -> FC"
                    ),
                    "data_dir": DATA_DIR,
                    "train_indices_hash": hash_array(train_indices),
                    "validation_indices_hash": hash_array(validation_indices)
                }

                atomic_torch_save(
                    best_checkpoint,
                    best_checkpoint_path
                )

            # ====================================================
            # SAVE RESUMABLE CHECKPOINT AFTER EVERY EPOCH
            # ====================================================
            checkpoint_state = {
                "checkpoint_type": "latest_epoch",
                "checkpoint_format": "model_history_rng_only_v2",
                "fold": fold_number,
                "epoch": epoch,
                "num_epochs": NUM_EPOCHS,
                "model_state_dict": cpu_state_dict(model),
                "history": history,
                "optimizer_state_saved": False,
                "best_val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "early_stopping_counter": early_stopping_counter,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "min_delta": MIN_DELTA,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "seed": SEED + fold_number,
                "data_dir": DATA_DIR,
                "train_indices_hash": hash_array(train_indices),
                "validation_indices_hash": hash_array(validation_indices),
                "class_names": CLASS_NAMES,
                "architecture": (
                    "CNN Residual Blocks -> "
                    "BiLSTM -> Additive Attention -> FC"
                ),
                "rng_state": get_rng_state(train_generator)
            }

            # Permanent epoch checkpoint: checkpoint_epoch_001.pth, etc.
            epoch_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch:03d}.pth"
            )

            atomic_torch_save(
                checkpoint_state,
                epoch_checkpoint_path
            )

            # Latest checkpoint always points to the most recently completed
            # epoch. This is the file used for automatic resume.
            #
            # IMPORTANT: optimizer_state_dict is intentionally absent.
            # The previous implementation crashed inside torch.save() while
            # serializing Adam's exp_avg_sq on this environment. A fresh Adam
            # optimizer is created when a run is resumed.
            atomic_torch_save(
                checkpoint_state,
                latest_checkpoint_path
            )

            # ----------------------------------------------------
            # Save human-readable progress immediately
            # ----------------------------------------------------
            progress = {
                "fold": fold_number,
                "status": "training",
                "resumed": resumed,
                "last_completed_epoch": epoch,
                "next_epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
                "early_stopping_counter": early_stopping_counter,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "checkpoint": epoch_checkpoint_path
            }

            save_json_atomic(
                progress,
                progress_path
            )

            print(
                f"Fold {fold_number} | "
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy * 100:.2f}% | "
                f"Best: {best_val_accuracy * 100:.2f}% "
                f"(Epoch {best_epoch}) | "
                f"Patience: {early_stopping_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            # ====================================================
            # EARLY STOPPING
            # ====================================================
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping triggered for Fold {fold_number}. "
                    f"No validation-accuracy improvement for "
                    f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
                )
                break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================
    if best_model_state is None:
        if os.path.exists(best_checkpoint_path):
            best_checkpoint = load_checkpoint(
                best_checkpoint_path
            )
            best_model_state = best_checkpoint[
                "model_state_dict"
            ]
            best_val_accuracy = best_checkpoint.get(
                "best_val_accuracy",
                best_checkpoint.get("val_accuracy", -1.0)
            )
            best_epoch = best_checkpoint.get(
                "best_epoch",
                best_checkpoint.get("epoch", -1)
            )
        else:
            raise RuntimeError(
                f"No best model was saved for Fold {fold_number}."
            )

    model.load_state_dict(best_model_state)
    model.eval()

    # ========================================================
    # FINAL VALIDATION PREDICTIONS
    # ========================================================
    all_true = []
    all_pred = []
    all_probs = []
    all_sample_indices = []

    current_position = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs, _ = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            batch_size_actual = labels.size(0)

            batch_indices = validation_indices[
                current_position:
                current_position + batch_size_actual
            ]

            current_position += batch_size_actual

            all_sample_indices.extend(
                batch_indices.tolist()
            )

            all_true.extend(
                labels.cpu().numpy().tolist()
            )

            all_pred.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probs.extend(
                probabilities.cpu().numpy().tolist()
            )

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_probs = np.array(all_probs)
    sample_indices = np.array(all_sample_indices)

    if not np.array_equal(
        sample_indices,
        validation_indices
    ):
        raise RuntimeError(
            f"Sample ordering mismatch in Fold {fold_number}."
        )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================
    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probs
    )

    metrics["fold"] = fold_number
    metrics["best_epoch"] = best_epoch
    metrics["best_val_accuracy"] = best_val_accuracy
    metrics["validation_samples"] = len(y_true)
    metrics["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
    metrics["epochs_completed"] = len(history)
    metrics["resumed"] = resumed

    # ========================================================
    # PER-CLASS METRICS
    # ========================================================
    per_class_precision = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    per_class_recall = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    per_class_rows = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        per_class_rows.append({
            "class_index": class_index,
            "class_name": class_name,
            "precision": per_class_precision[class_index],
            "recall": per_class_recall[class_index],
            "f1": per_class_f1[class_index],
            "support": int(np.sum(y_true == class_index))
        })

    per_class_df = pd.DataFrame(per_class_rows)

    per_class_df.to_csv(
        os.path.join(
            fold_dir,
            "per_class_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        os.path.join(
            fold_dir,
            "confusion_matrix.csv"
        )
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=6,
        zero_division=0
    )

    with open(
        os.path.join(
            fold_dir,
            "classification_report.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    # ========================================================
    # SAVE RAW PREDICTIONS
    # ========================================================
    prediction_df = pd.DataFrame({
        "sample_index": sample_indices,
        "true_label": y_true,
        "true_class": [CLASS_NAMES[x] for x in y_true],
        "predicted_label": y_pred,
        "predicted_class": [CLASS_NAMES[x] for x in y_pred],
        "prob_glioma": y_probs[:, 0],
        "prob_meningioma": y_probs[:, 1],
        "prob_notumor": y_probs[:, 2],
        "prob_pituitary": y_probs[:, 3]
    })

    prediction_df.to_csv(
        os.path.join(
            fold_dir,
            "fold_predictions.csv"
        ),
        index=False
    )

    np.savez_compressed(
        os.path.join(
            fold_dir,
            "raw_predictions.npz"
        ),
        sample_indices=sample_indices,
        y_true=y_true,
        y_pred=y_pred,
        y_probs=y_probs
    )

    # ========================================================
    # SAVE FOLD METRICS
    # ========================================================
    fold_metrics_df = pd.DataFrame([metrics])

    fold_metrics_df.to_csv(
        os.path.join(
            fold_dir,
            "fold_metrics.csv"
        ),
        index=False
    )

    # ========================================================
    # SAVE FOLD CONFIGURATION
    # ========================================================
    fold_config = {
        "fold": fold_number,
        "seed": SEED + fold_number,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_val_accuracy),
        "epochs_completed": int(len(history)),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "num_workers": NUM_WORKERS,
        "early_stopping": True,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "val_accuracy",
        "min_delta": MIN_DELTA,
        "resumed": resumed,
        "architecture": (
            "CNN Residual Blocks -> "
            "BiLSTM -> Additive Attention -> FC"
        ),
        "input_size": list(INPUT_SIZE),
        "classes": CLASS_NAMES,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params
    }

    save_json_atomic(
        fold_config,
        os.path.join(
            fold_dir,
            "fold_configuration.json"
        )
    )

    # ========================================================
    # SAVE CURVES
    # ========================================================
    history_df = pd.DataFrame(history)

    plt.figure(figsize=(10, 6))
    plt.plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        label="Training Accuracy"
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_accuracy"],
        label="Validation Accuracy"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"Proposed Model - Fold {fold_number} Accuracy"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fold_dir,
            "accuracy_curve.png"
        ),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training Loss"
    )
    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="Validation Loss"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Proposed Model - Fold {fold_number} Loss"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fold_dir,
            "loss_curve.png"
        ),
        dpi=300
    )
    plt.close()

    # ========================================================
    # MARK FOLD COMPLETE
    # ========================================================
    completion_record = {
        "fold": fold_number,
        "status": "complete",
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "epochs_completed": int(len(history)),
        "early_stopped": bool(
            len(history) < NUM_EPOCHS
            and len(history) > 0
            and history[-1].get("early_stopping_counter", 0)
            >= EARLY_STOPPING_PATIENCE
        ),
        "metrics_file": os.path.join(
            fold_dir,
            "fold_metrics.csv"
        ),
        "best_model_file": best_checkpoint_path,
        "latest_checkpoint_file": latest_checkpoint_path
    }

    save_json_atomic(
        completion_record,
        complete_path
    )

    save_json_atomic(
        completion_record,
        progress_path
    )

    # ========================================================
    # PRINT FOLD RESULT
    # ========================================================
    print("\n")
    print("-" * 80)
    print(f"FOLD {fold_number} FINAL RESULT")
    print("-" * 80)

    print(
        f"Accuracy             : "
        f"{metrics['accuracy'] * 100:.4f}%"
    )
    print(
        f"Micro Accuracy       : "
        f"{metrics['micro_accuracy'] * 100:.4f}%"
    )
    print(
        f"Macro-class Accuracy : "
        f"{metrics['macro_class_accuracy'] * 100:.4f}%"
    )
    print(
        f"Macro Precision      : "
        f"{metrics['macro_precision']:.6f}"
    )
    print(
        f"Micro Precision      : "
        f"{metrics['micro_precision']:.6f}"
    )
    print(
        f"Macro Recall         : "
        f"{metrics['macro_recall']:.6f}"
    )
    print(
        f"Micro Recall         : "
        f"{metrics['micro_recall']:.6f}"
    )
    print(
        f"Macro F1             : "
        f"{metrics['macro_f1']:.6f}"
    )
    print(
        f"Micro F1             : "
        f"{metrics['micro_f1']:.6f}"
    )
    print(
        f"Macro ROC-AUC        : "
        f"{metrics['macro_auc']:.6f}"
    )
    print(
        f"Micro ROC-AUC        : "
        f"{metrics['micro_auc']:.6f}"
    )
    print(f"Best Epoch           : {best_epoch}")
    print(f"Epochs Completed     : {len(history)}")
    print("-" * 80)

    return metrics


# 20. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("BEGINNING 5-FOLD CROSS-VALIDATION")
    print("=" * 80)


    overall_start = time.time()


    all_fold_metrics = []


    # ========================================================
    # RUN / RESUME FIVE FOLDS
    # ========================================================

    for fold_number in range(1, N_SPLITS + 1):

        train_indices = fold_indices[
            f"fold_{fold_number}_train"
        ]

        validation_indices = fold_indices[
            f"fold_{fold_number}_validation"
        ]

        fold_dir = os.path.join(
            SAVE_ROOT,
            f"Fold_{fold_number}"
        )

        complete_path = os.path.join(
            fold_dir,
            "FOLD_COMPLETE.json"
        )

        metrics_path = os.path.join(
            fold_dir,
            "fold_metrics.csv"
        )

        if os.path.exists(complete_path) and os.path.exists(metrics_path):
            print("\n" + "=" * 80)
            print(f"FOLD {fold_number} ALREADY COMPLETE - SKIPPING")
            print("=" * 80)

            completed_metrics_df = pd.read_csv(metrics_path)

            if len(completed_metrics_df) != 1:
                raise RuntimeError(
                    f"Invalid fold_metrics.csv for Fold {fold_number}."
                )

            all_fold_metrics.append(
                completed_metrics_df.iloc[0].to_dict()
            )
            continue

        fold_metrics = train_one_fold(
            fold_number,
            train_indices,
            validation_indices
        )

        all_fold_metrics.append(fold_metrics)


    # ========================================================
    # COMPLETE FOLD RESULTS
    # ========================================================

    fold_results_df = pd.DataFrame(

        all_fold_metrics
    )


    # Desired order
    result_columns = [

        "fold",

        "accuracy",

        "micro_accuracy",

        "macro_class_accuracy",

        "macro_precision",

        "micro_precision",

        "macro_recall",

        "micro_recall",

        "macro_f1",

        "micro_f1",

        "macro_auc",

        "micro_auc",

        "best_epoch",

        "best_val_accuracy",

        "validation_samples"
    ]


    fold_results_df = fold_results_df[
        result_columns
    ]


    # Save master fold results
    fold_results_df.to_csv(

        os.path.join(
            SAVE_ROOT,
            "PROPOSED_MODEL_5FOLD_RESULTS.csv"
        ),

        index=False
    )


    # ========================================================
    # CALCULATE MEAN / SD / SE
    # ========================================================

    metric_columns = [

        "accuracy",

        "micro_accuracy",

        "macro_class_accuracy",

        "macro_precision",

        "micro_precision",

        "macro_recall",

        "micro_recall",

        "macro_f1",

        "micro_f1",

        "macro_auc",

        "micro_auc"
    ]


    summary_rows = []


    for metric in metric_columns:

        values = (

            fold_results_df[
                metric
            ]
            .astype(float)
        )


        mean_value = values.mean()


        standard_deviation = values.std(

            ddof=1
        )


        standard_error = (

            standard_deviation
            / np.sqrt(N_SPLITS)
        )


        summary_rows.append({

            "metric":
                metric,

            "mean":
                mean_value,

            "std":
                standard_deviation,

            "standard_error":
                standard_error,

            "mean_minus_sd":
                mean_value - standard_deviation,

            "mean_plus_sd":
                mean_value + standard_deviation
        })


    summary_df = pd.DataFrame(
        summary_rows
    )


    # Save summary
    summary_df.to_csv(

        os.path.join(
            SAVE_ROOT,
            "PROPOSED_MODEL_5FOLD_SUMMARY.csv"
        ),

        index=False
    )


    # ========================================================
    # SAVE MEAN ± SD TABLE
    # ========================================================

    display_rows = []


    for _, row in summary_df.iterrows():

        display_rows.append({

            "Metric":
                row["metric"],

            "Mean":
                row["mean"],

            "SD":
                row["std"],

            "Mean ± SD":
                (
                    f"{row['mean']:.6f} ± "
                    f"{row['std']:.6f}"
                )
        })


    display_df = pd.DataFrame(
        display_rows
    )


    display_df.to_csv(

        os.path.join(
            SAVE_ROOT,
            "PROPOSED_MODEL_MEAN_SD_TABLE.csv"
        ),

        index=False
    )


    # ========================================================
    # SAVE HUMAN-READABLE FINAL SUMMARY
    # ========================================================

    summary_txt_path = os.path.join(

        SAVE_ROOT,

        "PROPOSED_MODEL_FINAL_5FOLD_SUMMARY.txt"
    )


    with open(

        summary_txt_path,

        "w"

    ) as f:

        f.write(
            "PROPOSED MODEL - "
            "5-FOLD STRATIFIED CROSS-VALIDATION\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )


        f.write(
            f"Dataset: {DATA_DIR}\n"
        )


        f.write(
            f"Total images: {len(dataset)}\n"
        )


        f.write(
            f"Classes: {dataset.classes}\n"
        )


        f.write(
            f"Number of folds: {N_SPLITS}\n"
        )


        f.write(
            f"Seed: {SEED}\n"
        )


        f.write(
            f"Epochs: {NUM_EPOCHS}\n"
        )


        f.write(
            f"Batch size: {BATCH_SIZE}\n"
        )


        f.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )


        f.write(
            f"Device: {DEVICE}\n\n"
        )


        f.write(
            "FOLD-WISE RESULTS\n"
        )


        f.write(
            "-" * 80
            + "\n"
        )


        f.write(
            fold_results_df.to_string(
                index=False
            )
        )


        f.write(
            "\n\n"
        )


        f.write(
            "MEAN / SD / SE\n"
        )


        f.write(
            "-" * 80
            + "\n"
        )


        for _, row in summary_df.iterrows():

            f.write(

                f"{row['metric']}\n"
            )

            f.write(

                f"  Mean = "
                f"{row['mean']:.8f}\n"
            )

            f.write(

                f"  SD   = "
                f"{row['std']:.8f}\n"
            )

            f.write(

                f"  SE   = "
                f"{row['standard_error']:.8f}\n\n"
            )


    # ========================================================
    # SAVE FINAL JSON RECORD
    # ========================================================

    elapsed_seconds = (

        time.time()
        - overall_start
    )


    final_record = {

        "experiment":
            "Proposed Model 5-Fold Stratified CV",

        "architecture":
            (
                "CNN Residual Blocks -> "
                "BiLSTM -> Additive Attention -> FC"
            ),

        "dataset":
            DATA_DIR,

        "total_images":
            int(len(dataset)),

        "number_of_folds":
            N_SPLITS,

        "seed":
            SEED,

        "epochs":
            NUM_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "num_workers":
            NUM_WORKERS,

        "device":
            str(DEVICE),
        "checkpoint_format":
            "model_history_rng_only_v2",
        "optimizer_state_saved":
            False,

        "runtime_minutes":
            elapsed_seconds / 60.0,

        "fixed_fold_file":
            FIXED_FOLDS_FILE,

        "fold_results_file":
            os.path.join(
                SAVE_ROOT,
                "PROPOSED_MODEL_5FOLD_RESULTS.csv"
            ),

        "summary_file":
            os.path.join(
                SAVE_ROOT,
                "PROPOSED_MODEL_5FOLD_SUMMARY.csv"
            ),

        "mean_sd_file":
            os.path.join(
                SAVE_ROOT,
                "PROPOSED_MODEL_MEAN_SD_TABLE.csv"
            )
    }


    with open(

        os.path.join(
            SAVE_ROOT,
            "FINAL_EXPERIMENT_RECORD.json"
        ),

        "w"

    ) as f:

        json.dump(

            final_record,

            f,

            indent=4
        )


    # ========================================================
    # FINAL TERMINAL OUTPUT
    # ========================================================

    print("\n\n")

    print("=" * 80)

    print(
        "PROPOSED MODEL - "
        "FINAL 5-FOLD RESULTS"
    )

    print("=" * 80)


    print("\nFOLD-WISE RESULTS:\n")


    print(

        fold_results_df.to_string(
            index=False
        )
    )


    print("\n\nMEAN ± SD:\n")


    for _, row in summary_df.iterrows():

        print(

            f"{row['metric']:25s}: "
            f"{row['mean']:.6f} ± "
            f"{row['std']:.6f}"
        )


    print("\n")

    print("=" * 80)

    print("EXPERIMENT COMPLETE")

    print("=" * 80)


    print(
        f"\nAll results saved in:\n"
        f"{SAVE_ROOT}"
    )


    print(
        f"\nFixed folds saved in:\n"
        f"{FIXED_FOLDS_FILE}"
    )


    print(
        f"\nTotal runtime: "
        f"{elapsed_seconds / 60:.2f} minutes"
    )


    print("\nIMPORTANT:")

    print(
        "The FIXED_5_FOLD_SPLITS.npz file "
        "must be reused for every baseline model."
    )

    print(
        "Do NOT generate new folds for "
        "VGG16, ResNet50, DenseNet121, "
        "or EfficientNet-B3."
    )


# ============================================================
# 21. WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()