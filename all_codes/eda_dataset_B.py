# ============================================================
# EXPLORATORY DATA ANALYSIS (EDA) - DATASET B
# Brain MRI Dataset
#
# This script performs:
# 1. Dataset structure analysis
# 2. Class distribution
# 3. Train vs Test distribution
# 4. Image dimension analysis
# 5. Pixel intensity statistics
# 6. Brightness analysis
# 7. Contrast analysis
# 8. Representative images per class
# 9. Class-wise average MRI images
# 10. Image-level statistics CSV
# 11. Dataset summary CSV
#
# NO TRAINING
# NO AUGMENTATION
# NO DATA MODIFICATION
# ============================================================

import os
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

TRAIN_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B\preprocessed_train_B"
)

TEST_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B\preprocessed_test_B"
)

SAVE_DIR = (
    r"C:\user\juli\thesis\original_file\othr folders"
    r"\Dataset_B_clean\Dataset_B\EDA_Dataset_B"
)

os.makedirs(SAVE_DIR, exist_ok=True)


CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary"
}

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# Supported image formats

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# 2. VERIFY DATASET STRUCTURE
# ============================================================

def verify_dataset_structure():

    print("\n" + "=" * 70)
    print("VERIFYING DATASET B STRUCTURE")
    print("=" * 70)

    for split_name, directory in [
        ("Train", TRAIN_DIR),
        ("Test", TEST_DIR)
    ]:

        if not os.path.isdir(directory):

            raise FileNotFoundError(
                f"{split_name} directory not found:\n"
                f"{directory}"
            )

        found_classes = sorted([
            folder
            for folder in os.listdir(directory)
            if os.path.isdir(
                os.path.join(directory, folder)
            )
        ])

        print(
            f"\n{split_name} classes:"
        )

        print(found_classes)

        if found_classes != CLASSES:

            raise RuntimeError(
                f"\nClass folder mismatch in {split_name}.\n"
                f"Expected: {CLASSES}\n"
                f"Found:    {found_classes}"
            )

    print(
        "\nDataset structure verification: PASS"
    )


# ============================================================
# 3. COLLECT IMAGE INFORMATION
# ============================================================

def collect_image_information():

    rows = []

    unreadable_images = []

    for split_name, directory in [
        ("Train", TRAIN_DIR),
        ("Test", TEST_DIR)
    ]:

        print(
            f"\nScanning {split_name} images..."
        )

        for class_name in CLASSES:

            class_directory = os.path.join(
                directory,
                class_name
            )

            filenames = sorted(
                os.listdir(class_directory)
            )

            for filename in filenames:

                if not filename.lower().endswith(
                    VALID_EXTENSIONS
                ):
                    continue

                image_path = os.path.join(
                    class_directory,
                    filename
                )

                image = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:

                    unreadable_images.append(
                        image_path
                    )

                    continue

                height, width = image.shape

                # --------------------------------------------
                # Basic pixel statistics
                # --------------------------------------------

                mean_intensity = float(
                    np.mean(image)
                )

                std_intensity = float(
                    np.std(image)
                )

                min_intensity = int(
                    np.min(image)
                )

                max_intensity = int(
                    np.max(image)
                )

                median_intensity = float(
                    np.median(image)
                )

                # --------------------------------------------
                # Percentiles
                # --------------------------------------------

                p25 = float(
                    np.percentile(image, 25)
                )

                p75 = float(
                    np.percentile(image, 75)
                )

                # --------------------------------------------
                # Contrast
                #
                # Standard deviation of grayscale intensities
                # is used as a simple image-level contrast
                # descriptor.
                # --------------------------------------------

                contrast = std_intensity

                # --------------------------------------------
                # Laplacian variance
                #
                # Useful descriptive measure of edge/sharpness
                # content.
                # --------------------------------------------

                laplacian_variance = float(
                    cv2.Laplacian(
                        image,
                        cv2.CV_64F
                    ).var()
                )

                rows.append({

                    "split":
                        split_name,

                    "class":
                        class_name,

                    "display_class":
                        DISPLAY_NAMES[class_name],

                    "filename":
                        filename,

                    "path":
                        image_path,

                    "width":
                        width,

                    "height":
                        height,

                    "mean_intensity":
                        mean_intensity,

                    "median_intensity":
                        median_intensity,

                    "std_intensity":
                        std_intensity,

                    "contrast":
                        contrast,

                    "min_intensity":
                        min_intensity,

                    "max_intensity":
                        max_intensity,

                    "p25_intensity":
                        p25,

                    "p75_intensity":
                        p75,

                    "laplacian_variance":
                        laplacian_variance
                })

    df = pd.DataFrame(rows)

    return df, unreadable_images


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(df):

    counts = (
        df.groupby(
            ["split", "display_class"]
        )
        .size()
        .unstack(fill_value=0)
    )

    # Force desired order

    desired_names = [
        DISPLAY_NAMES[c]
        for c in CLASSES
    ]

    counts = counts.reindex(
        columns=desired_names
    )

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    print(counts)

    counts.to_csv(
        os.path.join(
            SAVE_DIR,
            "class_distribution.csv"
        )
    )

    # --------------------------------------------------------
    # Train/Test grouped bar plot
    # --------------------------------------------------------

    ax = counts.T.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        "Class Distribution of Dataset B"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Number of Images"
    )

    plt.xticks(
        rotation=0
    )

    plt.legend(
        title="Dataset Split"
    )

    # Add values above bars

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%d",
            padding=3
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "01_class_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. TOTAL CLASS DISTRIBUTION
# ============================================================

def plot_total_class_distribution(df):

    counts = (
        df["display_class"]
        .value_counts()
        .reindex(
            [
                DISPLAY_NAMES[c]
                for c in CLASSES
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        counts.index,
        counts.values
    )

    ax.set_title(
        "Overall Class Distribution of Dataset B"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Number of Images"
    )

    ax.bar_label(
        bars,
        fmt="%d",
        padding=3
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "02_overall_class_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 6. IMAGE DIMENSION ANALYSIS
# ============================================================

def analyze_dimensions(df):

    dimension_counts = (
        df.groupby(
            ["width", "height"]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    dimension_counts.to_csv(
        os.path.join(
            SAVE_DIR,
            "image_dimensions.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("IMAGE DIMENSIONS")
    print("=" * 70)

    print(
        dimension_counts.to_string(
            index=False
        )
    )


# ============================================================
# 7. INTENSITY DISTRIBUTION BY CLASS
# ============================================================

def plot_mean_intensity_distribution(df):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    data = []

    labels = []

    for class_name in CLASSES:

        values = df.loc[
            df["class"] == class_name,
            "mean_intensity"
        ].values

        data.append(values)

        labels.append(
            DISPLAY_NAMES[class_name]
        )

    ax.boxplot(
        data,
        tick_labels=labels
    )

    ax.set_title(
        "Mean Pixel Intensity Distribution by Class"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Mean Grayscale Intensity"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "03_mean_intensity_by_class.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. CONTRAST DISTRIBUTION
# ============================================================

def plot_contrast_distribution(df):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    data = []

    labels = []

    for class_name in CLASSES:

        values = df.loc[
            df["class"] == class_name,
            "contrast"
        ].values

        data.append(values)

        labels.append(
            DISPLAY_NAMES[class_name]
        )

    ax.boxplot(
        data,
        tick_labels=labels
    )

    ax.set_title(
        "Image Contrast Distribution by Class"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Intensity Standard Deviation"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "04_contrast_by_class.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 9. LAPLACIAN VARIANCE / SHARPNESS DISTRIBUTION
# ============================================================

def plot_laplacian_distribution(df):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    data = []

    labels = []

    for class_name in CLASSES:

        values = df.loc[
            df["class"] == class_name,
            "laplacian_variance"
        ].values

        data.append(values)

        labels.append(
            DISPLAY_NAMES[class_name]
        )

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False
    )

    ax.set_title(
        "Laplacian Variance Distribution by Class"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Laplacian Variance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "05_laplacian_variance_by_class.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. GLOBAL PIXEL INTENSITY HISTOGRAM
# ============================================================

def plot_pixel_histograms(df):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for class_name in CLASSES:

        class_df = df[
            df["class"] == class_name
        ]

        # Aggregate histogram
        histogram_sum = np.zeros(
            256,
            dtype=np.float64
        )

        image_count = 0

        for path in class_df["path"]:

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            histogram = cv2.calcHist(
                [image],
                [0],
                None,
                [256],
                [0, 256]
            ).flatten()

            # Normalize each image histogram
            # before combining, so each image contributes
            # equally.

            histogram = histogram / (
                histogram.sum() + 1e-12
            )

            histogram_sum += histogram

            image_count += 1

        if image_count > 0:

            average_histogram = (
                histogram_sum /
                image_count
            )

            ax.plot(
                np.arange(256),
                average_histogram,
                label=DISPLAY_NAMES[class_name]
            )

    ax.set_title(
        "Average Pixel Intensity Distribution by Class"
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Average Relative Frequency"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "06_pixel_intensity_histogram.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. REPRESENTATIVE IMAGES
# ============================================================

def plot_representative_images(df):

    # Use TRAIN images only for representative visualization.

    train_df = df[
        df["split"] == "Train"
    ]

    number_per_class = 5

    fig, axes = plt.subplots(
        len(CLASSES),
        number_per_class,
        figsize=(15, 10)
    )

    for row, class_name in enumerate(
        CLASSES
    ):

        class_df = train_df[
            train_df["class"] == class_name
        ]

        # Deterministic random sampling

        sample_number = min(
            number_per_class,
            len(class_df)
        )

        selected = class_df.sample(
            n=sample_number,
            random_state=SEED
        )

        selected_paths = list(
            selected["path"]
        )

        for col in range(
            number_per_class
        ):

            ax = axes[row, col]

            if col < len(selected_paths):

                image = cv2.imread(
                    selected_paths[col],
                    cv2.IMREAD_GRAYSCALE
                )

                ax.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=255
                )

            ax.axis("off")

            if col == 0:

                ax.set_title(
                    DISPLAY_NAMES[class_name],
                    loc="left"
                )

    fig.suptitle(
        "Representative Preprocessed Images from Dataset B",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "07_representative_images.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. CLASS-WISE AVERAGE IMAGES
# ============================================================

def create_average_images(df):

    fig, axes = plt.subplots(
        1,
        len(CLASSES),
        figsize=(14, 4)
    )

    for i, class_name in enumerate(
        CLASSES
    ):

        class_df = df[
            df["class"] == class_name
        ]

        image_sum = None

        count = 0

        for path in class_df["path"]:

            image = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = image.astype(
                np.float64
            )

            if image_sum is None:

                image_sum = np.zeros_like(
                    image,
                    dtype=np.float64
                )

            image_sum += image

            count += 1

        if count == 0:
            continue

        average_image = (
            image_sum /
            count
        )

        axes[i].imshow(
            average_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[i].set_title(
            DISPLAY_NAMES[class_name]
        )

        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "Class-wise Average Images of Dataset B",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_DIR,
            "08_class_average_images.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 13. CLASS-WISE STATISTICAL SUMMARY
# ============================================================

def create_class_statistics(df):

    summary = (
        df.groupby(
            ["split", "display_class"]
        )
        .agg(

            image_count=(
                "filename",
                "count"
            ),

            mean_pixel_intensity=(
                "mean_intensity",
                "mean"
            ),

            sd_pixel_intensity=(
                "mean_intensity",
                "std"
            ),

            median_pixel_intensity=(
                "mean_intensity",
                "median"
            ),

            mean_contrast=(
                "contrast",
                "mean"
            ),

            sd_contrast=(
                "contrast",
                "std"
            ),

            mean_laplacian_variance=(
                "laplacian_variance",
                "mean"
            ),

            mean_width=(
                "width",
                "mean"
            ),

            mean_height=(
                "height",
                "mean"
            )

        )
        .reset_index()
    )

    # Round for readability

    numeric_columns = summary.select_dtypes(
        include=[np.number]
    ).columns

    summary[numeric_columns] = (
        summary[numeric_columns]
        .round(3)
    )

    summary.to_csv(
        os.path.join(
            SAVE_DIR,
            "classwise_statistical_summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("CLASS-WISE STATISTICAL SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# 14. DATASET SUMMARY
# ============================================================

def create_dataset_summary(df):

    train_count = int(
        (df["split"] == "Train").sum()
    )

    test_count = int(
        (df["split"] == "Test").sum()
    )

    total_count = len(df)

    summary_rows = [

        {
            "measure": "Training images",
            "value": train_count
        },

        {
            "measure": "Test images",
            "value": test_count
        },

        {
            "measure": "Total images",
            "value": total_count
        },

        {
            "measure": "Number of classes",
            "value": len(CLASSES)
        },

        {
            "measure": "Minimum width",
            "value": int(df["width"].min())
        },

        {
            "measure": "Maximum width",
            "value": int(df["width"].max())
        },

        {
            "measure": "Minimum height",
            "value": int(df["height"].min())
        },

        {
            "measure": "Maximum height",
            "value": int(df["height"].max())
        },

        {
            "measure": "Overall mean intensity",
            "value": round(
                df["mean_intensity"].mean(),
                3
            )
        },

        {
            "measure": "Overall mean contrast",
            "value": round(
                df["contrast"].mean(),
                3
            )
        }

    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            SAVE_DIR,
            "dataset_B_summary.csv"
        ),
        index=False
    )

    print("\n" + "=" * 70)
    print("DATASET B SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )


# ============================================================
# 15. TRAIN / TEST PERCENTAGES
# ============================================================

def calculate_split_percentages(df):

    total = len(df)

    train_number = (
        df["split"] == "Train"
    ).sum()

    test_number = (
        df["split"] == "Test"
    ).sum()

    train_percentage = (
        train_number /
        total *
        100
    )

    test_percentage = (
        test_number /
        total *
        100
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Training: "
        f"{train_number:,} "
        f"({train_percentage:.2f}%)"
    )

    print(
        f"Testing : "
        f"{test_number:,} "
        f"({test_percentage:.2f}%)"
    )


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS - DATASET B")
    print("=" * 70)

    print("\nTraining directory:")
    print(TRAIN_DIR)

    print("\nTesting directory:")
    print(TEST_DIR)

    print("\nEDA output directory:")
    print(SAVE_DIR)


    # --------------------------------------------------------
    # Verify folders
    # --------------------------------------------------------

    verify_dataset_structure()


    # --------------------------------------------------------
    # Collect image statistics
    # --------------------------------------------------------

    df, unreadable_images = (
        collect_image_information()
    )


    if len(df) == 0:

        raise RuntimeError(
            "No readable images were found."
        )


    # --------------------------------------------------------
    # Save image-level data
    # --------------------------------------------------------

    df.to_csv(
        os.path.join(
            SAVE_DIR,
            "dataset_B_image_level_statistics.csv"
        ),
        index=False
    )


    # --------------------------------------------------------
    # Report unreadable images
    # --------------------------------------------------------

    print(
        f"\nReadable images: {len(df):,}"
    )

    print(
        f"Unreadable images: "
        f"{len(unreadable_images):,}"
    )


    if unreadable_images:

        pd.DataFrame(
            {
                "unreadable_image":
                    unreadable_images
            }
        ).to_csv(
            os.path.join(
                SAVE_DIR,
                "unreadable_images.csv"
            ),
            index=False
        )


    # --------------------------------------------------------
    # Run analyses
    # --------------------------------------------------------

    plot_class_distribution(df)

    plot_total_class_distribution(df)

    analyze_dimensions(df)

    plot_mean_intensity_distribution(df)

    plot_contrast_distribution(df)

    plot_laplacian_distribution(df)

    plot_pixel_histograms(df)

    plot_representative_images(df)

    create_average_images(df)

    create_class_statistics(df)

    create_dataset_summary(df)

    calculate_split_percentages(df)


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(
        "\nAll results have been saved to:"
    )

    print(SAVE_DIR)

    print("\nGenerated outputs:")

    print(
        "01_class_distribution.png"
    )

    print(
        "02_overall_class_distribution.png"
    )

    print(
        "03_mean_intensity_by_class.png"
    )

    print(
        "04_contrast_by_class.png"
    )

    print(
        "05_laplacian_variance_by_class.png"
    )

    print(
        "06_pixel_intensity_histogram.png"
    )

    print(
        "07_representative_images.png"
    )

    print(
        "08_class_average_images.png"
    )

    print(
        "class_distribution.csv"
    )

    print(
        "image_dimensions.csv"
    )

    print(
        "classwise_statistical_summary.csv"
    )

    print(
        "dataset_B_image_level_statistics.csv"
    )

    print(
        "dataset_B_summary.csv"
    )