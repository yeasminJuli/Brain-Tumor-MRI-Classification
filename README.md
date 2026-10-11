# Brain Tumor MRI Classification

This project classifies brain MRI images into four classes: glioma, meningioma, no tumor and pituitary tumor. The model combines a residual CNN for feature extraction, a bidirectional LSTM for sequence processing and an attention layer that combines the sequence outputs before classification.

The original model achieved **99.47% accuracy on 1,311 Dataset A test images**. This repository focuses on that experiment and the column-wise and zigzag reshaping variants.

## Results

| Experiment | Test accuracy |
|---|---:|
| Original model | 99.47% |
| Column-wise reshaping | 99.24% |
| Zigzag reshaping | 99.31% |

These are saved single-run results. The [results page](results/README.md) explains their sources and the limits of comparing them. Confusion matrices and ROC curves are in [figures](figures/README.md).

## Finding the code

| Folder | Contents |
|---|---|
| [src/models](src/models) | Proposed model architecture |
| [scripts/preprocessing](scripts/preprocessing) | Image preprocessing |
| [scripts/augmentation](scripts/augmentation) | Training-image augmentation |
| [scripts/training](scripts/training) | Model training |
| [scripts/evaluation](scripts/evaluation) | Model evaluation |
| [scripts/explainability](scripts/explainability) | Attention weights, Grad-CAM++ and LIME |
| [scripts/ablation](scripts/ablation) | Column-wise and zigzag reshaping experiments |
| [results](results) | Saved metrics and classification reports |
| [figures](figures) | Saved evaluation figures |

The [script index](scripts/README.md) lists the individual files. [Methods](docs/METHODS.md) describes preprocessing, sequence construction and how to interpret the results.

## Getting started

Use Python 3.11 and install a PyTorch version suitable for your machine, followed by the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

To preprocess images or create augmented training images:

```bash
python scripts/prepare_data.py preprocess --input PATH_TO_RAW_CLASSES --output PATH_TO_PROCESSED_IMAGES
python scripts/prepare_data.py augment --input PATH_TO_TRAINING_CLASSES --output PATH_TO_AUGMENTED_IMAGES --copies 4
```

Arrange input images in class folders. These commands preserve the folder names. When preparing a new experiment, split the original images before augmentation so that related copies do not enter both training and validation.

The dataset-specific scripts retain the original experiment settings. Check their input, output and checkpoint paths before running them. Datasets and trained weights are not included, so the repository alone is not enough to reproduce every result.
