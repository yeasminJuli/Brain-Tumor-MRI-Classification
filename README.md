# Brain Tumor MRI Classification

A research companion for four-class brain MRI classification using a residual CNN, BiLSTM and additive attention.

## Start here

- **Explore the project:** [Project guide](docs/PROJECT_GUIDE.md)
- **Understand the experiment:** [Methods and limitations](docs/METHODS.md)
- **Inspect evidence:** [Results](results/README.md) · [Figures](figures/README.md)

## Clear folder structure

```text
src/          Reusable model, preprocessing and augmentation
scripts/      Clearly named workflows grouped by task and dataset
configs/      Example paths and class mappings
results/      Selected single-run and cross-dataset tables
figures/      Corresponding confusion matrices and ROC curves
docs/         Project guide, methods and verification notes
```

See the [script index](scripts/README.md) for preprocessing, training, evaluation and explanation workflows. The reusable [model definition](src/models/proposed.py) is kept separately from experiment scripts.

## Run data preparation

Install a suitable PyTorch build and the dependencies in `requirements.txt` using Python 3.11.

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py --help
python scripts/prepare_data.py preprocess --input PATH_TO_RAW_CLASSES --output PATH_TO_NEW_PROCESSED_FOLDER
python scripts/prepare_data.py augment --input PATH_TO_TRAINING_CLASSES --output PATH_TO_NEW_AUGMENTED_FOLDER --copies 4
```

These commands preserve class-folder names. For independent validation, split originals before augmentation. The historical C split-after-augmentation experiment is explicitly described in the methods.

## What is shared

This public view contains selected evidence and reference code. Full datasets, checkpoints, fold-level outputs, drafts and intermediate files are retained in the **Brain Tumor Research Archive** locally. Historical training/evaluation scripts require checking their data and checkpoint paths; checkpoints are not bundled. No clinical metadata or exact historical environment has been invented.
