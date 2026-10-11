# Brain Tumor MRI Classification

A research companion for four-class brain MRI classification using a residual CNN, BiLSTM and additive attention.

## Start here

- **Review the research:** [Reviewer guide](docs/REVIEWER_GUIDE.md)
- **Understand the experiment:** [Methods and limitations](docs/METHODS.md)
- **Inspect evidence:** [Results](results/README.md) · [Figures](figures/README.md)

## Clear folder structure

```text
src/          Reusable model, preprocessing and augmentation
scripts/      Data-preparation command and task-grouped reference scripts
configs/      Example paths and class mappings
results/      Selected single-run and cross-dataset tables
figures/      Corresponding confusion matrices and ROC curves
docs/         Reviewer guide and methods
```

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

This public view contains selected evidence and reference code. Full datasets, checkpoints, fold-level outputs, drafts and intermediate files are retained in the **Brain Tumor Research Archive** locally. Reference training/evaluation scripts may need machine-path and import updates; checkpoints are not bundled. No clinical metadata or exact historical environment has been invented.

The earlier detailed public layout remains in Git history. A local folder is not a GitHub private repository. This cleanup does not erase previously published data.
