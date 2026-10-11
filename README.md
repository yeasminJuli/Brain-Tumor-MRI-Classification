# Brain Tumor MRI Classification

Research code for four-class MRI classification with a residual CNN, BiLSTM and additive attention, alongside baseline, ablation, cross-validation and cross-dataset experiments.

## Start here

- [Script index](docs/script_index.md): find preprocessing, training, evaluation and explainability scripts.
- [Experiment notes](docs/experiment_notes.md): class mappings, missing metadata and differences between protocols.
- `results/`: saved numerical evidence. `figures/`: saved visual results.

## Project structure

```text
scripts/       Task-grouped historical scripts and a portable data-preparation CLI
src/           Reusable proposed model, preprocessing and augmentation functions
configs/       Example dataset paths and explicit class mappings
results/       Numerical results separated by experiment
figures/       Existing plots and selected explanation figures
docs/          Script index and experiment notes
data/          Full datasets locally; instructions only in GitHub
checkpoints/   Full weights locally; instructions only in GitHub
```

## Quick start

Use Python 3.11 and a suitable PyTorch installation. Dependency versions are not a reconstructed historical environment.

```powershell
python -m pip install -r requirements.txt
python scripts/prepare_data.py --help
python scripts/prepare_data.py preprocess --input "PATH_TO_RAW_CLASS_FOLDERS" --output "PATH_TO_NEW_PROCESSED_FOLDER"
python scripts/prepare_data.py augment --input "PATH_TO_TRAINING_CLASS_FOLDERS" --output "PATH_TO_NEW_AUGMENTED_FOLDER" --copies 4
```

The commands preserve class folder names and write lossless PNGs. Do not preprocess already processed images again. Partition originals before augmentation when independent validation is required.

## Training and evaluation

The historical scripts are organized by task under `scripts/historical/`; filenames and source contents are preserved. They are research records, not a fully migrated command-line application. Review their path settings and imports before running them. The reusable model is in `src/models/proposed.py`. No training was rerun to create this package.

The original four-class output order is glioma, meningioma, no tumor, pituitary. Dataset C folder names require the explicit mapping in `configs/dataset_paths.example.json`. Do not assume alphabetical folder order matches checkpoint outputs.

## Data and results

This organization task does not invent clinical metadata or claim source/patient independence. Original and cleaned experiments remain separate. See the notes before using a result in the manuscript. Historical scores are retained even where the protocol has limitations.

No license is assigned to the author's code by this packaging task. Dataset licenses and figure permissions remain those of their respective sources.

## GitHub contents

This is the compact package. Full datasets, weights, environments and per-image logs are excluded. Selected PNG figures are included. `docs/source_index.csv` records the source of copied artifacts.

## Existing repository files

The original all_codes, othr folders and ref directories are retained for continuity. Start with scripts, src, results, figures and docs for the organized version. No original research files were deleted during this update.

