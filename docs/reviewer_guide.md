# Reviewer guide

This repository accompanies research on four-class brain MRI classification with a residual CNN, BiLSTM and additive attention. It contains source code and saved experiment outputs. It is not a claim of independently reproduced clinical performance.

## Suggested reading order

1. Inspect the proposed architecture in [src/models/proposed.py](../src/models/proposed.py).
2. Read [experiment_notes.md](experiment_notes.md) before comparing experimental conditions.
3. Follow the result links below for the experiment of interest.
4. Consult [script_index.md](script_index.md) for the corresponding historical workflows.
5. Read [reproducibility.md](reproducibility.md) for what is and is not currently reproducible from the public package.

## Main evidence locations

| Topic | Location | Interpretation |
|---|---|---|
| Original A proposed-model evaluation | [Open](../results/dataset_a/othr%20folders/evaluation_figures_4070Sss) | Saved original evaluation; folder name alone does not verify hardware. |
| Original A training | [Open](../results/dataset_a/othr%20folders/training_outputs_proposed_model_orgnl) | Saved training history and model summary. |
| Dataset C and cross-dataset summary | [Open](../results/dataset_c/othr%20folders/Brain%20Cancer%20MRI%20Dataset_C/evaluation_summary.csv) | C-to-C, A-to-C and C-to-A are separate test conditions. |
| A model on C test images | [Open](../results/cross_dataset/A_to_C) | Cross-dataset evaluation; independence requires a separate overlap audit. |
| C model on A test images | [Open](../results/cross_dataset/C_to_A) | Reverse-direction cross-dataset evaluation. |
| A/B cross-dataset evaluations | [Open](../results/cross_dataset) | A-to-B and B-to-A folders retain their directional labels. |
| Five-fold experiments | [Open](../results/cross_validation) | Do not equate fold summaries with single-run test scores. |
| Cleaned experiments | [Open](../results/cleaned_experiments) | Separate data/protocol condition; not interchangeable with original results. |
| Reshaping ablation | [Open](../results/ablation) | Sequence construction and ordering experiments. |
| Selected attention figure | [Open](../figures/dataset_a/othr%20folders/Attention%20Waves%20-%20Manuscript%20Figures%20and%20Explanations/01_Correctly_Classified_Attention_Weights.png) | Weights over sequence steps, not anatomical tumor boundaries. |
| Selected Grad-CAM++ figure | [Open](../figures/dataset_a/othr%20folders/GradCAMpp_Final_Thesis_Figure.png) | Qualitative visualization, not a clinically validated segmentation. |
| Preprocessing | [Open](../src/preprocessing/pipeline.py) | Extracted original processing function. |
| Augmentation | [Open](../src/augmentation/transforms.py) | Extracted original augmentation function. |

## Reading the numbers

Confusion-matrix rows are true labels and columns are predicted labels in the provided C evaluation workflow. Most new metrics CSV files store scores from 0 to 1; some older reports use percentages. Read column names and generating scripts before combining tables. ROC AUC and classification accuracy measure different properties and must not be substituted for one another.

The source index records copies and their hashes. It does not prove that a particular script version produced a particular historical result. Final manuscript table/figure numbering has not been reconstructed; the author should map these links to the submitted revision.

## Scope of the supplied material

The main folders for this organized version are `src`, `scripts`, `results`, `figures`, `configs` and `docs`. The older `all_codes`, `othr folders` and `ref` directories remain for continuity. Missing patient-level metadata, dataset access conditions and the C split-after-augmentation limitation are documented rather than inferred away.
