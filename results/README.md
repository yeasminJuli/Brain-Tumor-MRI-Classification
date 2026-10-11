# Dataset A results

| Experiment | Test images | Accuracy |
|---|---:|---:|
| Original model | 1,311 | 99.47% |
| Column-wise variant | 1,311 | 99.24% |
| Zigzag variant | 1,311 | 99.31% |

The original score comes from [saved overall metrics](dataset_a/overall_metrics.csv); [class metrics](dataset_a/class_metrics.csv) are provided separately. The variant scores are the diagonal sum divided by the total of their saved confusion matrices: [column-wise](reshaping_ablation/column_wise/confusion_matrix.csv), 1,301/1,311, and [zigzag](reshaping_ablation/zigzag/confusion_matrix.csv), 1,302/1,311.

These are separately saved single-run results, not cross-validation averages or a newly rerun matched comparison. The older combined comparison table contains an inconsistent original-model result and is not used here. The variant reports retain the original saved values. Equal sample counts alone do not verify identical image ordering, checkpoint selection or training conditions across runs.

“Row-wise” is the historical name for the original model. Its actual implementation uses a direct tensor reshape, as explained in [Methods](../docs/METHODS.md).
