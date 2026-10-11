# Dataset A methods

The model combines a residual CNN, a bidirectional LSTM and additive attention. Output indices are glioma (0), meningioma (1), no tumor (2) and pituitary (3).

Preprocessing uses grayscale conversion, resizing to 256 x 256, CLAHE with clip limit 1.0 and an 8 x 8 grid, conditional 3 x 3 Gaussian smoothing when post-CLAHE Laplacian variance is below 60, and Otsu/morphological processing for contour-based cropping. Laplacian variance is not a direct noise detector; cropping is not tumor segmentation.

## Sequence construction

The original model directly reshapes the contiguous CNN tensor from (B, 256, 8, 8) to (B, 8, 2048), using `x.view(b, h, c*w)`. Although historically called row-wise, this operation does not explicitly gather spatial rows from an NCHW tensor. Its sequence steps should not be interpreted as anatomical image rows.

The column-wise variant explicitly permutes the tensor to gather columns. The zigzag variant gathers spatial rows and reverses the width direction on alternating rows. The training script retains the name `alternative` internally for this zigzag variant.

## Result interpretation

The public results focus on the original Dataset A evaluation and separately saved reshaping variants. They do not establish external generalization, clinical validity or patient-level independence. For independent validation in new experiments, split original images before augmentation. See the result documentation for the limits of the saved ablation comparison.

Historical scripts require checking data and checkpoint paths before use. Trained weights and a locked historical environment are not bundled. Dataset licensing and download provenance must be confirmed before redistribution.
