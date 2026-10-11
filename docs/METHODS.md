# Methods and interpretation

The proposed model combines a residual CNN, a bidirectional LSTM and additive attention. Its four output indices are glioma (0), meningioma (1), no tumor (2) and pituitary (3). Dataset C uses Healthy for index 2 and Pituitary Macroadenoma for index 3.

Preprocessing uses grayscale conversion, resizing to 256 x 256, CLAHE with clip limit 1.0 and an 8 x 8 grid, conditional 3 x 3 Gaussian smoothing when post-CLAHE Laplacian variance is below 60, and Otsu/morphological processing for contour-based cropping. Laplacian variance is not a direct noise detector, and this cropping is not tumor segmentation.

The preserved C experiment augmented 909 development images into 4,545 files, then split those files 80/20. Augmented relatives of 617 originals appeared in both training and validation. Its validation score is therefore not independent at original-image level. The 228 C test images were separate from that development partition.

Cross-dataset testing does not establish patient-level or image-level independence across source datasets. This archive lacks complete patient/scanner metadata, and dataset source/version details still require author confirmation. Original, cleaned and five-fold experiments are different conditions.

The private/local archive retains detailed fold outputs. A fold confusion matrix can contain thousands of entries when its evaluated partition includes augmented files or pooled samples; interpret the counts against that run's script and split, rather than treating them as unique patients. This statement is not a verification of every historical fold protocol.

Historical reference scripts preserve original research workflows and may require path/import adjustments. A filename is not proof of which script version generated a saved result. No exact historical dependency lockfile or trained weights are distributed here. Code/data licensing and dataset download provenance require author confirmation before a final journal reproducibility claim.
