# Experiment notes

This package organizes existing evidence; it does not rerun experiments or reconstruct missing clinical metadata.

- A is the original four-class MRI experiment. Verify the exact downloaded dataset version before citing a source: existing local descriptions disagree. The old README's dataset link is not treated as authoritative.
- B has separate training and evaluation scripts. Source descriptions and local folder counts must be checked together; do not infer patient identifiers.
- C uses the labels Glioma, Meningioma, Healthy and Pituitary Macroadenoma. The mapping to the original output indices is 0, 1, 2 and 3 respectively, not C's alphabetical folder order.
- The latest C experiment augmented the full development partition and then split the augmented files 80/20. Related images occur in training and validation. Its validation accuracy must not be described as independent image-level performance. The 228-image C test partition remained separate in that workflow.
- A-to-C and C-to-A are cross-dataset evaluations, not five-fold cross-validation. Cross-dataset image/patient independence has not been established by this organization task.
- Original, cleaned, cross-validation and ablation results are separate experimental conditions. Do not combine their scores into one result.
- `all_codes/train_model_withCustomized_cnn.py` currently contains LIME code despite its name. It is categorized as LIME; the reusable model is extracted from C's actual training script.
- Historical scripts may use obsolete local paths. Their contents are preserved. Reusable modules and the new preprocessing/augmentation command-line scripts have no embedded dataset paths.
- No patient-level, scanner or acquisition metadata is invented. The source index is a file provenance index created during organization, not clinical metadata.
- Figures are existing project outputs, not newly validated explanations or masks. Verify final manuscript selection and source-data redistribution terms before publishing MRI-containing composites.
