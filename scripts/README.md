# Workflow scripts

Scripts are grouped by task, then dataset. Names identify the operation and model where applicable. The reusable model is in [src/models/proposed.py](../src/models/proposed.py).

Start with [prepare_data.py](prepare_data.py) for command-line preprocessing and augmentation. The dataset-specific workflows preserve historical experiments: inspect their path settings before running, supply data and checkpoints, and run only the intended script. Renaming does not constitute rerunning or validating an experiment. Legacy and variant scripts are explicitly named.

## Ablation

- [train columnwise reshape (dataset_a)](ablation/dataset_a/train_columnwise_reshape.py)

## Augmentation

- [augment training images (dataset_a)](augmentation/dataset_a/augment_training_images.py)
- [augment training images (dataset_b)](augmentation/dataset_b/augment_training_images.py)
- [augment training images (dataset_c)](augmentation/dataset_c/augment_training_images.py)

## Cross dataset

- [evaluate external dataset (dataset_a)](cross_dataset/dataset_a/evaluate_external_dataset.py)
- [evaluate cross dataset (dataset_b)](cross_dataset/dataset_b/evaluate_cross_dataset.py)

## Cross validation

- [cross validate alternative reshape (dataset_a)](cross_validation/dataset_a/cross_validate_alternative_reshape.py)
- [cross validate columnwise reshape (dataset_a)](cross_validation/dataset_a/cross_validate_columnwise_reshape.py)
- [cross validate densenet121 (dataset_a)](cross_validation/dataset_a/cross_validate_densenet121.py)
- [cross validate efficientnet b3 (dataset_a)](cross_validation/dataset_a/cross_validate_efficientnet_b3.py)
- [cross validate proposed model (dataset_a)](cross_validation/dataset_a/cross_validate_proposed_model.py)
- [cross validate resnet50 (dataset_a)](cross_validation/dataset_a/cross_validate_resnet50.py)
- [cross validate vgg16 (dataset_a)](cross_validation/dataset_a/cross_validate_vgg16.py)
- [cross validate proposed model (dataset_b)](cross_validation/dataset_b/cross_validate_proposed_model.py)

## Data quality

- [audit laplacian image files (dataset_a)](data_quality/dataset_a/audit_laplacian_image_files.py)
- [audit laplacian zip archive (dataset_a)](data_quality/dataset_a/audit_laplacian_zip_archive.py)
- [build deduplicated datasets (dataset_a)](data_quality/dataset_a/build_deduplicated_datasets.py)
- [check dataset a b overlap (dataset_a)](data_quality/dataset_a/check_dataset_a_b_overlap.py)
- [preprocessing sensitivity pilot (dataset_a)](data_quality/dataset_a/preprocessing_sensitivity_pilot.py)
- [check cross dataset overlap (dataset_b)](data_quality/dataset_b/check_cross_dataset_overlap.py)

## Efficiency

- [benchmark single run models (dataset_a)](efficiency/dataset_a/benchmark_single_run_models.py)

## Evaluation

- [evaluate densenet121 (dataset_a)](evaluation/dataset_a/evaluate_densenet121.py)
- [evaluate efficientnet b3 (dataset_a)](evaluation/dataset_a/evaluate_efficientnet_b3.py)
- [evaluate proposed model (dataset_a)](evaluation/dataset_a/evaluate_proposed_model.py)
- [evaluate reshape variants (dataset_a)](evaluation/dataset_a/evaluate_reshape_variants.py)
- [evaluate resnet50 (dataset_a)](evaluation/dataset_a/evaluate_resnet50.py)
- [evaluate vgg16 (dataset_a)](evaluation/dataset_a/evaluate_vgg16.py)
- [evaluate inception v3 (dataset_b)](evaluation/dataset_b/evaluate_inception_v3.py)
- [evaluate proposed model (dataset_b)](evaluation/dataset_b/evaluate_proposed_model.py)
- [evaluate dataset c and transfer (dataset_c)](evaluation/dataset_c/evaluate_dataset_c_and_transfer.py)

## Explainability

- [explain gradcam plus plus (dataset_a)](explainability/dataset_a/explain_gradcam_plus_plus.py)
- [explain gradcam plus plus errors (dataset_a)](explainability/dataset_a/explain_gradcam_plus_plus_errors.py)
- [explain lime reference (dataset_a)](explainability/dataset_a/explain_lime_reference.py)
- [explain lime reproducible (dataset_a)](explainability/dataset_a/explain_lime_reproducible.py)
- [visualize attention weights (dataset_a)](explainability/dataset_a/visualize_attention_weights.py)

## Exploration

- [explore class distribution (dataset_a)](exploration/dataset_a/explore_class_distribution.py)
- [explore dataset b from a workflow (dataset_a)](exploration/dataset_a/explore_dataset_b_from_a_workflow.py)
- [explore class distribution (dataset_b)](exploration/dataset_b/explore_class_distribution.py)

## Preprocessing

- [preprocess images (dataset_a)](preprocessing/dataset_a/preprocess_images.py)
- [preprocess clahe median variant (dataset_b)](preprocessing/dataset_b/preprocess_clahe_median_variant.py)
- [preprocess images (dataset_b)](preprocessing/dataset_b/preprocess_images.py)
- [preprocess images (dataset_c)](preprocessing/dataset_c/preprocess_images.py)

## Statistics

- [compare five models statistically (dataset_a)](statistics/dataset_a/compare_five_models_statistically.py)
- [compare reshape variants statistically (dataset_a)](statistics/dataset_a/compare_reshape_variants_statistically.py)

## Training

- [train cnn bilstm legacy (dataset_a)](training/dataset_a/train_cnn_bilstm_legacy.py)
- [train densenet121 (dataset_a)](training/dataset_a/train_densenet121.py)
- [train efficientnet b3 (dataset_a)](training/dataset_a/train_efficientnet_b3.py)
- [train resnet50 (dataset_a)](training/dataset_a/train_resnet50.py)
- [train vgg16 (dataset_a)](training/dataset_a/train_vgg16.py)
- [train cnn bilstm legacy (dataset_b)](training/dataset_b/train_cnn_bilstm_legacy.py)
- [train proposed model group split (dataset_b)](training/dataset_b/train_proposed_model_group_split.py)
- [train proposed model legacy (dataset_b)](training/dataset_b/train_proposed_model_legacy.py)
- [train proposed model low memory (dataset_b)](training/dataset_b/train_proposed_model_low_memory.py)
- [train proposed model augmented split (dataset_c)](training/dataset_c/train_proposed_model_augmented_split.py)
