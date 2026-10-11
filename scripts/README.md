# Workflow scripts

Dataset A workflows are grouped by task. Historical scripts require checking data and checkpoint paths before execution. Start with [prepare_data.py](prepare_data.py) for command-line data preparation. The reusable [model](../src/models/proposed.py) is separate from the experiment scripts.

## Ablation

- [train columnwise and zigzag](ablation/dataset_a/train_columnwise_and_zigzag.py)

## Augmentation

- [augment training images](augmentation/dataset_a/augment_training_images.py)

## Data quality

- [audit laplacian image files](data_quality/dataset_a/audit_laplacian_image_files.py)
- [audit laplacian zip archive](data_quality/dataset_a/audit_laplacian_zip_archive.py)
- [preprocessing sensitivity pilot](data_quality/dataset_a/preprocessing_sensitivity_pilot.py)

## Efficiency

- [benchmark single run models](efficiency/dataset_a/benchmark_single_run_models.py)

## Evaluation

- [evaluate densenet121](evaluation/dataset_a/evaluate_densenet121.py)
- [evaluate efficientnet b3](evaluation/dataset_a/evaluate_efficientnet_b3.py)
- [evaluate proposed model](evaluation/dataset_a/evaluate_proposed_model.py)
- [evaluate reshape variants](evaluation/dataset_a/evaluate_reshape_variants.py)
- [evaluate resnet50](evaluation/dataset_a/evaluate_resnet50.py)
- [evaluate vgg16](evaluation/dataset_a/evaluate_vgg16.py)

## Explainability

- [explain gradcam plus plus](explainability/dataset_a/explain_gradcam_plus_plus.py)
- [explain gradcam plus plus errors](explainability/dataset_a/explain_gradcam_plus_plus_errors.py)
- [explain lime reference](explainability/dataset_a/explain_lime_reference.py)
- [explain lime reproducible](explainability/dataset_a/explain_lime_reproducible.py)
- [visualize attention weights](explainability/dataset_a/visualize_attention_weights.py)

## Exploration

- [explore class distribution](exploration/dataset_a/explore_class_distribution.py)

## Preprocessing

- [preprocess images](preprocessing/dataset_a/preprocess_images.py)

## Training

- [train cnn bilstm legacy](training/dataset_a/train_cnn_bilstm_legacy.py)
- [train densenet121](training/dataset_a/train_densenet121.py)
- [train efficientnet b3](training/dataset_a/train_efficientnet_b3.py)
- [train resnet50](training/dataset_a/train_resnet50.py)
- [train vgg16](training/dataset_a/train_vgg16.py)
