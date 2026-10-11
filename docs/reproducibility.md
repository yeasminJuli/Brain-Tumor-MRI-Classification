# Reproducibility and verification

## Verified during repository preparation (11 October 2026)

- All 475 files listed in `source_index.csv` matched the recorded SHA-256 hashes.
- All Python files under `scripts/` parsed successfully.
- The extracted proposed model ran on a synthetic 1 x 3 x 256 x 256 input: logits had shape 1 x 4, attention weights had shape 1 x 8 x 1, and the eight weights summed to one.
- The extracted model contains 3,817,413 parameters.
- `python scripts/prepare_data.py --help` completed successfully.

These are packaging and interface checks. They do not reproduce the paper's reported accuracy, retrain models or validate explanation quality.

## To reproduce an experiment

1. Obtain the exact dataset release and checkpoint used for that experiment. The public package does not include MRI datasets or trained weights.
2. Verify source provenance, class counts and class-to-output mapping. In C, Healthy maps to no-tumor (index 2); Pituitary Macroadenoma maps to pituitary (index 3).
3. Choose one experimental condition: original, cleaned, five-fold, ablation or cross-dataset. Do not mix their data splits or checkpoints.
4. Find its workflow in `script_index.md`. Historical scripts preserve machine-specific paths and some import-time behavior; update their path settings/imports in a working copy before execution. The example JSON config is documentation and is not automatically consumed by those scripts.
5. Record actual Python, framework, CUDA, GPU, seed, preprocessing and split settings. `requirements.txt` lists dependencies, not a verified historical lockfile.
6. Evaluate the fixed checkpoint on the intended test partition. Keep evaluation data separate from model selection and document any overlap that could not be excluded.
7. Compare outputs against the saved tables, allowing for documented differences in environment and protocol. Exact numerical reproduction has not been verified for every historical run.

## Before attaching this repository to a journal submission

- Link the specific commit or branch containing these organized materials; the default branch may still show the old layout.
- Confirm which experiment supports each manuscript table and figure.
- Confirm exact dataset sources/versions, checkpoint availability and code/data licensing with the author. No new license or distribution permission is asserted here.
- Keep the documented C validation limitation in the methods: augmentation preceded the file-level split, so related originals occur on both sides of validation.
- Do not describe cross-dataset testing as independent patient-level validation without the necessary overlap evidence.
