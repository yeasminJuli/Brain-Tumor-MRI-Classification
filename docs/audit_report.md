# Repository audit: 11 October 2026

## What passed

- 475 source-index hashes matched their copied artifacts.
- 148 tracked Python files passed syntax parsing before the audit script was added.
- 91 relative links in the main README and documentation resolved at the time of audit.
- All 105 tracked PNG files were readable.
- Nine evaluation accuracy tables matched their confusion matrices. Available per-class precision, recall, F1 and support were also reconciled.
- The C-to-C, A-to-C and C-to-A macro ROC AUC scores were recomputed from the original local prediction files and matched the saved scores. Those per-image files are not included in this public package.
- The proposed model passed a synthetic forward test with four logits, eight normalized attention weights and 3,817,413 parameters.
- The preprocessing command produced the same pixels as its underlying function. Augmentation produced four variants plus the original, was reproducible for a fixed seed, and refused to overwrite an existing output folder.
- No tracked model weights, ZIP archives, raw JPEG files or files above 50 MB were found. A scan for common credential formats found no matches. This is not a comprehensive security or medical privacy certification.

Machine-readable results are in [audit_results.json](audit_results.json). Run the public checks with `python tests/audit_repository.py` from the repository root after installing dependencies. Counts may increase as documentation or tests are added.

## What this does not establish

1. **Dataset C validation is not independent at original-image level.** The requested augmentation-before-split procedure shares augmented relatives between training and validation. The saved validation accuracy should not be used as evidence of unseen-original performance.
2. **Dataset independence is not established.** Cross-dataset evaluation does not itself prove absence of duplicate images, related slices or patients across datasets.
3. **Historical workflows are not all ready to run unchanged.** Syntax checks do not validate machine-specific paths, historical imports, scientific correctness or correspondence to a manuscript result.
4. **Exact historical environments and clinical metadata are incomplete.** The audit environment in the JSON is today's verification environment, not a reconstructed training environment.
5. **The original and cleaned experiments differ.** In the reconciled tables the cleaned test set has 1,319 images, whereas original A testing has 1,311. Keep their results and protocols distinct.
6. **Final manuscript consistency needs author review.** Check dataset citations, manuscript table/figure assignments, checkpoint availability, and rights to publish source-derived images. No code license or clinical validation is inferred by this audit.

No reported experimental scores were altered. Original repository files and history are preserved.
