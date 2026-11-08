# Reconstruction alignment outcome — 6 November 2026

The pre-experiment progress document remains the baseline record. All checksums in BEST_MODEL_MANIFEST.json were revalidated after these experiments.

24 runs compared ordinary decoder training, configuration-ranking reconstruction, one-step training-through-refinement, and their combination. All original branch/head/trace tensors matched the saved hybrid5 checkpoint bitwise. Two gradient/isolation tests passed.

Validation selected zero refinement steps. Best positive-update candidate was ranking, one step:65.04% validation,66.08% test, versus unchanged66.96% validation/67.91% test. Ordinary decoder control under the matched update produced65.11% test at one step. Neither ranking nor meta-training established an improvement. Combined objective was worse under this fixed decoder-only training recipe.

Best current deployable image-only candidate remains the original sparse four-branch hybrid5 MLP with soft magnitude inference:67.91% test,134,074 parameters,1.593MB model tensor state. No decoder or refinement is recommended from these results. Short-term pseudo-error memory remains a separately qualified option only for persistent temporal contexts.

Detailed results: ../../results/mnist_reconstruction_alignment_20261106/REPORT.md

Scope: these new objectives trained the decoder only, retaining the proven classifier trajectory. End-to-end meta-training of branch weights, alternative decoder architectures, or protected/generative class-conditioned objectives were not tested. Failure here does not establish impossibility of learned verification.
