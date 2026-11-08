# Sparse-input MLP reproducibility

Split MNIST; original 168-neuron, four-branch MLP; three seeds × forward/reverse orders, two epochs per task. No replay, pretraining, task ID or saved routes in image-only inference. Test set is used for fixed, prespecified evaluation, not tuning.

| Normalization | Inputs per branch | Image-only raw acc | Image-only tanh acc | Oracle acc | Parameters | Model MB | Forgetting pp | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| False | 784 | 56.56 ± 4.78% | 53.75 ± 3.71% | 97.39 ± 0.90% | 529,210 | 2.120 | 30.42 | 3.7 |
| False | 392 | 61.78 ± 1.60% | 62.47 ± 1.47% | 98.09 ± 0.34% | 265,786 | 3.173 | 28.06 | 3.7 |
| False | 196 | 63.47 ± 1.10% | 64.52 ± 1.45% | 98.33 ± 0.31% | 134,074 | 1.593 | 28.33 | 3.5 |
| True | 784 | 59.46 ± 4.20% | 56.78 ± 4.22% | 96.76 ± 0.70% | 529,210 | 2.120 | 26.02 | 3.7 |
| True | 392 | 57.17 ± 2.48% | 57.62 ± 2.66% | 98.22 ± 0.48% | 265,786 | 3.173 | 31.30 | 3.8 |
| True | 196 | 62.05 ± 3.18% | 61.83 ± 2.25% | 98.21 ± 0.45% | 134,074 | 1.593 | 28.40 | 3.6 |

Accuracy ± SD uses three seed-level averages over the two orders. Memory includes compact weights and int64 connection indices; excludes optimizer, temporary matrices, activations and oracle archives. Runtime is training only on CPU, two threads. Raw inference retains the winning raw response; tanh inference retains its bounded response, matching training. Oracle uses true task identity only to select archived routes, with the full ten-class head.

Paired raw image-only differences against dense, with normalization matched:

- normalized=False, 392 inputs: +5.23 pp; descriptive seed-level 95% t interval [-2.70, +13.15] pp; positive in 5/6 seed/order pairs.
- normalized=False, 196 inputs: +6.91 pp; descriptive seed-level 95% t interval [-5.52, +19.35] pp; positive in 4/6 seed/order pairs.
- normalized=True, 392 inputs: -2.29 pp; descriptive seed-level 95% t interval [-17.16, +12.59] pp; positive in 4/6 seed/order pairs.
- normalized=True, 196 inputs: +2.59 pp; descriptive seed-level 95% t interval [-0.38, +5.56] pp; positive in 4/6 seed/order pairs.

Fixed shared learning rate; no independent per-arm tuning. Sparse branches access fixed random pixel subsets, which can overlap. This does not implement local receptive fields. Dense versus sparse keeps neuron/branch counts fixed, not parameter counts. Three seeds do not establish broad reproducibility across datasets.

Paired tanh-output differences against dense (same inference rule):

- normalized=False, 392 inputs: +8.72 pp; descriptive seed-level 95% t interval [+2.93, +14.50] pp; positive in 6/6 seed/order pairs.
- normalized=False, 196 inputs: +10.76 pp; descriptive seed-level 95% t interval [-0.35, +21.88] pp; positive in 6/6 seed/order pairs.
- normalized=True, 392 inputs: +0.85 pp; descriptive seed-level 95% t interval [-11.70, +13.39] pp; positive in 4/6 seed/order pairs.
- normalized=True, 196 inputs: +5.05 pp; descriptive seed-level 95% t interval [-1.78, +11.88] pp; positive in 5/6 seed/order pairs.
