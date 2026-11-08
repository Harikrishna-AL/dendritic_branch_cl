# Combined image-only selector experiment

Sparse 168-neuron four-branch MLP; 196 fixed random pixel inputs per branch. Three seeds × forward/reverse orders; two epochs per task; no replay, prototypes, pretraining or task identity at inference. Known boundaries during training. Hyperparameters are fixed in protocol.json.

| Training / inference | Validation accuracy | Validation oracle | Model MB | Parameters | Forgetting pp | Training s |
|---|---:|---:|---:|---:|---:|---:|
| baseline / primary | 63.62 ± 2.26% | 98.50% | 1.593 | 134,074 | 28.89 | 3.8 |
| baseline / mean | 55.57 ± 2.76% | 98.50% | 1.593 | 134,074 | 18.33 | 3.8 |
| baseline / magnitude_soft | 65.79 ± 1.69% | 98.50% | 1.593 | 134,074 | 24.55 | 3.8 |
| magnitude_joint / primary | 20.06 ± 0.14% | 86.22% | 1.593 | 134,074 | 98.63 | 4.2 |
| selector_hard / primary | 19.91 ± 0.11% | 98.33% | 1.782 | 181,370 | 99.12 | 4.9 |
| selector_soft / primary | 19.85 ± 0.02% | 61.43% | 1.782 | 181,370 | 98.92 | 4.7 |
| selector_soft_slow / primary | 19.85 ± 0.05% | 98.33% | 1.782 | 181,370 | 98.52 | 4.7 |

## Validation-selected official test comparison

Selected: baseline / magnitude_soft; mean validation 65.79%.

- baseline / primary: 64.52 ± 1.45% test accuracy.
- baseline / magnitude_soft: 66.70 ± 1.01% test accuracy.
- Paired seed-level difference: +2.18 pp; descriptive 95% t interval [+1.02, +3.33] pp.

Accuracy ± SD uses three seed-level means over task orders. Model memory includes compact branch weights and connection indices, excludes optimizer, temporary activations and diagnostic archives. Training time includes intermediate validation. Oracle uses true task identity to choose saved branch routes and is not image-only.

Baseline primary chooses greatest absolute raw response but forwards its tanh value. Mean and magnitude_soft are frozen post-training alternatives; magnitude_soft uses softmax(abs(raw)) weights. magnitude_joint adds image-only hard-winner CE to trace CE + auxiliary CE. Learned selectors add a 784→32→672 network, image-only CE, and soft-target supervision from per-image relative counterfactual branch errors. Hard selection uses straight-through softmax gradients. Soft selection uses weighted tanh branches. Slow variant reduces branch/head LR from .0005 to .0001 after task one while maintaining selector LR .0005. No importance-based weight protection was implemented.

Selector targets assess one-neuron substitutions relative to the trace route, not optimal simultaneous routes. Selectors can forget. These are fixed-recipe comparisons, not separately tuned methods. Selection is based on aggregate validation accuracy, and official test results are reported for selected configuration and baseline only.
