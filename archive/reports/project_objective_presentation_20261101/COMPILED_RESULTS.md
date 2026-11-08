# Compiled evidence for the project presentation

This registry combines the discussed experimental cohorts without pooling their statistics. Raw/frozen nearest-class-mean methods remain excluded from presentation views as requested. Statistical PCA controls are separated. No leaderboard values are substituted for local measurements. Detailed protocols and reconstruction limitations remain in linked source reports.

## MNIST baseline suite

| Method | Accuracy ± seed SD | Numerical payload MB |
|---|---:|---:|
| Replay: 1,000 images | 90.55 ± 0.29% | 2.911 |
| Replay: 200 images | 81.59 ± 0.93% | 2.277 |
| Branch + auxiliary loss | 74.76 ± 0.53% | 2.136 |
| Branch CE-only | 72.11 ± 2.64% | 2.136 |
| Online EWC | 23.38 ± 0.36% | 6.354 |
| Synaptic Intelligence | 20.34 ± 0.02% | 8.472 |
| Fine-tuning | 20.31 ± 0.02% | 2.118 |
| Numenta (5% activity) | 18.57 ± 1.99% | 3.357 |
| DER++: 200 images | 88.44 ± 0.40% | 2.285 |
| DER++: 1,000 images | 94.52 ± 0.11% | 2.951 |
| ER-ACE: 200 images | 85.72 ± 0.58% | 2.277 |
| ER-ACE: 1,000 images | 91.13 ± 0.25% | 2.911 |
| Numenta (20% activity)* | 20.28 ± 0.03% | 3.357 |
| Numenta (ReLU)* | 20.18 ± 0.06% | 3.356 |
| OCAR (ICML 2025), adapted | 93.06 ± 0.13% | 14.315 |
| RFE (TMLR 2026), adapted | 59.26 ± 0.52% | 3.524 |
| PEC (ICLR 2024), size-matched | 91.35 ± 0.19% | 2.117 |
| Streaming LDA, PCA50 | 86.06 ± 0.20% | 0.544 |
| Streaming QDA, PCA50 | 95.21 ± 0.17% | 0.724 |
| RLS / ridge, PCA50 | 83.23 ± 0.15% | 0.349 |
| VFDT (River), PCA50 | 33.62 ± 2.98% | 0.912 |
| MIST-G reconstruction, PCA50 | 87.06 ± 0.36% | 0.934 |
| Joint training | 96.97 ± 0.11% | 2.118 |
| hat | 75.22 ± 0.04% | 2.132 |
| twta | 36.32 ± 0.81% | 2.141 |
| ftn_slow | 71.62 ± 2.52% | 2.116 |

## Main interpretation

Error-trace training outperforms fixed/random/activation-WTA gates in a one-seed CE-only pilot. HAT is competitive and slightly higher on mean autonomous MNIST accuracy. PEC is stronger still. Auxiliary-loss benefit is uncertain. CIFAR-10 shows both access and retention limitations. Possible novelty lies in the specific counterfactual-error selection rule and a future input-dependent routing mechanism, not branch/task masking in general.

## Cohort reports

- [Matched branch/Numenta](../../results/matched_numenta_20261031/REPORT.md)
- [Broader MNIST controls](../leaderboard_constraints_20261031/README.md)
- [Recent MNIST controls](../recent_cl_comparison_20261031/README.md)
- [Gate and WTA pilot](../../results/mnist_gate_training_wta_20261101/COMPARISON.md)
- [HAT/TWTA/FTN adaptations](../../results/rawpixel_gating_baselines_20261101/REPORT.md)
- [CIFAR-10](../cifar10_routing_gap_20261031/README.md)
- [Earlier, different CIFAR-100 framework](../continual_learning_results_20261027/RESULTS.md)

## Scope

The JSON registry includes exact per-mode CIFAR-10 results and per-order gate pilot values. The presentation appendix retains exploratory and earlier-protocol checks separately. No three-seed SD is assigned to the one-seed gate pilot. Numerical state is not peak RAM or runtime. Earlier CIFAR-100/DCNet experiments use a different method and cannot be merged with the current branch/error-trace study.
