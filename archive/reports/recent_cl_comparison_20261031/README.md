# Recent continual-learning comparison: 2025–2026 publications

**Later extension:** the [missing-method study](../leaderboard_constraints_20261031/README.md)
adds size-matched PEC and PCA50 statistical/tree comparators. PEC reaches
91.35% versus the branch model's 74.76%. The earlier highlighted selected-method
view is historical and does not establish leadership among the expanded
exemplar-free comparison.

These are new measured **adaptations**, not original-paper reproductions or a verified latest-SOTA leaderboard. Recent publication year alone does not establish SOTA. Published scores on other datasets are not mixed into Split-MNIST plots.

| Method | Accuracy ± seed SD | Retained state MB | Additional optimizer MB | Deployed coefficients |
|---|---:|---:|---:|---:|
| OCAR (ICML 2025), adapted | 93.06 ± 0.13% | 14.314882 | 0.000000 | 529,480 |
| RFE (TMLR 2026), adapted | 59.26 ± 0.52% | 3.524176 | 2.823968 | 528,014 |

## What the evidence supports

The matched branch model has 74.76% autonomous accuracy versus 97.39% with oracle route selection: a 22.63-point diagnostic access gap. Oracle selection uses the true task/pair to select a route, with all ten output classes eligible. This motivates context/routing research; it does not guarantee recovery or establish novelty. The auxiliary-loss estimate is +2.65 points over the CE-only branch, with a descriptive paired 95% t interval of −3.30 to +8.61 points across only three seed means.

Raw class means (82.32%, 0.03144 MB) and DER++ with 1,000 images (94.52%, 2.951096 MB) remain important measured targets. A short adapted run that performs poorly does not refute a method’s native published results.

## Recent literature selection and status

- **OCAR, ICML 2025:** [official publication](https://proceedings.mlr.press/v267/urettini25a.html), [official code](https://github.com/edo-urettini/CL_stability). Curvature-aware replay using K-FAC. Reports improved continual metrics on its native benchmarks; adapted and measured here.
- **RFE, TMLR January 2026:** [paper](https://openreview.net/pdf?id=9NnhVME4Q6), [official code](https://github.com/mail-research/retrospective-feature-estimation). Chains retrospector modules to estimate old feature spaces. Relevant recent replay-free comparator, not claimed here as an overall SOTA winner; adapted and measured here.
- **LOR, CVPR 2026:** [official publication](https://openaccess.thecvf.com/content/CVPR2026/html/Lai_Beyond_Myopic_Alignment_Lookahead_Optimization_for_Online_Class-Incremental_Learning_CVPR_2026_paper.html). Claims SOTA on Seq-CIFAR10/100 and Seq-TinyImageNet. No official implementation was verified during this audit; **not run**, no invented Split-MNIST value.
- **SES, ICLR 2025:** [official paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/92ce40962b4098f7bf6eed33128fc606-Paper-Conference.pdf). Includes Split-MNIST Class-IL and structural-entropy sample selection. Its task-level coreset selection has a different data-access regime from an online reservoir; **not run** here.
- **MoAL, CVPR 2025:** [official publication](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Knowledge_Memorization_and_Rumination_for_Pre-trained_Model-based_Class-Incremental_Learning_CVPR_2025_paper.html). Uses pre-trained models and analytical learning; not directly comparable to this from-scratch MLP budget. **Not run** here.

This is a targeted recent-literature extension, not an exhaustive 2025–2026 SOTA audit.

## Shared controls

Same cached MNIST, fixed 55k training / 5k validation split and official 10k test, standardized pixels, two current-task epochs, batch 128, seeds 101/102/103 and forward/reverse task orders. All ten classes compete at test time without true task IDs. Six validation configurations per method on seed 113 forward; frozen before any official-test evaluation. SD is across three seed means after averaging orders, not six independent seeds.

## Material adaptations

**OCAR:** same 784→666 tanh MLP as previous dense baselines. Uses the official nngeometry FIM, PMatKFAC inverse, and gradient-vector kernels. The released NGPlugin is configured with CLEAR/domain-specific weighting and warns that Class-IL requires manual changes. Our explicit Class-IL branch weights replay Fisher contributions by known/current class count, using all ten outputs, rather than importing that CLEAR branch unchanged. SGD is intrinsic to the natural-gradient algorithm; no Adam and no gradient clipping, as upstream. One update per batch. Same 200-image uint8 reservoir as previous replay arms. EMA is the released CIFAR100 value 0.1385127; validation tests three LRs (.0005/.001/.003) × two damping increments (3.67164e−8, 1e−5). This preserves exact library curvature kernels but is **not** a claim of reproducing the full original training pipeline.

**RFE:** uses the official auxiliary CNN and Retrospector classes verbatim via AST loading. Main extractor 784→444 ReLU; d_aux=32 and d_projection=16; all five two-way classifiers, five auxiliary CNNs and four retrospectors are counted. This yields 528,014 deployed coefficients (0.23% fewer than the branch model). ReLU matches the original nonnegative feature space; it differs from the earlier tanh backbone. Auxiliary inputs use bilinear 28→32 resizing and repeat grayscale across RGB channels to satisfy the official module shape. Uses two additional module-training epochs per task, Adam .005, and a continuous backbone Adam with validation LR .0005/.001/.003 × regularization 0/1. No augmentation or learning-rate scheduler in this short shared-budget adaptation. The two-way task heads are evaluated for every image and combined exactly as the official Class-IL inference; no test task is supplied. Teacher extractor/classifier copies required to continue learning are counted separately from deployed coefficients. These smaller modules and short schedules differ materially from the original ResNet experiment.

## Resources and checks

Numeric retained state includes models, teachers, context/routes, curvature factors and retained inverse factors, raw replay buffers, and mapping/scalar state. Optimizer state is additional. This is not peak system RAM: dataset caches, gradients, temporary modules/activations and Python/RNG overhead are excluded. RFE’s temporary module Adam peak is recorded separately in JSON, as are extra module presentations and training time. A parameter match is not a compute or peak-memory match.

Three focused tests pass: Class-IL predictions match the actual upstream RFE inference; curvature/EMA agree with the upstream Fisher implementation and remove hooks; all auxiliary RFE modules fit the declared parameter budget. Evaluation state hashes remain unchanged. Pinned upstream commits and exact source are in the experiment directory.

## Artifacts

- [All measured baseline values](01_all_baselines_recent.png)
- [Complete Pareto plot](02_complete_pareto_recent.png)
- [Retained-state components](03_retained_state_components.png)
- [All data CSV](all_baselines.csv); editable SVG copies accompany every PNG.
- Earlier diagnostic and ablation plots remain in `../continual_learning_evidence_20261031/`.

## Highlighted presentation view

[Project opportunity plot, class-mean methods excluded](04_project_highlight_no_class_means.png) (editable SVG also available). Panel A explicitly restricts comparison to no stored historical examples; Panel B retains all continual neural methods, including higher-accuracy replay methods. The blue star is visual emphasis, not a claim of being a statistical outlier. Original complete plots and data remain available.
