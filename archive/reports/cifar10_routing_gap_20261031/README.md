# Split CIFAR-10: from-scratch CNN routing-gap experiment

**New measured CNN experiment. No pretrained encoder, PCA, historical-example replay or test-time learning.** The aim is to test whether the MNIST route-access diagnosis survives natural images; failure to reproduce that diagnosis is an informative outcome.

| Method | Current gate | Prototype router | Learned context | Entropy router | True-pair oracle | Any route correct |
|---|---:|---:|---:|---:|---:|---:|
| Auxiliary loss | 18.37 ± 0.14% | 24.91 ± 0.87% | 30.17 ± 0.73% | 23.00 ± 3.13% | 61.83 ± 1.91% | 63.09 ± 2.62% |
| CE-only | 18.40 ± 0.21% | 24.40 ± 2.57% | 30.16 ± 2.78% | 21.40 ± 0.86% | 64.07 ± 5.81% | 64.34 ± 5.62% |

| Capacity / comparator control | Final test accuracy | Numerical model state |
|---|---:|---:|
| PEC | 33.77 ± 1.11% | 2.091708 MB |
| Offline joint auxiliary | 64.46 ± 1.24% | 2.102008 MB |
| Offline joint CE-only | 63.04 ± 0.77% | 2.102008 MB |

## Diagnostic findings

- **Auxiliary loss:** true-pair oracle minus prototype accuracy = 36.92 points; oracle minus learned-context accuracy = 31.66 points. Earlier-pair validation acquisition 86.13% versus final oracle 54.36%, a 31.77-point acquisition-to-final decline.
- **CE-only:** true-pair oracle minus prototype accuracy = 39.66 points; oracle minus learned-context accuracy = 33.91 points. Earlier-pair validation acquisition 85.91% versus final oracle 57.00%, a 28.91-point acquisition-to-final decline.

### Task-order check

| Objective / task order | Prototype | Learned context | Oracle |
|---|---:|---:|---:|
| aux / forward | 27.09% | 31.71% | 63.80% |
| aux / reverse | 22.74% | 28.63% | 59.86% |
| ce / forward | 26.15% | 31.13% | 61.89% |
| ce / reverse | 22.66% | 29.19% | 66.24% |

Auxiliary minus CE-only prototype accuracy: +0.51 points; descriptive paired95%t interval [-7.98, 8.99] across only three seed means.

The true-pair oracle is not a guaranteed recoverable ceiling or proof that all weights/skills survived. The any-route-correct diagnostic is a label-informed best-of-the-five-archives reference. If acquisition-to-final oracle accuracy falls substantially, retention/representation or classifier calibration also needs work; improved autonomous selection alone cannot be assumed to restore acquisition performance. Neither saved-route diagnostic searches the full combinatorial configuration space. Low archived-route accuracy does not prove that no other configuration retains useful capability. The joint models are independently trained capacity controls, not inference configurations of the sequential model.

## Protocol

Official CIFAR-10 archive matches MD5c58f30108f718f92721af3b95e74349a. Downloaded from MIT Open Learning’s mirror after the original server proved slow; the archive is byte-equivalent under the official checksum. 45,000 fit images,5,000 validation images (500/class, split RNG20261102), and official10,000 test images. Natural class pairs0/1,2/3,4/5,6/7,8/9, forward/reverse order, final seeds201/202/203; tuning seed213 forward. Five epochs per task, batch128, continuous Adam, clipping5, standard channel normalization, no augmentation, CPUtwo threads. This is a short resource-controlled CNN experiment, not a full CIFAR-SOTA reproduction or strict single-pass online benchmark. The5epoch schedule differs from the preceding MNIST2epoch schedule; scores must not be pooled across datasets.

**Branch network:** three3×3 stride2 padded Conv/ReLU layers3→16→32→64, flatten1024, 122neurons each with4tanh branches and one shared10-class head.525,014 trainable coefficients. The original per-neuron branch-error EMA.95 and hard minimum-error gate are preserved. Auxiliary loss is ordinary CE+.5×mean CE of single-neuron branch substitutions; it does not enumerate all joint branch combinations. CE-only retains error feedback, gates and route archives, removing only the auxiliary gradient. The encoder learns continually by backpropagation; no claim of isolated or biologically local CNN gradients.

**Autonomous routers:** prototype uses nearest full standardized RGB input task mean. Learned context uses128fixed random tanh features of8×8pooled RGB, online Gram/cross statistics and a ridge classifier over the five training phases. It uses known training boundaries to obtain context targets; it is not unsupervised task discovery. No old samples/features are replayed. Phase-end route archives use the same training boundary privilege. Entropy and margin compare all five route configurations; they cost additional branch/head evaluations. All ten classes remain eligible in every route and in every autonomous prediction. Only the true-pair/cyclic-wrong/best-of-bank diagnostics consult test labels to select or assess routes.

**PEC:** official PecCNN classes with GELU/InstanceNorm and2×2adaptive pooling, output99. Ten students with16/32 channels plus one shared frozen random teacher with117/234 channels,522,927 total coefficients (0.40% fewer than the branch network). All teacher weights are counted once; none is pretrained. Its intrinsic architecture and trainable fraction differ; this is a total-size comparison, not identical capacity or FLOPs. The frozen per-image teacher targets are cached for current-task epochs and discarded at the task boundary. No historical target cache survives into later task training. This is equivalent to recomputing the deterministic teacher without augmentation; the temporary current-task cache is working memory, not reported retained state.

**Selection:** each branch objective independently chooses LR .0005/.001/.003 by final prototype validation accuracy. PEC tests those LRs plus.01. Joint controls independently test the same three LRs using ordinary validation accuracy. Learned-context ridge .1/1/10 and entropy temperature .5/1/2 are selected on the validation run of each learning-rate candidate. All selections are frozen before final-test inference. The same selected configuration is then used on all six sequential confirmation runs. SD is across three seed means after averaging orders. No test-based best-router selection is reported as a primary result.

Selected settings are saved in `../../results/cifar10_routing_gap_20261031/selected_hyperparameters.json`.

## Memory and reproducibility

Numerical retained state includes model weights and error traces, phase-end route indices, and the selected router’s means or random projection/Gram/cross/head tensors. Model/route/prototype and model/route/learned-router costs are separate deployment variants. Adam state is additional. Peak RAM, dataset caches, gradients, activations, temporary teacher targets and allocator/Python overhead are excluded. Offline joint reference memory entries show the model only and do not charge its privileged access to the full training dataset.

Checks cover exact counterfactual neuron replacement, equivalence to explicit restored gates, fixed-state inference, online router statistics versus a batch solution, all10outputs and full teacher/student parameter accounting. The saved split, protocol, source, validation trials, all24confirmation runs and checkpoints are in the results directory. The retention audit uses held-out validation images only after models/settings are frozen; it fits no parameters.

## Artifacts

- [Routing conditions and oracle diagnostics](01_routing_and_oracle.png)
- [Joint capacity, PEC and memory](02_capacity_pec_and_memory.png)
- [Earlier-pair acquisition versus final oracle retention](03_acquisition_and_retention.png)
- [Summary CSV](summary.csv), JSON, editable SVGs, and per-seed gap/loss-effect values accompany the figures.
