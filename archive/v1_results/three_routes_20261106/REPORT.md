# Three routes to closing the image-only gap: analytic head, self-scoring routes, temporal context

Exploratory, 6 November 2026. Baseline: sparse 168×4 hybrid5 MLP, Split MNIST, 3 seeds × 2 orders, image-only test 67.91 ± 1.79%, true-task oracle 97.98%. All retraining reproduces the saved hybrid5 checkpoints bitwise (asserted). All hyper-parameters are chosen on the 5,000-image validation split; test is evaluated once per selected setting. ± is SD over three seed means (orders averaged). The MNIST test set has been used repeatedly in this project; treat the numbers as exploratory.

## Route 1: replay-free online analytic head — NEGATIVE

At each task boundary, one no-grad pass over the current task's training images accumulates ridge statistics (Gram and cross-correlation) with the weights current at that time; the head is solved once at the end. No old images are revisited.

| Head | Features | Test acc | Statistics |
|---|---|---:|---:|
| Learned (baseline) | 168 mixed | 67.91 ± 1.79% | – |
| Online ridge, λ=100 (val-selected) | 168 mixed | 26.88 ± 0.53% | 121 KB |
| Online ridge, λ=100 (val-selected) | 672 branch | 21.78 ± 0.19% | 1.84 MB |

Only the last-trained pair is classified (per-pair test ≈ 100% for the last pair, 0–14% for the rest). The soft-magnitude features of old tasks drift so much that their stored statistics no longer describe the final network. The 87.8%/93.5% joint-ridge probes in `mnist_readout_diagnosis_20261106` were only possible because they recomputed old-task features with the final weights, which needs the old images. **The bottleneck is representation drift in the gated mixture, not just head bias.** The chosen λ lies at the grid edge; a larger λ cannot fix stale statistics.

## Route 2: each archived route scores its own classes — NEGATIVE

The learner knows which classes it trained in each task (training-time knowledge, not a test-time task ID). For a test image, all five archived hard routes are evaluated. Each route t scores the image for its own classes (max logit, energy, task probability, or margin), optionally normalised by own-task training statistics recorded at its boundary. The highest-scoring task is chosen, and prediction is made within it.

| Scoring (val-selected: energy, normalised) | Test acc | Task-identification acc |
|---|---:|---:|
| energy, normalised | 48.22 ± 4.60% | 48.78% |
| max-logit, raw | 45.77 ± 1.57% | 45.86% |
| Oracle task, same routes | 97.98% | 100% |

Routes are highly distinct: 77.9% of neurons choose a different branch between any two task routes (75% would be random). The routes nonetheless claim foreign inputs, because each was trained only on its own task's data and never on rejection. Task identification is above chance (20%) but worse than the stateless model's own implied task accuracy (~68%). Making routes work as out-of-distribution detectors would require a training-time objective (PEC-style), not post-hoc scoring.

## Route 3: temporal context with switch handling — POSITIVE, with qualifications

Frozen hybrid5 checkpoints, causal label-free inference (predict, then update). Streams: fixed 200-image task blocks, variable-length blocks (geometric, mean 50), and iid mixed. **Every method is selected on validation by the mean accuracy over all three stream types**, so failing on mixed streams is penalised.

| Method | Blocked 200 | Variable (mean 50) | Mixed |
|---|---:|---:|---:|
| none (stateless) | 67.91 ± 1.79 | 67.91 ± 1.79 | 67.91 ± 1.79 |
| class-prior EMA | 70.50 ± 3.14 | 69.68 ± 2.93 | 63.02 ± 3.36 |
| class-prior EMA, gated | 71.92 ± 3.22 | 70.56 ± 3.19 | 67.61 ± 1.93 |
| pseudo branch state | 78.48 ± 3.92 | 76.81 ± 3.30 | 61.49 ± 2.93 |
| **pseudo branch state, gated** | **75.98 ± 2.83** | **74.58 ± 2.75** | **67.69 ± 1.89** |
| task-level filter (fixed switch rate) | 76.38 ± 4.11 | 76.17 ± 3.94 | 59.80 ± 3.78 |
| task-level filter (switch-rate mixture) | 82.75 ± 4.28 | 80.93 ± 3.90 | 57.29 ± 9.68 |

Paired seed-level differences, 95% t intervals (n = 3):

| Stream | gated branch − stateless | gated branch − gated prior | filter mixture − stateless |
|---|---:|---:|---:|
| Blocked | +8.06 [+5.47, +10.66] | +4.05 [+2.47, +5.63] | +14.84 [+8.57, +21.10] |
| Variable | +6.66 [+4.20, +9.12] | +4.01 [+2.73, +5.30] | +13.02 [+7.48, +18.55] |
| Mixed | −0.23 [−0.70, +0.25] | +0.08 [−0.04, +0.20] | −10.62 [−30.55, +9.31] |

Gate: the state strength is scaled by an EMA (0.9) of whether consecutive stateless predictions fall in the same digit pair, thresholded at c0 (selected value 0.3). The identical gate is applied to the class-prior control.

Interpretation:
- **Switch handling works:** the gated branch state never hurts mixed streams (−0.23 pp, interval spans 0) and keeps most of its gain in persistent streams.
- **Branch-specific beyond the shared control:** with identical gating, branch state beats the class-prior state by about +4 pp. That is half of the earlier ungated +8.4 pp; part of the earlier margin was the prior's poor switch handling.
- **The task-level filter is the strongest in persistent streams but not robust:** it uses the archived routes (≈98% within-task), so a correct task inference pays off heavily, yet it fails on mixed streams despite the switch-rate mixture. It is a non-dendritic, task-level mechanism and needs its own switch detection before it can be used.
- **Scope:** inference-time adaptation on frozen weights only. It exploits temporal persistence of the test stream and does not reduce training-time forgetting. Three seeds; exploratory test reuse.

## Inference cost and a label-free Taylor shortcut

The pseudo-error target is the model's own prediction, so the first-order shortcut from `mnist_gradient_branch_scoring` applies without true labels: g_i = Σ_c (p_c − onehot(pred)_c) W_ci, and branch error ≈ g_i·(z_ib − h_i). Across the six checkpoints, the Taylor shortcut picks the same best branch as the exact probes 98.6% of the time. Gated-state test accuracy is unchanged (Taylor 76.00 / 74.60 / 67.69 vs exact 75.98 / 74.58 / 67.69 on blocked / variable / mixed).

Per-image streaming latency, batch 1, CPU, 2 threads, one checkpoint (Python-overhead dominated; not a hardware benchmark): stateless 119 µs, exact probes 185 µs (+55%), Taylor probes 149 µs (+25%). Script: `../../bdh_cl/scripts/three_routes_inference_cost.py`.

## Files

- `../../bdh_cl/scripts/three_routes_analytic_head.py` → `analytic_head/`
- `../../bdh_cl/scripts/three_routes_self_scoring.py` → `self_scoring/`
- `../../bdh_cl/scripts/three_routes_temporal.py` → `temporal/` ; gated-prior control `../../bdh_cl/scripts/three_routes_prior_gated.py` → `temporal/prior_gated_test.json`
- Run with the `cl-tokenformer/.venv` interpreter and `PYTHONPATH=.` from `bdh-d/`.
