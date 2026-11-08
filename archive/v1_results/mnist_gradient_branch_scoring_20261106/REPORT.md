# Gradient-based branch scoring and cheaper auxiliary training

Matched sparse MNIST MLP: three seeds × two orders × five methods = 30 training runs. Same 168 neurons × four branches, 196 inputs/branch, 134,074 parameters, about 1.593 MB persistent model state. Two epochs per task; Adam .0005, batch128, clip5; same initialization and batch order. All image-only predictions use fixed soft magnitude weighting of tanh branches.

| Method | Image-only validation acc | Clean ms/step | Step speedup | Exact-step fraction | Observed full training s |
|---|---:|---:|---:|---:|---:|
| exact | 65.79 ± 1.69% | 4.201 | 1.00× | 100.0% | 3.819 |
| linear | 65.80 ± 2.04% | 1.460 | 2.88× | 0.0% | 1.301 |
| quadratic | 64.37 ± 1.62% | 1.665 | 2.52× | 0.0% | 1.477 |
| hybrid5 | 66.96 ± 2.30% | 1.983 | 2.12× | 20.1% | 1.828 |
| hybrid10 | 66.55 ± 1.70% | 1.718 | 2.45× | 10.0% | 1.594 |

Validation-selected approximate method: **hybrid5**.

| Method | Image-only test acc | Oracle test acc | Label-free state: persistent contexts | Label-free state: mixed stream |
|---|---:|---:|---:|---:|
| exact | 66.70 ± 1.01% | 98.33 ± 0.31% | 78.54 ± 1.73% | 49.46 ± 0.92% |
| hybrid5 | 67.91 ± 1.79% | 97.98 ± 0.77% | 79.07 ± 4.16% | 50.60 ± 3.06% |

Selected method minus exact, paired seed-level image-only test difference: **+1.22 pp**; descriptive 95% t interval [-2.51, +4.95] pp. Not a prespecified noninferiority test.

## Approximation quality on exact-baseline trajectories

| Estimate | Batch best-branch agreement | Per-example best-branch agreement | Mean exact-loss regret | Mean loss error |
|---|---:|---:|---:|---:|
| linear | 95.67% | 98.62% | 6.43e-06 | 0.000129 |
| quadratic | 99.94% | 99.48% | 2.45e-09 | 3.34e-06 |

Quality checks sample every 100 batches of exact training, excluded from training timing. Batch agreement compares minimum mean substitution loss across branches for each neuron; per-example agreement compares each image/neuron. Regret measures true substitution loss above the true best branch when selecting by approximate scores. These checks do not establish approximation quality on arbitrary architectures.

## Implementation

For selected hidden output h_i and alternative z_ib, delta=z_ib-h_i. Per-example hidden gradient is g_i = sum_c (p_c-onehot(y)_c) W_ci, the closed-form equivalent of CE backprop through the current linear head. First-order branch score is CE + g_i delta. Quadratic score adds .5 Var_p(W_ci) delta². EMA .95 uses these estimated absolute errors as before.

Auxiliary surrogate is not removed. Exact loss is CE + .5 mean(substitution CE). Taylor replacement is 1.5 CE + .5 mean(g_i delta), adding .25 mean(curvature delta²) for quadratic. Residual probabilities are stopped, but W and delta remain differentiable so the surrogate includes classifier-head updates; a stopped-g-only loss would omit some head gradients. No second backward pass is needed.

Hybrid5 uses exact objective and exact feedback on batches 0,5,10,... and linear surrogate/feedback otherwise. Hybrid10 uses batches 0,10,20,... . Trace and optimizer state never reset. No inverse-frequency loss scaling.

## Timing and limits

- Clean benchmark: original exact implementation without Taylor/quality calculations, or corresponding cheaper objective, then backward, clip, Adam, trace update. Three rotated method-order repeats, ten warmup steps and 100 measured steps per repeat on CPU two threads. Median step time reported. Reuses one current-task minibatch on one trained checkpoint; this is a local throughput benchmark, not an entire end-to-end application benchmark.
- Observed full training seconds exclude evaluation and quality probes. The exact training-run implementation also computes unused Taylor intermediates; therefore clean benchmark, not full-training ratio, is the primary speed comparison. Hybrid clean benchmark likewise uses the clean exact reference on correction steps.
- No B×neurons×branches×classes alternative-logit tensor is created on approximate steps. Branch responses, sparse projection materialization and normal classification forward/backward remain. Fixed architecture has ten classes; larger-architecture throughput is not measured.
- Closed-form gradient/curvature exploits the linear head. Deeper-layer branch scoring can use local backprop gradients, but this surrogate implementation is not yet a generic deep-network solution.
- All settings were fixed in advance; approximate arm selected on mean validation only; official tests evaluated for exact and selected method. Previously used MNIST test set makes this an exploratory experiment. Three seed-level averages over orders define SD; more seeds needed for precise accuracy claims.
- Inference uses the same algorithms and model size; it is not made faster by cheaper training. Optional short-term-state evaluation still computes its exact label-free counterfactual probes; decay .9 and strength .9 remain fixed without retuning.
- Three tests passed: gradient scale agrees with autograd, second-order correction improves small-perturbation loss approximation, and exact mode reproduces the original auxiliary objective.
