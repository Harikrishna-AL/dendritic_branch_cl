# Stacking branch layers with the current MNIST recipe

Exploratory, 6 November 2026. Split MNIST, same split and normalisation as hybrid5; 2 epochs per task, Adam .0005, batch 128, clip 5; seeds 101–103 × forward/reverse order; ± is SD over seed means.

Branch network of depth L: layer 1 has 168 neurons × 4 tanh branches with 196 random pixels per branch; each further layer has 168 × 4 branches, each reading 42 of the previous 168 outputs (25%). Training at every layer: hard branch per neuron = argmin of an EMA(.95) of estimated substitution loss, CE + g·(z_branch − h), with g the per-example autograd gradient of CE w.r.t. that layer's output. Loss = 1.5 CE + .5 mean over layers of mean(g_detached · delta), i.e. hybrid5's linear surrogate on every step. Inference: soft-magnitude mixing at every layer. Oracle: archived routes at every layer plus true task. Control: ordinary tanh MLP of the same depth with width matched to the parameter count, CE only. One fixed recipe, no tuning.

| Depth | Model | Params | Acquisition | Image-only test | Oracle test |
|---:|---|---:|---:|---:|---:|
| 1 | branch | 134,074 | 98.8 | **69.63 ± 0.37** | 98.48 ± 0.44 |
| 1 | MLP | 133,570 | 98.9 | 20.26 ± 0.02 | – |
| 2 | branch | 162,970 | 98.7 | 54.59 ± 6.89 | 97.85 ± 0.43 |
| 2 | MLP | 163,095 | 99.2 | 20.31 ± 0.03 | – |
| 3 | branch | 191,866 | 97.9 | 46.17 ± 3.83 | 95.61 ± 4.29 |
| 3 | MLP | 191,825 | 99.3 | 20.28 ± 0.04 | – |

Per-run image-only test: depth 2 = 44.0, 56.2, 42.1, 60.3, 57.5, 67.5; depth 3 = 53.3, 47.8, 54.5, 34.5, 48.1, 38.9.

## Interpretation

- **Retention survives depth.** With the right configuration at every layer, old tasks stay at 96–98%. Ordinary MLPs forget completely at every depth (≈20%).
- **Image-only access degrades with depth and becomes unstable:** 69.6% → 54.6% → 46.2%, with depth-2 runs ranging 42–68%. Likely cause, not yet tested: every layer is trained on hard one-branch outputs but tested on soft mixtures, so from layer 2 on, each layer receives inputs unlike any it was trained on, and the mismatch compounds.
- **Side observation (not a validated comparison):** the depth-1 generic recipe (69.63 ± 0.37% test, 68.70% validation) scored above hybrid5 (67.91 ± 1.79% test, 66.96% validation) on the same seeds and orders, with lower variance. It differs from hybrid5 in using the first-order estimate on every step, with the gradient taken by autograd and detached in the auxiliary term. It needs a proper validation-selected comparison before replacing the baseline.

Code: `../../bdh_cl/scripts/multilayer_branch_recipe.py`.
