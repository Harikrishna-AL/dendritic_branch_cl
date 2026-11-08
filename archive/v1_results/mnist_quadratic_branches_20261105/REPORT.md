# Quadratic branch interaction pilot

Three seeds101–103, each forward/reverse order. Two branches per168neurons. Same training data split and original trace+auxiliary objective, continuous Adam.0005, twoepochs/pair, batch128. One fixed recipe, no per-arm hyperparameter tuning.

| Arm | Image-only accuracy ± seed SD | Earlier-pair drop | Oracle saved-route accuracy | Parameters |
|---|---:|---:|---:|---:|
| original | 56.26 ± 1.13% | 33.56 pp | 85.11% | 265450 |
| normalized | 56.03 ± 3.46% | 31.23 pp | 84.07% | 265450 |
| extra_linear | 55.40 ± 3.11% | 33.01 pp | 83.17% | 792634 |
| quadratic | 49.30 ± 1.15% | 49.43 pp | 86.21% | 792634 |

Image-only inference uses the strongest absolute unsquashed response per neuron, retaining its sign. Training uses tanh and error-trace selection in every arm. Oracle selects saved indices using the true pair, retaining tanh and all10eligible outputs. This is the TWO-branch study, so its oracle baseline differs from the earlier FOUR-branch97.39% result.

Quadratic response is first(x)+alpha*u(x)*v(x). Alpha starts at zero. Normalization is LayerNorm over784input dimensions without trainable affine parameters. Extra-linear response uses first(x)+alpha*(u(x)+v(x))/sqrt2, matching the added parameter count and normalization while omitting multiplication. This factorized linear control is algebraically redundant, rather than a wider independent nonlinear network.

This is a low-rank raw-pixel branch adaptation inspired by the paper, using one learned product per branch. The paper evaluates channel-wise quadratic CNN layers with different training recipes. The experiment does not reproduce those results. No clear gain occurred in image-only accuracy or forgetting under this recipe. Oracle accuracy improved1.10pp versus original, with substantial seed variability.

One focused implementation test passed: zero-initialized quadratic arm exactly equals the normalized baseline, counterfactual logits match explicit substitutions, and both product projections receive gradients once alpha is nonzero. Evaluation checks immutable model tensors. All24runs completed. Source snapshot, protocol, per-stage results and aggregate SD are retained.
