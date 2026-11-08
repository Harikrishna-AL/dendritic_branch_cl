# Four-branch quadratic interaction pilot

Same four-branch168neuron architecture as the original97.39%oracle experiment. Three seeds101–103, forward/reverse order. Two epochs per pair, batch128, continuousAdam.0005clip5, error-trace training plus.5counterfactualCE. All settings fixed before the runs.

| Model | Image-only accuracy ± seed SD | Acquisition-to-final loss, earlier pairs | Oracle saved route | Parameters |
|---|---:|---:|---:|---:|
| original | 56.56 ± 4.78% | 30.42 pp | 97.39% | 529,210 |
| normalized | 59.46 ± 4.20% | 26.02 pp | 96.76% | 529,210 |
| extra_linear | 56.29 ± 5.40% | 30.09 pp | 96.59% | 1,583,578 |
| quadratic | 51.07 ± 10.81% | 48.07 pp | 97.77% | 1,583,578 |

Image-only inference selects each neuron’s maximum-absolute raw response and retains its sign. It uses no task labels/prototypes/archives/errorfeedback. Oracle selects stored indices using true pair and tanh activations, with all10classes eligible. All models train tanh branches with error selection.

Normalized arms apply per-image LayerNorm without trainable affine parameters to all784inputs. Quadratic adds alpha*u(x)*v(x), with alpha initializedzero. Extra-linear adds alpha*(u(x)+v(x))/sqrt2 with exactly the same extra parameters. This linear control is a factorized linear response, not a wider nonlinear classifier.

The original reproduces56.56%image-only and97.39%oracle. Normalization alone increases mean image-only accuracy2.90pp and reduces mean acquisition-to-final loss4.40pp at unchanged parameter count. Quadratic gives+.38pporacle but lower image-only accuracy and larger variability. Three seed means and one fixed recipe do not establish statistical superiority or optimized quadratic performance. This is a low-rank MLP adaptation, not a reproduction of the paper’s CNN experiments.

Both2and4branch implementation checks pass: zero-initialized interaction equals normalized baseline; counterfactual logits agree with explicit substitutions; product projections receive gradients. Inference tensors remain unchanged, and all24runs completed. Per-stage raw data, source snapshot and protocol accompany this report.
