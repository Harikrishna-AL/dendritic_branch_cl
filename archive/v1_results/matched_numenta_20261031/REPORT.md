# Matched-size Numenta comparison

**Primary budget: free learnable coefficients after fixed masks. All methods are retrained under a common protocol.**

| Model | Free coefficients | Allocated parameters | Selected learning rate |
|---|---:|---:|---:|
| numenta | 529,354 | 556,810 | 0.0005 |
| branch_aux | 529,210 | 529,210 | 0.0005 |
| branch_ce | 529,210 | 529,210 | 0.001 |

| Model | Inferred-context accuracy ± seed SD | Oracle-context accuracy | Old-class accuracy |
|---|---:|---:|---:|
| numenta | 18.57 ± 1.99% | 18.65% | 0.34% |
| branch_aux | 74.76 ± 0.53% | 97.39% | 74.15% |
| branch_ce | 72.11 ± 2.64% | 93.45% | 70.48% |

## Fairness controls

Numenta: original upstream two-layer architecture scaled to [64,64], five 784D segments per neuron, 50% feedforward weight masks, original 5% kWinners (3 winners/layer). Branch: 168 neurons × 4 tanh branches. Approximately 529k free coefficients each, within 0.03%; allocated dense parameter counts differ by about 5.2%. Masks, buffers and route archives are separately counted. No claim of equal active units, depth, executed FLOPs or memory.

Split MNIST, 500 held-out training images per digit (5,000 validation) selected with seed 20261101; remaining 55,000 for all training runs. Official 10,000 test images never used for tuning. Identical standard normalization (x/255-.13062755)/.30810780 for all methods. Forward and reverse pair orders; matched seed/stage/epoch minibatch permutations across methods.

Exactly three learning-rate candidates (.0005,.001,.003) per method, one separate calibration seed (113), forward order, two epochs per task. Select highest final unrestricted ten-class validation accuracy with inferred context; ties select smaller rate. Freeze before evaluating test data. Same small validation budget, not an exhaustive hyperparameter optimization or equal-FLOP tuning budget.

Five task-mean prototypes from the same 55,000 training inputs. Each method is allowed the same training boundaries and current-task prototype pass. Numenta uses current-task mean during learning; branch arms archive their current route at each boundary. Inference independently selects nearest prototype by Euclidean distance, then supplies its vector to Numenta or its associated route to branch arms. No test IDs, label masks, replay or historical weight checkpoints. Extra archived branch routes and traces count toward persistent state.

Two epochs per pair, batch 128, Adam, one optimizer across tasks, gradient clip 5 for every method; learning rate independently validation-selected. Numenta trains ordinary CE, retains native initialization and kWinners, and rezeros masked weights after updates. Branch auxiliary objective is CE+.5 mean single-neuron counterfactual CE; CE-only arm drops auxiliary gradients but retains identical counterfactual error-trace updates. No SI in any arm.

## Interpretation limits

One matched coefficient budget, three final seeds, two related task orders, and three LR trials per arm. Different depth, activation sparsity, initialization and model structure remain inherent method differences. Fixed task-mean routing is boundary-aware during training, not the paper's autonomous statistical clustering. This is a stronger controlled Split MNIST comparison, not an exact paper reproduction, proof of general superiority, or evidence that the auxiliary loss uniquely causes an advantage.

Primary results average forward/reverse orders within each seed before computing SD. Task-mean context acquisition uses known training boundaries in every arm. Oracle context is only a diagnostic; no output label mask is applied. The CE-only branch arm retains the same error-trace feedback and archives, so it isolates the auxiliary-loss effect.

Exact per-order results, allocation and persistent-state bytes, optimizer state, clipped-gradient counts, acquisition diagnostics, training times and batch-256 inference latencies are in results.json. Training seconds exclude validation, prototype preparation and output saving. These are wall-clock measurements on this CPU environment, not FLOP estimates or M4 benchmarks.

Evaluation asserts that model tensors and route archives remain unchanged. Official test data is evaluated only after validation selection is frozen. Validation indices, common prototype contexts, protocol, source, chosen hyperparameters and final checkpoints are saved.
