# Local branch plasticity: what blocks acquisition, and can recruitment fix it?

The local recruitment rule substantially improves the branch model on Fashion MNIST without adding stored state. It does not establish a general advantage over a comparably sized ordinary MLP with a presentation-based learning-rate schedule. The fixed recruitment rule fails to improve overall MNIST accuracy on the paired transfer check. These positive and negative findings are both retained.

## Controlled changes

All local arms use the same 64 neurons × 4 branches, 256 input-response units, fixed class-code table, input weights, previous-minibatch error state, usage state, neighbor coactivity tables and inference decoder. Persistent numeric storage remains 812,480 bytes. The SGD MLP is 784→256→10 with 814,120 weight/bias bytes, no momentum or replay. A scalar training-presentation counter drives its schedule; it is not an image/configuration memory bank. Temporary working storage and Python overhead are excluded.

1. **Protection on/off:** original rate is η/(1+successful usage/τ); the off arm uses constant η. Successful-usage state is still recorded for a controlled comparison.
2. **Local MSE/BCE gradients:** original update is input × (desired−response) × (1−response²). BCE uses input × (desired−response), the exact gradient for probability (1+tanh(a))/2 and target (1+code)/2. Nominal η, codes, branch-error selection and reliable coactivity stay fixed. This tests the saturation factor, not removal of coded teaching. Gradient magnitudes differ, and this is not a broad independently tuned comparison of all teaching schemes.
3. **Novelty-triggered recruitment:** if a neuron’s best branch has previous-minibatch mean squared error greater than 0.4, select its least-used branch instead. Otherwise select its lowest-error branch as before. The threshold is predefined, not selected on test data. It uses existing local error/usage state and does not receive a task boundary or ID. A high local error is not proof of a task change.
4. **Relative recycling:** additionally measure damping relative to the least-used branch in that neuron, keeping one candidate more plastic. This changes allocation and protection together; interpret it as a package.
5. **Slowing MLP:** η(t)=η₀/(1+t/s), with t the number of current training presentations. No domain resets or boundary information. The primary schedules use s=100,000/400,000; a follow-up broadens to 1,000/3,000/10,000/30,000. Validation also chooses schedules with recent-domain accuracy close to the branch model.

## Protocol and selection

MNIST and Fashion MNIST, all ten classes in every domain, identity followed by random pixel permutations. No task IDs, replay, input prototypes or saved configuration archive enters the learner. Each current domain receives two passes; original source images reappear transformed in later domains as in standard permuted-MNIST.

Development uses fresh seeds 360/370, a stratified 50,000/10,000 training split and a fixed 2,000-example validation subset, excluded from every training domain. Six local variants and the MLP schedules are compared at 16 domains. Local selection maximizes the harmonic mean of all-domain and recent-four accuracy. A separate success gate requires newest accuracy +5 points, first-four accuracy within −5 points, and overall accuracy within −2 points of the original branch validation result.

Fashion novelty recruitment passes that gate. The selected MNIST relative-recycling variant fails it and is reported as a diagnostic, not a solution. Confirmation uses three fresh paired seeds per dataset (361–363 / 371–373), 32 domains, all 60,000 training samples, and all 10,000 test samples per domain at the final checkpoint. Intermediate curves use a fixed stratified 2,000-example test subset. Hyperparameters stay fixed. No test labels are supplied to prediction or used to select settings.

The stronger slowing-MLP and MNIST recruitment-transfer checks reuse the same confirmation seeds, so they are paired follow-ups, not extra independent replications. The recruitment transfer is explicitly adaptive: it tests the already defined rule on MNIST after observing its Fashion confirmation. No new threshold or update rule is tuned. Slowing schedules match recent-domain validation accuracy at 16 domains; they are not assumed to remain exactly acquisition-matched at 32. Actual gaps are shown below.

## Development ablations

| Dataset | Variant | All-domain accuracy | First four | Newest |
|---|---|---:|---:|---:|
| mnist | original | 63.90% | 71.07% | 64.85% |
| mnist | protection_off | 60.17% | 51.91% | 71.60% |
| mnist | bce_protected | 64.20% | 70.99% | 63.90% |
| mnist | bce_unprotected | 58.77% | 48.50% | 73.90% |
| mnist | novelty_allocation | 62.33% | 67.74% | 67.80% |
| mnist | relative_recycling | 64.21% | 53.74% | 77.30% |
| mnist | mlp_constant | 70.18% | 50.36% | 95.65% |
| mnist | mlp_decay_100000 | 70.73% | 57.16% | 90.40% |
| mnist | mlp_decay_400000 | 71.73% | 54.75% | 94.15% |
| fashion_mnist | original | 48.34% | 58.21% | 40.25% |
| fashion_mnist | protection_off | 48.34% | 40.06% | 55.80% |
| fashion_mnist | bce_protected | 50.48% | 60.13% | 38.60% |
| fashion_mnist | bce_unprotected | 51.68% | 42.48% | 70.55% |
| fashion_mnist | novelty_allocation | 64.03% | 67.75% | 67.35% |
| fashion_mnist | relative_recycling | 56.72% | 53.51% | 60.65% |
| fashion_mnist | mlp_constant | 63.45% | 53.49% | 83.35% |
| fashion_mnist | mlp_decay_100000 | 68.62% | 69.95% | 72.95% |
| fashion_mnist | mlp_decay_400000 | 70.38% | 69.06% | 77.70% |

Removing protection raises newest-domain accuracy but substantially reduces early retention. Changing the local gradient alone has little effect under absolute protection. Its acquisition benefit is larger without protection, especially on Fashion MNIST. Novelty recruitment is the clearest favorable change on Fashion validation; relative recycling does not preserve old accuracy on MNIST.

## Full-test confirmation at 32 domains

| Dataset | Model | All-domain accuracy | First-four accuracy | Newest accuracy |
|---|---|---:|---:|---:|
| mnist | Original branches | 53.37% | 65.32% | 54.48% |
| mnist | Novelty recruitment | 52.44% | 58.53% | 63.84% |
| mnist | Constant MLP | 54.03% | 27.09% | 96.55% |
| mnist | Primary slowing MLP | 57.80% | 39.45% | 94.15% |
| mnist | Stronger slowing MLP | 61.32% | 63.37% | 70.30% |
| fashion_mnist | Original branches | 41.50% | 57.55% | 33.04% |
| fashion_mnist | Novelty recruitment | 57.79% | 60.80% | 63.81% |
| fashion_mnist | Constant MLP | 44.45% | 29.22% | 82.30% |
| fashion_mnist | Primary slowing MLP | 59.40% | 51.42% | 74.96% |
| fashion_mnist | Stronger slowing MLP | 61.14% | 64.72% | 67.30% |

![Retention and acquisition](/Users/harikrishna/Documents/lossfunk/graph_cl/q_graph_cl/bdh-d/reports/local_graph_plasticity_20261013/retention_acquisition.png)

Primary slowing MLP uses the independently selected 400,000 schedule on both datasets. Stronger slowing MLP is selected to approximate the selected local variant’s recent-four validation accuracy: 10,000 on MNIST and 100,000 on Fashion. It was not selected by these test scores. The MNIST validation target was relative recycling, not the later novelty-transfer arm. See the complete control table below rather than interpreting this as exact matching of all final methods.

## Every additional confirmation, including unsuccessful changes

| Dataset | Follow-up / diagnostic | All-domain accuracy | First four | Recent four | Newest |
|---|---|---:|---:|---:|---:|
| mnist | selected local (relative_recycling) | 52.66% | 34.72% | 74.69% | 74.39% |
| mnist | acquisition_match_original | 51.39% | 71.81% | 41.39% | 43.46% |
| mnist | acquisition_match_selected | 61.32% | 63.37% | 65.96% | 70.30% |
| mnist | best_extended_schedule | 58.86% | 47.14% | 79.00% | 83.41% |
| fashion_mnist | selected local (novelty_allocation) | 57.79% | 60.80% | 61.99% | 63.81% |
| fashion_mnist | acquisition_match_original | 39.47% | 50.99% | 33.82% | 37.43% |
| fashion_mnist | acquisition_match_selected | 61.14% | 64.72% | 61.79% | 67.30% |

## What is supported

On Fashion MNIST, the original local allocation was a major bottleneck. Recruiting a less-used branch when existing branches recently fit poorly improves current learning without sacrificing the mean early-domain retention in these three paired runs. That is a real within-model improvement with the same stored state, training data and feedback. It is not a universal improvement: the exact fixed rule improves current MNIST accuracy at the cost of old accuracy and slightly lower overall accuracy.

The ordinary slowing MLP removes a substantial part of the apparent special retention advantage. On Fashion, its stronger schedule has higher means than novelty recruitment for overall, first-four and newest accuracy. On MNIST, the stronger slowing schedule also outperforms novelty transfer on all three means. The broad claim that dendrites inherently retain better than ordinary networks is not supported by this comparison. Three seeds are insufficient for a broad significance claim.

## Remaining uncertainty and next discriminating direction

Usage-based slowing helps preserve old performance but restricts later adaptation. Local recruitment can improve that balance. The current experiment does not establish that branch structure, rather than plasticity allocation and fixed coded teaching, supplies a unique advantage. Teaching labels still provide a fixed desired response at every neuron; a biological source of that supervision is not demonstrated. Learned coactivity counts are explicit distributed memory, although examples and task-indexed configurations are not stored.

The next useful comparison is a local flat learner with the same novelty/protection rule and teaching feedback, plus acquisition-focused alternatives to the fixed random class codes. A further claim should survive the ordinary slowing controls and the known inference-cost disadvantage. No conference-readiness or biological-realism claim is made.

## Verification and reproduction

Original variant exactly reproduces the prior branch training tensors on a small sequence. MSE and BCE local updates match their selected-unit autograd gradients. Slowing SGD follows the declared presentation-based schedule. All candidate choices and failed gates are archived.

- `bdh_cl/scripts/local_graph_plasticity.py`, `results/local_graph_plasticity_20261013/`.
- `bdh_cl/scripts/local_graph_matched_slowing.py`, `results/local_graph_matched_slowing_20261013/`.
- `bdh_cl/scripts/local_graph_recruitment_transfer.py`, `results/local_graph_recruitment_transfer_20261013/`.

Runtime: `/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python`, two CPU threads, cached real datasets. No external email or submission was sent.
