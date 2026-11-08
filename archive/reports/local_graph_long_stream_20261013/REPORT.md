# Does local branch memory degrade more slowly as domains accumulate?

This experiment extends the fixed-capacity comparison to 4, 8, 16 and 32 sequential pixel mappings on MNIST and Fashion MNIST. Reported means use three fresh paired seeds per dataset. See all-domain accuracy, retention of exactly the same first four domains, and newest-domain learning separately; none alone establishes a superior continual learner.

## Main finding

The original four-branch model loses much less accuracy on exactly the same first four domains than the MLP. MNIST: branches 82.70%→68.08% (14.62 pp drop), MLP 89.62%→24.79% (64.84 pp drop). Fashion MNIST: branches 62.62%→53.19% (9.43 pp drop), MLP 77.30%→25.31% (51.99 pp drop). This supports slower loss of early-domain performance for the current branch learning rule.

At 32domains, the eight-branch MNIST model has 56.49% all-domain accuracy versus 53.21% for the MLP, with 71.00% first-four accuracy. But newest-domain accuracy is 45.41% versus 96.15% for the MLP. On Fashion MNIST, both branch variants still trail MLP all-domain accuracy (38.38%/37.12% versus 41.86%). The central 8-slot model is strongest on Fashion all-domain accuracy at 46.66%, while the original branches retain the first four domains better.

These results expose a stability-plasticity trade-off rather than a universal win. Local usage damping and fixed coded teaching differ from ordinary fixed-rate SGD; slower forgetting cannot yet be attributed uniquely to dendritic structure. No significance claim is made from three paired seeds.

## Protocol

- All ten classes appear in every domain. Domain 1 is ordinary images; later domains use fixed random permutations. Every model sees identical domain prefixes and sample order for its seed.
- Two passes over all 60,000 current-domain training examples; no replay of earlier-domain images, task IDs, boundary notifications, optimizer resets or extra model capacity. Source images reappear under new transformations as in standard permuted-MNIST.
- All settings are frozen from the prior training-validation selections for eight domains. There is no tuning on the current official test results or separate optimization at 16/32 domains. This measures scaling of these fixed rules, not the best achievable performance of each architecture at every length.
- Final checkpoint accuracy uses all 10,000 test images for every seen domain. Acquisition is measured immediately after each domain using the same fixed stratified 2,000-example test subset. Labels are used only to calculate metrics at test time, never to select routes or update the model.
- Architectures stay fixed: 64×4 branches (812,480 bytes), 32×8 branches (815,104 bytes), SGD MLP784→256→10 (814,120 bytes), and 256-unit central activity memory with eight prototype slots (824,352 bytes). Numeric state excludes gradients and temporary working buffers. Central capacity is eight throughout, so its prototype capacity is also overloaded by long streams; a larger capacity at a rebalanced parameter budget would be a separate control.

![Scaling curves](/Users/harikrishna/Documents/lossfunk/graph_cl/q_graph_cl/bdh-d/reports/local_graph_long_stream_20261013/scaling.png)

## Full results

| Dataset | Domains | Model | All-domain accuracy | First-four accuracy | Newest-domain accuracy |
|---|---:|---|---:|---:|---:|
| mnist | 4 | 64 neurons × 4 branches | 82.70% | 82.70% | 80.99% |
| mnist | 4 | 32 neurons × 8 branches | 80.06% | 80.06% | 79.03% |
| mnist | 4 | MLP 256 / SGD | 89.62% | 89.62% | 96.75% |
| mnist | 4 | Central memory / 8 slots | 82.08% | 82.08% | 80.46% |
| mnist | 8 | 64 neurons × 4 branches | 75.68% | 79.56% | 69.94% |
| mnist | 8 | 32 neurons × 8 branches | 78.29% | 78.87% | 76.37% |
| mnist | 8 | MLP 256 / SGD | 82.77% | 73.27% | 96.69% |
| mnist | 8 | Central memory / 8 slots | 78.76% | 80.13% | 77.76% |
| mnist | 16 | 64 neurons × 4 branches | 65.39% | 74.47% | 57.81% |
| mnist | 16 | 32 neurons × 8 branches | 69.09% | 75.61% | 61.99% |
| mnist | 16 | MLP 256 / SGD | 70.89% | 51.18% | 96.70% |
| mnist | 16 | Central memory / 8 slots | 59.42% | 68.14% | 75.14% |
| mnist | 32 | 64 neurons × 4 branches | 54.40% | 68.08% | 50.23% |
| mnist | 32 | 32 neurons × 8 branches | 56.49% | 71.00% | 45.41% |
| mnist | 32 | MLP 256 / SGD | 53.21% | 24.79% | 96.15% |
| mnist | 32 | Central memory / 8 slots | 47.13% | 65.31% | 70.17% |
| fashion_mnist | 4 | 64 neurons × 4 branches | 62.62% | 62.62% | 62.21% |
| fashion_mnist | 4 | 32 neurons × 8 branches | 63.03% | 63.03% | 65.09% |
| fashion_mnist | 4 | MLP 256 / SGD | 77.30% | 77.30% | 80.56% |
| fashion_mnist | 4 | Central memory / 8 slots | 69.45% | 69.45% | 69.84% |
| fashion_mnist | 8 | 64 neurons × 4 branches | 55.08% | 61.71% | 46.68% |
| fashion_mnist | 8 | 32 neurons × 8 branches | 56.40% | 57.73% | 54.72% |
| fashion_mnist | 8 | MLP 256 / SGD | 72.02% | 67.18% | 80.62% |
| fashion_mnist | 8 | Central memory / 8 slots | 65.30% | 68.10% | 65.42% |
| fashion_mnist | 16 | 64 neurons × 4 branches | 45.98% | 57.54% | 36.99% |
| fashion_mnist | 16 | 32 neurons × 8 branches | 46.56% | 48.87% | 44.02% |
| fashion_mnist | 16 | MLP 256 / SGD | 58.68% | 42.47% | 81.95% |
| fashion_mnist | 16 | Central memory / 8 slots | 50.99% | 51.32% | 61.89% |
| fashion_mnist | 32 | 64 neurons × 4 branches | 38.38% | 53.19% | 36.54% |
| fashion_mnist | 32 | 32 neurons × 8 branches | 37.12% | 40.89% | 36.79% |
| fashion_mnist | 32 | MLP 256 / SGD | 41.86% | 25.31% | 82.39% |
| fashion_mnist | 32 | Central memory / 8 slots | 46.66% | 46.09% | 61.14% |

## Degradation from 4 to 32 domains

A positive drop means worse accuracy. First-four drop is the clearest same-input retention comparison; all-domain drop also reflects an expanding evaluation set and reduced acquisition of later mappings.

| Dataset | Model | All-domain drop (pp) | First-four drop (pp) | First-four accuracy retained at32 relative to4 |
|---|---|---:|---:|---:|
| mnist | 64 neurons × 4 branches | 28.30 | 14.62 | 82.3% |
| mnist | 32 neurons × 8 branches | 23.57 | 9.06 | 88.7% |
| mnist | MLP 256 / SGD | 36.42 | 64.84 | 27.7% |
| mnist | Central memory / 8 slots | 34.95 | 16.78 | 79.6% |
| fashion_mnist | 64 neurons × 4 branches | 24.23 | 9.43 | 85.3% |
| fashion_mnist | 32 neurons × 8 branches | 25.91 | 22.14 | 64.8% |
| fashion_mnist | MLP 256 / SGD | 35.44 | 51.99 | 32.8% |
| fashion_mnist | Central memory / 8 slots | 22.80 | 23.37 | 66.4% |

## Learning and forgetting on the same subset

Forgetting since acquisition is mean max(0, accuracy immediately after learning an old domain minus its final accuracy). Signed backward transfer is mean final-minus-initial accuracy for old domains. These differ from the earlier best-observed-minus-final metric, and should not be mixed numerically.

| Dataset at32 domains | Model | Mean acquisition | Latest acquisition | Forgetting since acquisition (pp) | Signed backward transfer (pp) |
|---|---|---:|---:|---:|---:|
| mnist | 64 neurons × 4 branches | 58.66% | 47.08% | 8.76 | -8.74 |
| mnist | 32 neurons × 8 branches | 58.45% | 43.03% | 6.27 | -6.21 |
| mnist | MLP 256 / SGD | 95.23% | 94.77% | 44.38 | -44.38 |
| mnist | Central memory / 8 slots | 70.93% | 64.58% | 27.63 | -27.63 |
| fashion_mnist | 64 neurons × 4 branches | 43.59% | 36.23% | 5.82 | -5.45 |
| fashion_mnist | 32 neurons × 8 branches | 46.28% | 36.37% | 9.81 | -9.69 |
| fashion_mnist | MLP 256 / SGD | 83.55% | 84.37% | 42.57 | -42.57 |
| fashion_mnist | Central memory / 8 slots | 62.18% | 60.82% | 16.45 | -16.03 |

## Interpretation boundaries

Preserving the first four domains while learning the newest poorly is a stability-plasticity trade-off, not a complete solution. A branch advantage in initial-domain retention does not automatically imply higher average accuracy or a better memory/computation trade-off. The models differ in fixed coded teaching versus learned output heads and in local usage damping versus fixed-rate SGD; these tests do not isolate branch structure from those learning-rule differences. A plasticity-matched ordinary control or stronger regularized MLP is needed before attributing retention uniquely to dendrites.

Three seeds give limited uncertainty; domains within a seed are correlated observations, not independent replicates. No top-conference or biological-realism claim follows from these small benchmarks.

## Reproduction

Script: `bdh_cl/scripts/local_graph_long_stream.py`. Results: `results/local_graph_long_stream_20261013/`. Archived protocol, frozen settings, driver source, per-seed/per-domain metrics and final model states. Historical configurations are not stored or used by the learner. Final checkpoints are offline analysis artifacts.

Python: `/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python`; two CPU threads, cached real MNIST and Fashion MNIST.
