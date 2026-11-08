# Branch protection vs random per-synapse metaplasticity (Cho et al. 2025)

Exploratory, 7 November 2026. The question is whether branch-structured protection beats random weight-level protection in a parameter-matched ordinary network.

Control rule, from Cho, Lee, Baek & Paik, *Neural Networks* 107762 (2025):
- Every weight gets a fixed random flexibility f ~ U(0,1).
- Each update is scaled by S = 1 − tanh²(α·(1−f)/f·|Δw|), where Δw = (weight after the previous task) − (initial weight), refreshed at each task boundary.
- Implemented on the Adam step as w ← w_old + S·(w_adam − w_old). No replay, and no task ID during training.

Setup:
- **MNIST:** depth-1 branch network (168×4 branches, 196 pixels each; 134,074 params) vs a tanh MLP of matched width (133,570).
- **CIFAR:** branched CNN `conv_b4_sparse` (160,778) vs an ordinary tanh CNN 32/64/128 → dense 32 → head (159,146).
- **Schedule:** 2 epochs/task, Adam .0005; seeds 101–103 / 201–203 × forward/reverse; ± is SD over seed means; paired intervals are 95% t over 3 seeds.
- **α:** grid 3–30,000, selected on validation CIL. α=0 is the plain network.

Metrics:
- **CIL:** image-only, all 10 classes.
- **TIL:** prediction restricted to the true pair (needs the task ID at test).
- **Learned:** TIL on each task right after learning it (validation).
- **Loss:** learned − final TIL, first 4 tasks (validation).
- **Branch rows:** the true-route row uses the archived route of the true task.

## Test results

| Benchmark | Model | CIL | TIL | Learned → final TIL (val) | Loss |
|---|---|---:|---:|---:|---:|
| MNIST | branch, image-only | **69.63 ± 0.37** | **97.61 ± 0.33** | – | – |
| MNIST | branch, true route | 98.48 ± 0.44 | 98.50 ± 0.46 | 98.8 → 98.6 | 0.2 |
| MNIST | MLP plain | 20.26 ± 0.02 | 83.83 ± 2.32 | 98.9 → 84.2 | 18.4 |
| MNIST | MLP + metaplasticity, α=1000 (best val CIL) | 51.46 ± 2.37 | 93.87 ± 2.27 | 96.0 → 93.6 | 2.9 |
| MNIST | MLP + metaplasticity, α=300 (best val TIL) | 23.90 ± 0.82 | 94.19 ± 1.44 | 98.5 → 94.2 | 5.4 |
| CIFAR | branch, image-only | 18.41 ± 1.20 | 65.44 ± 1.13 | – | – |
| CIFAR | branch, true route | 69.71 ± 4.84 | **73.45 ± 1.64** | 77.5 → 73.9 | 4.5 |
| CIFAR | CNN plain | 16.22 ± 0.06 | 57.79 ± 3.08 | 76.7 → 57.8 | 23.6 |
| CIFAR | CNN + metaplasticity, α=3 (best val CIL) | 16.27 ± 0.09 | 57.61 ± 1.21 | 76.8 → 57.7 | 23.9 |
| CIFAR | CNN + metaplasticity, α=30000 (lowest loss) | – | – | 58.9 → 57.7 | 1.5 |

Paired differences, branch − best metaplastic control (test):
- **MNIST:** CIL +18.17 [+12.75, +23.59]; TIL with true route +4.62 [+0.13, +9.12]; image-only TIL +3.74 [−1.19, +8.67].
- **CIFAR:** CIL +2.15 [−0.65, +4.95]; image-only TIL +7.83 [+5.34, +10.31]; TIL with true route +15.84 [+12.82, +18.85].

Full α sweep (validation): MNIST CIL peaks at α=1000 (50.4%, 92% of weights frozen at the end). Beyond that the network stops learning new tasks: learned TIL falls to 85.5 / 74.7 / 67.6 at α = 3000 / 10000 / 30000. On CIFAR no α improves the final TIL (55.6–58.4% for every α). Large α removes forgetting only by stopping learning: at α=30000, learned 58.9 → final 57.7.

## Interpretation

- **Branches beat random synapse-level protection on both benchmarks.** At matched size, the branch network keeps more of each task: final TIL 98.5 vs 94.2 (MNIST) and 73.5 vs 58.4 (CIFAR), using the best α for each metric.
- **The control is a strong baseline on MNIST, not on CIFAR.** On MNIST it lifts a plain MLP from 20.3 to 51.5% CIL, which is far stronger than the original paper's evaluation suggests. On CIFAR it only trades learning for retention: as α grows, forgetting falls from 24 to 1.5 points, but learning falls from 77 to 59%, so the final accuracy never moves.
- **What the branches add.** They remove the stability–plasticity trade-off that random stability cannot escape: new tasks are learned at full accuracy (98.8 / 77.5%) and old ones are kept (loss 0.2 / 4.5 points). Random flexibility assigns stability without regard to which task a weight serves. Branch routing assigns whole branches to tasks.
- **Limit.** Image-only CIL on CIFAR is still near the control (+2.2, interval crosses zero): the access problem is unchanged.

Notes: a few long-lived CIFAR processes hung inside the BLAS thread pool, a machine issue unrelated to the method. The remaining configurations were therefore run one process per run (`metaplastic_one.py`, 2 threads), with denormal flushing on for runs after the first restart. One configuration re-run in isolation finished normally (36–41 s).

Code: `../../bdh_cl/scripts/metaplastic_control.py`, `metaplastic_one.py`, `metaplastic_summary.py`. Raw: `mnist.json`, `cifar.json`.
