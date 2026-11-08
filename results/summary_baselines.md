# Baselines at matched size

± = SD over 3 seed means (forward + reverse order averaged); [a, b] = paired 95% t-interval.

## HAT vs branch network: mnist

HAT settings (validation, true task given):

| setting | within-pair | all-10 |
|---|---:|---:|
| lr0.01_do0 | 95.97 ± 0.62 | 95.85 ± 0.48 |
| lr0.01_do1 | 92.62 ± 2.46 | 92.59 ± 2.42 |
| lr0.03_do0 | 98.39 ± 0.15 | 98.39 ± 0.15 |
| lr0.03_do1 | 97.85 ± 0.33 | 97.85 ± 0.33 |
| lr0.1_do0 | 98.72 ± 0.20 | 98.72 ± 0.20 |
| lr0.1_do1 | 98.75 ± 0.03 | 98.75 ± 0.03 |
| lr0.3_do0 | 99.04 ± 0.17 | 97.39 ± 3.02 |
| lr0.3_do1 | 98.91 ± 0.09 | 98.91 ± 0.09 |
| lr1_do0 | 89.69 ± 2.24 | 59.42 ± 4.94 |
| lr1_do1 | 91.16 ± 3.53 | 55.41 ± 6.76 |

Selected HAT setting: lr0.3_do0. Test:

| Model | params | all-10, true task | within-pair, true task | all-10, pixel-key router | learned → kept | forgetting |
|---|---:|---:|---:|---:|---:|---:|
| branch network | 134,074 | 98.48 ± 0.44 | 98.50 ± 0.46 | 90.95 ± 0.36 | 98.7 → 98.5 | 0.20 ± 0.15 |
| HAT | 133,537 | 97.30 ± 2.90 | 98.92 ± 0.09 | 91.19 ± 0.11 | 99.1 → 99.0 | 0.03 ± 0.03 |
| branch network, image only (no task, no key) | | 69.63 ± 0.37 | | | | |

Branch − HAT:

- all-10, true task: +1.18 [-4.94, +7.30]
- within-pair, true task: -0.43 [-1.34, +0.49]
- all-10, pixel-key router: -0.24 [-0.87, +0.38]

## HAT vs branch network: cifar10

HAT settings (validation, true task given):

| setting | within-pair | all-10 |
|---|---:|---:|
| lr0.01_do0 | 53.52 ± 1.06 | 53.52 ± 1.06 |
| lr0.01_do1 | 52.12 ± 1.01 | 52.12 ± 1.01 |
| lr0.03_do0 | 58.07 ± 3.03 | 58.07 ± 3.02 |
| lr0.03_do1 | 55.42 ± 2.18 | 55.41 ± 2.18 |
| lr0.1_do0 | 67.02 ± 5.89 | 67.00 ± 5.90 |
| lr0.1_do1 | 57.39 ± 3.90 | 57.36 ± 3.87 |
| lr0.3_do0 | 74.18 ± 2.44 | 74.06 ± 2.43 |
| lr0.3_do1 | 62.57 ± 6.98 | 58.26 ± 9.89 |
| lr1_do0 | 66.41 ± 1.28 | 54.93 ± 6.35 |
| lr1_do1 | 59.91 ± 4.66 | 18.21 ± 4.82 |

Selected HAT setting: lr0.3_do0. Test:

| Model | params | all-10, true task | within-pair, true task | all-10, pixel-key router | learned → kept | forgetting |
|---|---:|---:|---:|---:|---:|---:|
| branch network | 160,778 | 68.01 ± 1.56 | 73.02 ± 0.41 | 32.79 ± 0.41 | 77.4 → 71.4 | 5.96 ± 0.49 |
| HAT | 160,426 | 74.04 ± 2.66 | 74.14 ± 2.71 | 32.75 ± 0.90 | 72.3 → 72.9 | -0.57 ± 1.51 |
| branch network, image only (no task, no key) | | 18.69 ± 0.83 | | | | |

Branch − HAT:

- all-10, true task: -6.04 [-14.49, +2.42]
- within-pair, true task: -1.12 [-7.14, +4.91]
- all-10, pixel-key router: +0.04 [-1.63, +1.70]

## Gate-rule ablation (MNIST, test)

| Training rule | all-10, image only | all-10, pixel-key router | all-10, true route | forgetting | routes differ |
|---|---:|---:|---:|---:|---:|
| error_trace | 69.63 ± 0.37 | 90.95 ± 0.36 | 98.48 ± 0.44 | 0.20 ± 0.15 | 77% |
| fixed_random | 20.15 ± 0.04 | 89.28 ± 1.24 | 20.23 ± 0.01 | 2.67 ± 1.79 | 0% |
| most_active | 20.41 ± 0.19 | 90.32 ± 0.30 | 37.02 ± 2.49 | 1.56 ± 0.61 | 50% |
| random_per_batch | 20.15 ± 0.01 | 89.81 ± 0.41 | 20.19 ± 0.02 | 1.45 ± 0.93 | 75% |
| task_assigned | 43.05 ± 1.19 | 90.90 ± 0.16 | 98.39 ± 0.20 | 0.27 ± 0.11 | 74% |
