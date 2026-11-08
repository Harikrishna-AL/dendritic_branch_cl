# Training runs: cifar10

78 runs. ± is the SD over seed means (orders averaged); intervals are paired 95% t over seeds.

## Metaplasticity sweep (validation)

| α | CIL | TIL | learned → final TIL | forgetting | frozen weights |
|---:|---:|---:|---:|---:|---:|
| 0 | 16.01 ± 0.47 | 58.48 ± 1.58 | 76.9 → 53.1 | 23.78 ± 3.08 | 0% |
| 3 | 16.34 ± 0.14 | 58.32 ± 1.57 | 76.4 → 52.5 | 23.90 ± 4.89 | 1% |
| 10 | 15.97 ± 0.27 | 56.73 ± 1.97 | 75.6 → 50.9 | 24.62 ± 6.20 | 5% |
| 30 | 16.06 ± 0.39 | 57.44 ± 2.02 | 76.8 → 51.7 | 25.04 ± 2.21 | 13% |
| 100 | 14.86 ± 1.75 | 56.30 ± 2.89 | 76.1 → 51.8 | 24.30 ± 1.14 | 34% |
| 300 | 14.74 ± 1.31 | 56.16 ± 0.36 | 75.7 → 51.9 | 23.78 ± 2.93 | 63% |
| 1000 | 15.81 ± 1.44 | 57.69 ± 2.58 | 71.9 → 53.7 | 18.27 ± 4.23 | 87% |
| 3000 | 15.03 ± 2.40 | 55.59 ± 1.69 | 67.7 → 53.3 | 14.42 ± 1.34 | 96% |
| 10000 | 12.71 ± 0.91 | 57.97 ± 3.08 | 64.6 → 58.0 | 6.56 ± 2.25 | 99% |
| 30000 | 13.29 ± 0.92 | 57.69 ± 0.84 | 61.7 → 60.2 | 1.50 ± 0.51 | 100% |

## Test

| Model | CIL | TIL | learned → final TIL (val) | forgetting (val) |
|---|---:|---:|---:|---:|
| branch network, image only | 18.69 ± 0.83 | 66.16 ± 0.68 | 77.4 → 71.4 | 5.96 ± 0.49 |
| branch network, true route | 68.01 ± 1.56 | 73.02 ± 0.41 | 77.4 → 71.4 | 5.96 ± 0.49 |
| plain | 16.03 ± 0.60 | 58.40 ± 1.84 | 76.9 → 53.1 | 23.78 ± 3.08 |
| plain + metaplasticity, α=3 (best val CIL) | 16.34 ± 0.10 | 58.22 ± 2.01 | 76.4 → 52.5 | 23.90 ± 4.89 |
| plain + metaplasticity, α=3 (best val TIL) | 16.34 ± 0.10 | 58.22 ± 2.01 | 76.4 → 52.5 | 23.90 ± 4.89 |

Paired differences, branch − metaplasticity control:

- vs α=3 (best val CIL), CIL, image only: +2.34 [+0.18, +4.51]
- vs α=3 (best val CIL), TIL, image only: +7.94 [+3.69, +12.19]
- vs α=3 (best val CIL), TIL, true route: +14.80 [+9.65, +19.95]
- vs α=3 (best val TIL), CIL, image only: +2.34 [+0.18, +4.51]
- vs α=3 (best val TIL), TIL, image only: +7.94 [+3.69, +12.19]
- vs α=3 (best val TIL), TIL, true route: +14.80 [+9.65, +19.95]

## Architectures (test)

| Model | arch | params | learned CIL (val) | CIL image only | CIL true route | forgetting | routes differ |
|---|---|---:|---:|---:|---:|---:|---|
| branch | conv_b1_wide | 159,690 | 80.9 | 23.09 ± 0.45 | 60.63 ± 5.71 | 9.77 ± 1.55 | 76% |
| branch | conv_b4 | 160,778 | 76.4 | 20.25 ± 0.68 | 69.90 ± 2.29 | 4.42 ± 2.97 | 64%, 71%, 71%, 77% |
| branch | conv_b4_sparse | 160,778 | 77.9 | 18.69 ± 0.83 | 68.01 ± 1.56 | 5.96 ± 0.49 | 61%, 67%, 71%, 78% |
| plain | cnn | 159,146 | 77.5 | 16.03 ± 0.60 | – | 23.78 ± 3.08 | – |
