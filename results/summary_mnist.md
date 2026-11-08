# Training runs: mnist

90 runs. ± is the SD over seed means (orders averaged); intervals are paired 95% t over seeds.

## Metaplasticity sweep (validation)

| α | CIL | TIL | learned → final TIL | forgetting | frozen weights |
|---:|---:|---:|---:|---:|---:|
| 0 | 19.81 ± 0.02 | 84.21 ± 2.39 | 98.9 → 80.5 | 18.41 ± 2.98 | 0% |
| 3 | 19.79 ± 0.01 | 84.25 ± 4.20 | 98.9 → 80.6 | 18.34 ± 5.31 | 2% |
| 10 | 19.77 ± 0.01 | 85.00 ± 2.03 | 98.9 → 81.5 | 17.38 ± 2.59 | 5% |
| 30 | 19.77 ± 0.03 | 88.16 ± 2.84 | 98.9 → 85.5 | 13.39 ± 3.64 | 15% |
| 100 | 20.01 ± 0.18 | 90.03 ± 0.96 | 98.9 → 87.9 | 10.99 ± 1.28 | 40% |
| 300 | 23.20 ± 0.75 | 94.16 ± 1.75 | 98.6 → 93.2 | 5.42 ± 2.12 | 71% |
| 1000 | 50.39 ± 1.79 | 93.65 ± 2.26 | 97.3 → 94.4 | 2.92 ± 1.12 | 93% |
| 3000 | 40.45 ± 2.22 | 84.86 ± 4.73 | 88.5 → 87.8 | 0.74 ± 0.62 | 98% |
| 10000 | 24.92 ± 1.08 | 74.21 ± 6.74 | 76.3 → 75.7 | 0.66 ± 1.20 | 99% |
| 30000 | 19.83 ± 0.02 | 66.95 ± 3.61 | 70.4 → 69.6 | 0.77 ± 1.11 | 100% |

## Test

| Model | CIL | TIL | learned → final TIL (val) | forgetting (val) |
|---|---:|---:|---:|---:|
| branch network, image only | 69.63 ± 0.37 | 97.61 ± 0.33 | 98.7 → 98.5 | 0.20 ± 0.15 |
| branch network, true route | 98.48 ± 0.44 | 98.50 ± 0.46 | 98.7 → 98.5 | 0.20 ± 0.15 |
| plain | 20.26 ± 0.02 | 83.83 ± 2.32 | 98.9 → 80.5 | 18.41 ± 2.98 |
| plain + metaplasticity, α=1000 (best val CIL) | 51.46 ± 2.37 | 93.87 ± 2.27 | 97.3 → 94.4 | 2.92 ± 1.12 |
| plain + metaplasticity, α=300 (best val TIL) | 23.90 ± 0.82 | 94.19 ± 1.44 | 98.6 → 93.2 | 5.42 ± 2.12 |

Paired differences, branch − metaplasticity control:

- vs α=1000 (best val CIL), CIL, image only: +18.17 [+12.75, +23.59]
- vs α=1000 (best val CIL), TIL, image only: +3.74 [-1.20, +8.67]
- vs α=1000 (best val CIL), TIL, true route: +4.62 [+0.13, +9.12]
- vs α=300 (best val TIL), CIL, image only: +45.72 [+43.21, +48.23]
- vs α=300 (best val TIL), TIL, image only: +3.41 [+0.35, +6.48]
- vs α=300 (best val TIL), TIL, true route: +4.30 [+1.74, +6.87]

## Architectures (test)

| Model | arch | params | learned CIL (val) | CIL image only | CIL true route | forgetting | routes differ |
|---|---|---:|---:|---:|---:|---:|---|
| branch | 1 | 134,074 | 98.8 | 69.63 ± 0.37 | 98.48 ± 0.44 | 0.20 ± 0.15 | 77% |
| branch | 2 | 162,970 | 98.2 | 49.98 ± 4.54 | 98.04 ± 0.37 | 0.10 ± 1.10 | 74%, 78% |
| branch | 3 | 191,866 | 97.3 | 40.28 ± 3.91 | 94.49 ± 5.28 | 0.24 ± 0.76 | 74%, 76%, 79% |
| plain | 1 | 133,570 | 98.9 | 20.26 ± 0.02 | – | 18.41 ± 2.98 | – |
| plain | 2 | 163,095 | 99.2 | 20.31 ± 0.03 | – | 29.17 ± 9.51 | – |
| plain | 3 | 191,825 | 99.3 | 20.28 ± 0.04 | – | 34.43 ± 2.60 | – |
