# Upper bound: gate loss with real old images (NOT legal; ceiling only)

cur = gate-loss weight on the current task; old = weight on replayed old images (target = their stored route). Classification loss never sees old images. Split MNIST test, mean ± SD over 3 seed means.

| layers | cur | old | image only (soft) | pick one route by route mass | true route | forgetting | route mass (mean over layers) | soft picks last task |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | 66.3 ± 1.7 | 65.9 ± 1.4 | 96.7 ± 1.8 | 2.1 ± 1.3 | 55.0 ± 0.5 | 0.5 ± 0.5 |
| 1 | 0.1 | 0.1 | 51.6 ± 2.0 | 62.1 ± 3.4 | 98.1 ± 0.2 | 0.5 ± 0.1 | 45.1 ± 0.3 | 64.8 ± 2.5 |
| 1 | 1 | 1 | 70.7 ± 0.1 | 76.8 ± 0.4 | 97.7 ± 0.4 | 0.8 ± 0.3 | 59.6 ± 0.6 | 43.3 ± 0.7 |
| 1 | 1 | 10 | 80.4 ± 1.1 | 78.2 ± 0.8 | 96.9 ± 0.9 | 0.9 ± 0.2 | 68.1 ± 0.1 | 13.3 ± 1.2 |
| 2 | 1 | 1 | 74.7 ± 1.4 | 79.4 ± 0.6 | 96.0 ± 1.2 | 2.7 ± 1.3 | 48.6 ± 0.6 | 33.2 ± 1.1 |
| 3 | 1 | 1 | 71.1 ± 3.0 | 71.0 ± 5.6 | 94.0 ± 3.0 | 4.6 ± 3.3 | 44.7 ± 0.3 | 23.6 ± 1.5 |
