| Method | blocked | geometric | mixed | selected (validation) |
|---|---:|---:|---:|---|
| none | 69.63 ± 0.37 | 69.63 ± 0.37 | 69.63 ± 0.37 | {} |
| class_prior | 73.45 ± 0.92 | 72.50 ± 1.02 | 65.48 ± 0.99 | {'decay': 0.9, 'strength': 0.3} |
| class_prior_gated | 74.45 ± 1.24 | 73.09 ± 1.15 | 69.43 ± 0.41 | {'decay': 0.8, 'strength': 1.0, 'c0': 0.3} |
| branch_state | 83.07 ± 0.71 | 81.12 ± 0.76 | 64.39 ± 0.39 | {'decay': 0.9, 'strength': 0.5} |
| branch_state_gated | 78.65 ± 1.18 | 77.56 ± 1.01 | 69.50 ± 0.37 | {'decay': 0.9, 'strength': 0.9, 'c0': 0.3} |
| branch_state_supervised | 95.65 ± 0.23 | 93.08 ± 0.20 | 69.08 ± 0.25 | {'decay': 0.9, 'strength': 0.5} |

- branch_state_gated − none: blocked +9.02 [+6.98, +11.06]; geometric +7.93 [+6.28, +9.57]; mixed -0.13 [-0.17, -0.08]
- branch_state_gated − class_prior_gated: blocked +4.20 [+3.60, +4.80]; geometric +4.47 [+3.96, +4.98]; mixed +0.08 [-0.04, +0.19]
