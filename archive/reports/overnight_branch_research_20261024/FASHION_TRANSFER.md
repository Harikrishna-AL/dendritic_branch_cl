# Locked transfer to Split Fashion MNIST

Five fresh seeds 610–614, both orders, no Fashion-specific tuning. All settings are transferred from MNIST. The backbone learns sequentially from current labels; association memories receive only current inputs/activity. All automatic outputs compete across ten classes.

| Method | Dimension | Accuracy | Seed SD | Extra memory |
|---|---:|---:|---:|---:|
| no_message | 16 | 55.48% | 1.00 pp | 27,200 B |
| learned_pmi | 16 | 52.02% | 2.07 pp | 29,760 B |
| shuffled_pmi | 16 | 51.37% | 2.28 pp | 29,760 B |
| no_message | 64 | 58.18% | 1.88 pp | 104,000 B |
| learned_pmi | 64 | 58.13% | 2.24 pp | 106,560 B |
| shuffled_pmi | 64 | 56.10% | 1.77 pp | 106,560 B |
| original_local | — | 44.66% | 3.37 pp | 26,000 B |
| global_ridge | — | 67.67% | 1.82 pp | 41,984 B |
| direct_ridge | — | 76.66% | 0.00 pp | 18,944 B |

Learned messages minus no-message,64D: -0.05 pp (95% paired-seed t interval -0.55 to +0.45).

No evidence of transfer of the MNIST communication gain. The 16D communication condition also deteriorates. A source-trained fixed protocol may be suboptimal on Fashion; this study deliberately does not tune it and does not establish an intrinsic impossibility. No external route bank, task-ID or old-label revisits enter automatic predictions. The separately computed historical oracle is task-assisted.

[Locked protocol](../../results/overnight_branch_research_20261024/pmi_fashion_transfer/protocol.json), [all90automatic readouts](../../results/overnight_branch_research_20261024/pmi_fashion_transfer/results.json), [saved states](../../results/overnight_branch_research_20261024/pmi_fashion_transfer/memories), [source driver](../../bdh_cl/scripts/overnight_pmi_confirmation.py).

## Class-order detail

| Method | Forward order | Reverse order |
|---|---:|---:|
| no_message | 57.57% | 58.79% |
| learned_pmi | 56.66% | 59.60% |
| global_ridge | 66.49% | 68.85% |
| direct_ridge | 76.66% | 76.66% |

Fashion messages hurt forward order and help reverse order; averaging orders does not establish a robust communication benefit. Both orders were fixed before confirmation.
