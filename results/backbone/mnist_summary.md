# Split MNIST on frozen ImageNet ResNet-18 features

Mean ± SD over 3 seed means (forward + reverse orders). SimpleCIL is deterministic; RanPAC over 3 projection seeds (order-invariant). RanPAC here has no first-session backbone adaptation.

| method | all-10, image only | task right | class right given task | within-pair, task given | forgetting |
|---|---:|---:|---:|---:|---:|
| branch network on raw pixels (earlier) | 69.6 ± 0.4 | 70.1 ± 0.4 | 99.4 ± 0.1 | 98.5 | 0.2 |
| branch network, soft mixing (no task info) | 88.6 ± 0.4 | 89.0 ± 0.4 | 99.6 ± 0.1 | 99.3 ± 0.2 | 0.0 ± 0.0 |
| branch network + per-task feature key | 94.8 ± 0.1 | 95.3 ± 0.0 | 99.4 ± 0.1 | | |
| plain MLP, matched size | 38.3 ± 0.2 | 38.6 ± 0.2 | 99.2 ± 0.1 | 95.7 ± 0.5 | 5.1 ± 0.7 |
| SimpleCIL (cosine class means) | 77.1 | 79.2 | 97.3 | | 0 by construction |
| RanPAC (random ReLU 5,000-d + ridge) | 97.9 ± 0.0 | 98.1 ± 0.0 | 99.8 ± 0.0 | | 0 by construction |
| linear probe, all tasks jointly (ceiling, not legal) | 97.9 | | | | |

Branch route mass on these features: 33.6 ± 0.1 (chance 25.0; on raw MNIST pixels 31.6).

- branch soft − plain MLP: +50.3 [+48.9, +51.7]
- branch soft − SimpleCIL: 11.5 ± 0.4
- branch + key − RanPAC (difference of means; RanPAC seeds are projection seeds, so unpaired): -3.1
- branch soft − RanPAC (difference of means): -9.3

## Accuracy vs numbers stored

Branch network: trained parameters + route indices (+ keys: per-task mean and 50 principal directions of 512-d features). SimpleCIL: class means. RanPAC/ridge: running sums G and C (the projection is regenerated from a seed).

| method | numbers stored | all-10, image only |
|---|---:|---:|
| branch network, soft mixing | 88,378 + 840 route indices | 88.6 ± 0.4 |
| branch network + per-task keys | 219,193 | 94.8 ± 0.1 |
| SimpleCIL | 5,120 | 77.1 |
| RanPAC, 100-d projection | 11,000 | 87.1 ± 0.3 |
| RanPAC, 200-d projection | 42,000 | 92.0 ± 0.7 |
| RanPAC, 300-d projection | 93,000 | 93.9 ± 0.3 |
| RanPAC, 500-d projection | 255,000 | 95.3 ± 0.3 |
| RanPAC, 1000-d projection | 1,010,000 | 96.7 ± 0.1 |
| RanPAC, 5000-d projection | 25,050,000 | 97.9 ± 0.0 |
| ridge on raw 512-d features | 267,264 | 96.7 |

## Top cross-task confusions (true -> predicted, summed over runs)

- branch, soft: 5->3 (735), 7->1 (550), 7->4 (337), 7->9 (225), 2->5 (189), 2->7 (152), 4->9 (125), 5->6 (112)
- branch + key: 5->3 (458), 7->4 (313), 9->7 (138), 5->2 (136), 1->4 (105), 2->4 (97), 9->4 (93), 8->3 (78)
- SimpleCIL: 2->5 (147), 7->4 (146), 3->5 (143), 7->1 (132), 5->3 (121), 4->1 (113), 5->6 (76), 7->9 (73)
- RanPAC: 5->3 (60), 7->4 (33), 9->4 (32), 7->2 (22), 9->7 (21), 8->2 (16), 6->0 (13), 4->9 (8)
