# Split CIFAR10 on frozen ImageNet ResNet-18 features

Mean ± SD over 3 seed means (forward + reverse orders). SimpleCIL is deterministic; RanPAC over 3 projection seeds (order-invariant). RanPAC here has no first-session backbone adaptation.

| method | all-10, image only | task right | class right given task | within-pair, task given | forgetting |
|---|---:|---:|---:|---:|---:|
| branch CNN on raw pixels (earlier) | 18.7 ± 0.8 | | | 73.0 ± 0.4 | 6.0 ± 0.5 |
| branch network, soft mixing (no task info) | 77.7 ± 0.5 | 79.0 ± 0.5 | 98.4 ± 0.1 | 96.6 ± 0.3 | 0.3 ± 0.2 |
| branch network + per-task feature key | 79.8 ± 0.3 | 81.8 ± 0.0 | 97.5 ± 0.3 | | |
| plain MLP, matched size | 32.6 ± 0.4 | 33.0 ± 0.4 | 98.8 ± 0.0 | 94.1 ± 0.9 | 3.3 ± 0.6 |
| SimpleCIL (cosine class means) | 70.4 | 73.7 | 95.5 | | 0 by construction |
| RanPAC (random ReLU 5,000-d + ridge) | 88.0 ± 0.1 | 89.2 ± 0.1 | 98.6 ± 0.1 | | 0 by construction |
| linear probe, all tasks jointly (ceiling, not legal) | 87.3 | | | | |

Branch route mass on these features: 31.7 ± 0.2 (chance 25.0; on raw MNIST pixels 31.6).

- branch soft − plain MLP: +45.1 [+44.6, +45.7]
- branch soft − SimpleCIL: 7.3 ± 0.5
- branch + key − RanPAC (difference of means; RanPAC seeds are projection seeds, so unpaired): -8.2
- branch soft − RanPAC (difference of means): -10.2

## Accuracy vs numbers stored

Branch network: trained parameters + route indices (+ keys: per-task mean and 50 principal directions of 512-d features). SimpleCIL: class means. RanPAC/ridge: running sums G and C (the projection is regenerated from a seed).

| method | numbers stored | all-10, image only |
|---|---:|---:|
| branch network, soft mixing | 88,378 + 840 route indices | 77.7 ± 0.5 |
| branch network + per-task keys | 219,193 | 79.8 ± 0.3 |
| SimpleCIL | 5,120 | 70.4 |
| RanPAC, 100-d projection | 11,000 | 72.9 ± 1.1 |
| RanPAC, 200-d projection | 42,000 | 78.7 ± 0.4 |
| RanPAC, 300-d projection | 93,000 | 81.4 ± 0.6 |
| RanPAC, 500-d projection | 255,000 | 83.6 ± 0.2 |
| RanPAC, 1000-d projection | 1,010,000 | 85.6 ± 0.1 |
| RanPAC, 5000-d projection | 25,050,000 | 88.0 ± 0.1 |
| ridge on raw 512-d features | 267,264 | 85.7 |

## Top cross-task confusions (true -> predicted, summed over runs)

- branch, soft: 5->3 (1141), 7->4 (808), 3->5 (789), 2->4 (767), 2->6 (349), 2->0 (336), 3->6 (291), 6->3 (288)
- branch + key: 4->7 (1021), 3->5 (959), 1->9 (683), 5->3 (666), 4->6 (526), 2->6 (469), 7->5 (361), 0->8 (310)
- SimpleCIL: 2->4 (204), 4->7 (201), 0->8 (173), 0->9 (149), 1->9 (147), 5->3 (133), 5->7 (116), 2->6 (112)
- RanPAC: 5->3 (307), 3->5 (274), 2->4 (165), 2->6 (145), 7->4 (141), 9->1 (134), 0->8 (133), 3->6 (129)
