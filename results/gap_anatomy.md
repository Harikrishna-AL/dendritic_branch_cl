# Anatomy of the image-only gap (Split MNIST test)

Mean ± SD over 3 seed means. Probes are linear, trained on 20,000 training images from ALL tasks (not legal; they measure information content).

## 1. Error types

| model | image only | task right | class right, given task right | errors: wrong task = last task | errors: other wrong task | errors: right task, wrong class |
|---|---:|---:|---:|---:|---:|---:|
| default, 1 layer | 69.6 ± 0.4 | 70.1 ± 0.4 | 99.4 ± 0.1 | 7.3 ± 0.7 | 22.6 ± 1.1 | 0.4 ± 0.1 |
| default, 3 layers | 40.3 ± 3.9 | 41.9 ± 4.7 | 96.4 ± 2.7 | 6.7 ± 6.4 | 51.4 ± 9.9 | 1.6 ± 1.4 |
| upper bound, 1 layer (1/10) | 80.4 ± 1.1 | 82.9 ± 0.3 | 97.0 ± 1.5 | 0.7 ± 0.3 | 16.4 ± 0.6 | 2.5 ± 1.3 |
| upper bound, 3 layers (1/1) | 70.7 ± 2.5 | 76.6 ± 0.9 | 92.2 ± 2.6 | 7.4 ± 1.4 | 16.0 ± 2.0 | 5.9 ± 1.9 |

## 2. Most common cross-task confusions (true -> predicted, total over runs)

- default, 1 layer: 9->4 (1773), 0->5 (1329), 8->5 (1264), 4->9 (902), 7->9 (788), 5->3 (745), 8->3 (419), 3->5 (323)  [top-8 lists cover 10235 errors]
- default, 3 layers: 5->8 (2331), 7->9 (2187), 3->8 (2028), 7->3 (1849), 2->8 (1804), 6->2 (1713), 4->9 (1408), 9->3 (1343)  [top-8 lists cover 24449 errors]
- upper bound, 1 layer (1/10): 9->4 (1285), 8->3 (644), 8->5 (585), 9->7 (575), 5->3 (397), 0->6 (230), 1->8 (222), 7->2 (189)  [top-8 lists cover 5187 errors]
- upper bound, 3 layers (1/1): 5->8 (1315), 4->9 (984), 3->8 (686), 7->9 (674), 4->8 (494), 2->8 (447), 8->2 (375), 5->0 (356)  [top-8 lists cover 8092 errors]

## 3. Is the information there? Linear probes (all-task training data)

Pixels: 10-class 91.1, task 87.1.

| model | image only | probe on last-layer output: 10-class | ...: task | probe on all branch responses: 10-class | ...: task |
|---|---:|---:|---:|---:|---:|
| default, 1 layer | 69.6 ± 0.4 | 92.1 ± 0.2 | 89.5 ± 0.1 | 91.7 ± 0.1 | 87.5 ± 0.1 |
| default, 3 layers | 40.3 ± 3.9 | 89.8 ± 0.3 | 87.3 ± 0.2 | 93.8 ± 0.1 | 92.7 ± 0.2 |
| upper bound, 1 layer (1/10) | 80.4 ± 1.1 | 91.8 ± 0.0 | 90.6 ± 0.1 | 91.2 ± 0.1 | 87.5 ± 0.1 |
| upper bound, 3 layers (1/1) | 70.7 ± 2.5 | 89.6 ± 0.3 | 88.9 ± 0.2 | 93.7 ± 0.1 | 92.7 ± 0.5 |
