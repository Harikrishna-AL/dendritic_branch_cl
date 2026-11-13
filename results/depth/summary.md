# Depth diagnostic (Split MNIST, test)

Mean ± SD over 3 seed means (forward + reverse orders averaged). All-10 accuracy unless noted.

## Baseline

| layers | image only (soft) | true route | within-pair, true route |
|---:|---:|---:|---:|
| 1 | 69.6 ± 0.4 | 98.5 ± 0.4 | 98.5 ± 0.5 |
| 2 | 50.0 ± 4.5 | 98.0 ± 0.4 | 98.0 ± 0.4 |
| 3 | 40.3 ± 3.9 | 94.5 ± 5.3 | 97.5 ± 0.7 |

## Layer swap

fix layer i: true route at layer i only, soft elsewhere. soft layer i: soft at layer i only, true route elsewhere.

| layers | layer | fix layer i | soft layer i |
|---:|---:|---:|---:|
| 1 | 1 | 98.5 ± 0.4 | 69.6 ± 0.4 |
| 2 | 1 | 97.0 ± 1.4 | 91.9 ± 1.1 |
| 2 | 2 | 91.9 ± 1.1 | 97.0 ± 1.4 |
| 3 | 1 | 82.3 ± 7.5 | 89.1 ± 2.9 |
| 3 | 2 | 84.6 ± 6.8 | 95.6 ± 3.1 |
| 3 | 3 | 86.9 ± 4.1 | 92.7 ± 4.4 |

## Strongest branch = true route branch (chance 25%)

| layers | layer | upstream soft | upstream true route |
|---:|---:|---:|---:|
| 1 | 1 | 37.2 ± 0.8 | 37.2 ± 0.8 |
| 2 | 1 | 34.6 ± 0.7 | 34.6 ± 0.7 |
| 2 | 2 | 36.9 ± 0.5 | 48.9 ± 0.6 |
| 3 | 1 | 31.3 ± 0.9 | 31.3 ± 0.9 |
| 3 | 2 | 35.5 ± 0.3 | 42.9 ± 0.2 |
| 3 | 3 | 35.2 ± 0.6 | 55.5 ± 1.1 |

## Corrupting one layer's route (true routes elsewhere)

| layers | layer | other task: all-10 | other task: within-pair | random: all-10 | random: within-pair |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.3 ± 1.3 | 92.3 ± 1.3 | 41.1 ± 8.4 | 96.0 ± 0.3 |
| 2 | 1 | 86.8 ± 4.7 | 87.8 ± 4.7 | 92.8 ± 1.3 | 92.8 ± 1.3 |
| 2 | 2 | 10.9 ± 3.1 | 93.9 ± 4.0 | 80.6 ± 4.7 | 97.1 ± 1.0 |
| 3 | 1 | 87.8 ± 3.1 | 90.8 ± 1.3 | 90.7 ± 2.2 | 90.9 ± 1.7 |
| 3 | 2 | 88.3 ± 3.5 | 93.1 ± 4.7 | 94.4 ± 4.7 | 94.5 ± 4.6 |
| 3 | 3 | 15.7 ± 2.7 | 92.5 ± 3.9 | 89.2 ± 1.6 | 95.9 ± 2.4 |

## Training signal at the end of training

| layers | layer | mean abs dCE/dh (x batch) | median gap between two lowest traces |
|---:|---:|---:|---:|
| 1 | 1 | 0.0012 | 0.0001 |
| 2 | 1 | 0.0033 | 0.0000 |
| 2 | 2 | 0.0033 | 0.0001 |
| 3 | 1 | 0.0047 | 0.0000 |
| 3 | 2 | 0.0048 | 0.0000 |
| 3 | 3 | 0.0047 | 0.0001 |

## Fix 1: sharper soft mixing softmax(|r|/T); T = 0 is hard argmax

| layers | T=1 | T=0.5 | T=0.25 | T=0.1 | T=0.05 | T=0 | T picked on val → test |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 69.6 ± 0.4 | 69.1 ± 0.3 | 68.0 ± 0.4 | 66.9 ± 0.5 | 66.5 ± 0.5 | 66.0 ± 0.4 | T=1: 69.6 ± 0.4 |
| 2 | 50.0 ± 4.5 | 50.8 ± 2.7 | 49.2 ± 2.7 | 47.5 ± 2.4 | 46.9 ± 2.2 | 46.2 ± 2.0 | T=0.5: 50.8 ± 2.7 |
| 3 | 40.3 ± 3.9 | 41.6 ± 5.6 | 40.1 ± 6.6 | 38.9 ± 6.1 | 38.3 ± 6.0 | 37.6 ± 5.5 | T=0.5: 41.6 ± 5.6 |

## Fix 2: one stored route for the whole network

agree_all: route whose branches the network itself prefers (mean log soft weight of the route branch, all layers). agree_layer1: same, first layer only. confidence: route with the most confident within-pair prediction.

| layers | method | all-10 | task picked correctly |
|---:|---|---:|---:|
| 1 | agree_all | 58.6 ± 1.0 | 58.9 ± 1.1 |
| 1 | agree_layer1 | 58.6 ± 1.0 | 58.9 ± 1.1 |
| 1 | confidence | 57.5 ± 2.1 | 57.5 ± 2.1 |
| 2 | agree_all | 49.6 ± 2.3 | 49.8 ± 2.1 |
| 2 | agree_layer1 | 49.7 ± 1.6 | 50.1 ± 1.4 |
| 2 | confidence | 52.5 ± 1.7 | 52.6 ± 1.6 |
| 3 | agree_all | 49.0 ± 6.6 | 49.5 ± 6.8 |
| 3 | agree_layer1 | 45.9 ± 1.9 | 46.4 ± 1.5 |
| 3 | confidence | 49.3 ± 4.2 | 49.3 ± 4.2 |
