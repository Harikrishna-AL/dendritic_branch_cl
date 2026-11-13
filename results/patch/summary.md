# Patch vs random input subsets (first layer), Split MNIST test

patch: each first-layer branch reads one 14x14 window (196 pixels, same as random). Mean ± SD over 3 seed means; [a, b] = paired 95% t-interval.

| layers | inputs | image only | true route | forgetting | route mass (mean over layers) | pixel-key router |
|---:|---|---:|---:|---:|---:|---:|
| 1 | random | 69.6 ± 0.4 | 98.5 ± 0.4 | 0.2 ± 0.2 | 31.6 ± 0.4 | 90.9 ± 0.4 |
| 1 | patch | 70.1 ± 1.5 | 98.7 ± 0.3 | 0.3 ± 0.2 | 32.9 ± 0.1 | 91.1 ± 0.3 |
| 2 | random | 50.0 ± 4.5 | 98.0 ± 0.4 | 0.1 ± 1.1 | 29.0 ± 0.1 | – |
| 2 | patch | 59.9 ± 1.6 | 98.8 ± 0.0 | 0.5 ± 0.1 | 30.1 ± 0.2 | 91.1 ± 0.0 |
| 3 | random | 40.3 ± 3.9 | 94.5 ± 5.3 | 0.2 ± 0.8 | 27.3 ± 0.1 | – |
| 3 | patch | 39.3 ± 2.5 | 96.8 ± 2.7 | -0.6 ± 2.1 | 27.9 ± 0.2 | 90.9 ± 0.0 |

- 1 layer(s), patch − random: image only +0.5 [-3.9, +4.8]; true route +0.2 [-1.0, +1.5]; route mass +1.3 [+0.5, +2.0]
- 2 layer(s), patch − random: image only +9.9 [+0.7, +19.1]; true route +0.7 [-0.1, +1.5]; route mass +1.2 [+0.5, +1.8]
- 3 layer(s), patch − random: image only -1.0 [-7.1, +5.1]; true route +2.3 [-14.5, +19.1]; route mass +0.6 [-0.3, +1.5]
