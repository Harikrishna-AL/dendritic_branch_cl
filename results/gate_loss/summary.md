# Gate loss sweep (Split MNIST, test)

Gate loss = cross-entropy of softmax(|response|) towards the learning branch (BranchNetwork.gate_loss). Route mass = soft weight on the true route branch, image-only inference. Mean ± SD over 3 seed means.

| layers | gate loss | image only | true route | forgetting | route mass |
|---:|---|---:|---:|---:|---:|
| 1 | off | 69.6 ± 0.4 | 98.5 ± 0.4 | 0.2 ± 0.2 | ≈ 31.6 (measured on checkpoints) |
| 1 | 0.01 full | 49.2 ± 2.6 | 98.2 ± 0.5 | 0.1 ± 0.5 | 35.0 ± 0.2 |
| 1 | 0.01 up | 69.2 ± 3.8 | 98.6 ± 0.2 | 0.2 ± 0.1 | 32.3 ± 0.5 |
| 1 | 0.03 full | 38.7 ± 2.1 | 98.2 ± 0.2 | 0.3 ± 0.2 | 36.7 ± 0.1 |
| 1 | 0.03 up | 65.3 ± 6.2 | 98.5 ± 0.3 | 0.2 ± 0.1 | 33.2 ± 0.9 |
| 1 | 0.1 full | 28.1 ± 1.7 | 97.9 ± 0.3 | 0.4 ± 0.4 | 38.8 ± 0.3 |
| 1 | 0.1 up | 55.6 ± 3.0 | 98.1 ± 0.8 | -0.1 ± 0.6 | 34.1 ± 1.1 |
| 1 | 1 full | 20.7 ± 0.1 | 97.0 ± 0.6 | 1.0 ± 0.4 | 41.0 ± 0.7 |
| 1 | 1 up | 19.1 ± 1.3 | 93.8 ± 3.5 | 5.4 ± 4.1 | 39.8 ± 0.3 |
| 2 | off | 50.0 ± 4.5 | 98.0 ± 0.4 | 0.1 ± 1.1 | ≈ 29.0 (measured on checkpoints) |
| 2 | 0.03 full | 37.2 ± 6.7 | 97.7 ± 0.4 | 0.0 ± 1.5 | 31.1 ± 0.3 |
| 2 | 0.03 up | 47.2 ± 3.5 | 97.8 ± 0.5 | -0.1 ± 2.2 | 29.5 ± 0.3 |
| 3 | off | 40.3 ± 3.9 | 94.5 ± 5.3 | 0.2 ± 0.8 | ≈ 27.3 (measured on checkpoints) |
| 3 | 0.03 full | 33.6 ± 3.5 | 97.1 ± 1.5 | -0.7 ± 2.1 | 28.5 ± 0.1 |
| 3 | 0.03 up | 31.7 ± 1.5 | 96.3 ± 2.9 | -0.4 ± 1.2 | 27.4 ± 0.1 |
