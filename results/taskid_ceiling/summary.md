Joint (non-continual) training of the plain CNN on all CIFAR-10 tasks at once, seed 201. Test accuracy, %.

| Head | epoch 2 | epoch 10 | epoch 20 |
|---|---:|---:|---:|
| 5-way task head | 50.9 | 59.4 | 61.2 |
| 5-way task head, flip + crop | 50.6 | 58.7 | 64.0 |
| 10-class head → task identified | 52.0 | 62.8 | 65.7 |
| 10-class head, all-10 accuracy | 45.8 | 58.2 | 60.8 |
