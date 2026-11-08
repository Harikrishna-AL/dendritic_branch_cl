# Ceiling for image-only task identification, split CIFAR-10

Exploratory, 7 November 2026. One seed (201), no tuning.

The question: how well can *any* single-image route picker identify the task, if it uses the same small network? To answer it, the 159k-parameter tanh CNN from `metaplastic_control.py` is trained **jointly on all 5 tasks at once**. This is not continual learning: every task's data is available throughout. The 5 tasks are fixed class pairs (0,1), (2,3), … (8,9).

Setup: Adam .0005, batch 128, 20 epochs. Task ID is the 5-way task label. The 10-class head's task ID is read as the predicted class // 2. Test accuracy, %:

| Training target | Epoch 2 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|
| 5-way task head | 52.1 | 56.9 | 60.3 | 62.1 |
| 5-way task head, flip+crop augmentation | 50.4 | 54.5 | 58.6 | 63.2 |
| 10-class head → task ID | 52.0 | 59.3 | 62.8 | **65.7** |
| 10-class head, all-10 accuracy | 45.8 | 53.5 | 58.2 | 60.8 |

## Reference points (continual, no stored data)

| Model | Task found | All-10 accuracy |
|---|---:|---:|
| Branch network, image-only | ~19 | 18.4 |
| Pixel PCA key | 43.1 | 33.1 |
| Joint ceiling (this test) | ~65 | ~61 |

## Interpretation

- **The ceiling is about 63–66% task ID and 61% all-10 accuracy.** It is not 90%: one in three CIFAR images can't be placed in its task, even with all the data.
- **About two-thirds of the gap is already closed.** The pixel key reaches 43% of the 65% ceiling without stored data. Getting the rest needs cross-task discrimination, the thing that continual learning without old data can't train.
- **Extra training budget barely moves the ceiling.** At the continual-learning budget (2 epochs of each task, about 2 epochs in total) the joint ceiling is only about 52%.
- **Consequence: stop chasing single-image access on CIFAR with this network.** Report task-given results (73.5% within pair with the true route) and stream results instead.
- **Caveat:** the ceiling belongs to this small network. A larger network or pretrained features would raise it, but the baselines would have to change with it.

Code: `../../bdh_cl/scripts/taskid_ceiling.py`. Raw: `*.json`, `log_*.txt`.
