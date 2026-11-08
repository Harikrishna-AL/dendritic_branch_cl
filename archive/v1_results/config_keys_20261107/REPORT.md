# Can a stored key per configuration pick the right branch route?

Exploratory, 7 November 2026. Cheap test, no tuning. MNIST: hybrid5, retrained (image-only 67.91 ± 1.79 reproduces the checkpoints). CIFAR: branched CNN `conv_b4_sparse`, retrained. Seeds 101–103 (MNIST) / 201–203 (CIFAR) × forward/reverse; ± is SD over seed means.

At each task boundary a Gaussian key is stored for that task's configuration. It is fitted on the current task's training images only, with the weights current at that time (no replay). For a test image: score it under all 5 keys, take the most likely task, then predict within that task's two classes using its archived route. "Task found" = task-ID accuracy.

| Key | Contents | MNIST acc / task found | CIFAR acc / task found |
|---|---|---:|---:|
| none: soft magnitude (current) | – | 67.91 / – | 18.79 / – |
| route_diag | responses of the route's branches, diagonal | 29.99 / 30.5 | 20.09 / 25.2 |
| branch_diag | all branch responses, diagonal | 26.41 / 26.8 | 18.60 / 22.8 |
| branch_pca | all branch responses, PCA-50 | 20.89 / 21.1 | 16.28 / 20.0 |
| *branch_diag_final* | *same, keys refit with final weights (not CL-legal)* | *74.52 / 75.8* | *25.80 / 33.4* |
| route_snap_diag | route responses via a frozen network copy per task | 24.69 / 25.7 | 19.47 / 28.2 |
| branch_snap_diag | all branch responses via a frozen copy | 32.56 / 33.5 | 20.48 / 29.4 |
| branch_snap_pca | PCA-50 via a frozen copy | 88.57 / 89.9 | 18.66 / 25.2 |
| randproj_diag | random pixel projections, same size as branch_diag | 79.96 / 81.2 | 29.44 / 37.6 |
| pixel_diag | pixels, diagonal | 56.47 / 57.2 | 28.54 / 36.4 |
| **pixel_pca** | **pixels, PCA-50 (global, SQDA-like per task)** | **90.45 / 91.9** | **33.13 / 43.1** |
| oracle | true task | 97.98 / 100 | 71.25 / 100 |

## Interpretation

- **Branch-local keys fail.** Keys on branch responses are near chance in both benchmarks (MNIST 21–30%, CIFAR 16–20%), worse than the current magnitude rule on MNIST.
- **Main cause: drift.** Branches keep training after their task, so responses recorded at task end no longer match the later network. The same diagonal key refit with final weights reaches 74.5% on MNIST. That refit needs old data, so it is not allowed.
- **Freezing a copy per task doesn't rescue the branch code.** Only the PCA key in a frozen copy works on MNIST (88.6%), and it is still below a plain pixel key (90.5%) while storing a network per task. On CIFAR every branch key stays ≤ 20.5%. Likelihoods from different frozen networks are also not on a common scale, which likely hurts the diagonal keys.
- **The learned branch code is a worse key than random features.** Random pixel projections of the same size (80.0 / 29.4) beat every legal branch key. Branches are trained to discriminate within a task, not to describe what their inputs look like.
- **The winner is a global input key, and it is not dendritic.** A per-task PCA Gaussian on pixels gives 90.5% on MNIST (+22.5 over image-only) and 33.1% on CIFAR (+14.3). It is drift-free because pixels never change. This is essentially SQDA (95.2% per class) applied per task.

## Consequence

The strongest image-only route picker found so far is a non-branch density key; the branches only supply the within-task answer. A defensible system is "global input key picks the route, branches store the routes", with the key output also feeding the stream branch memory (CIFAR: starting signal 43% task-ID instead of ~19%). The branch-specific claim must then rest on retention (routes), not on selection.

Code: `../../bdh_cl/scripts/config_keys.py`, `../../bdh_cl/scripts/config_keys_snapshot.py`. Logs and per-run JSON in this folder.
