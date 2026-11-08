# DCNet-inspired branch competition pilot — Split CIFAR-100

Completed 26 October 2026. This is a raw-pixel branch-model adaptation, not a reproduction of DCNet or a matched SOTA comparison. Six sequential runs: three seeds for each of two fixed loss conditions. No test-set selection.

## Results

Percentages; ± is sample standard deviation across initialization/shuffle seeds with one fixed class order. Incremental average is mean seen-class accuracy after each stage; final accuracy evaluates all 100 classes after stage ten.

| Training | Inference | Incremental average | Final accuracy |
|---|---|---:|---:|
| ioe_dac | head_probability | 20.39 ± 0.12 | 12.64 ± 0.27 |
| ioe_dac | cosine | 21.59 ± 0.20 | 13.91 ± 0.20 |
| ioe | head_probability | 20.46 ± 0.24 | 12.80 ± 0.20 |
| ioe | cosine | 21.43 ± 0.19 | 13.92 ± 0.10 |

Correct-task oracle diagnostic: **48.02% ± 0.32**. This is a separately labelled diagnostic, not no-task-ID accuracy. Autonomous task assignment from the winning head is 16.38% on average. The gap identifies cross-head competition as a major bottleneck; modest within-task accuracy also limits the model.

## Implementation

- Known training boundaries define ten successive class blocks. Only the current block enters updates. No retained old images, individual features, generated examples or joint calibration.
- Raw RGB pixels feed 128 neuron groups with ten tanh branches each. Current-task response energy selects one unused branch per group at the boundary. Train those rows, then commit them. Earlier committed rows stay exactly unchanged; all ten tasks fit without branch reuse.
- Fixed random orthonormal 128-dimensional class targets supervise normalised embeddings. Both arms use current-head cross entropy at temperature 0.1 plus squared distance to the target. `ioe_dac` additionally uses supervised contrastive compaction after epoch one, with temperature adjusted using historical concentration.
- Primary inference concatenates task-head softmax probabilities and selects the highest-scoring class. Secondary inference compares raw cosine scores. Both rules were fixed before testing. Prediction receives only inputs, never true task IDs or labels; every seen class competes.

## Protocol and resources

Official cached CIFAR-100 archive and existing loader; class permutation seed 2026. 45,000 fit images, 5,000 unused validation images, 10,000 official test images. Five epochs per block, Adam 0.001, batch 128, seeds 701/702/703. No pretraining or augmentation. Input preprocessing is fixed x/255 - 0.5, with no fitting or future-task lookahead.

Persistent state: **15,808,808 bytes = 15.809 MB**. Includes the whole preallocated branch bank, targets, routes, flags, concentration statistics and input centering. This is not peak training memory: temporary extracted layer, Adam state, verification snapshots, activations and harness data cost more. Total wall time across six runs, including evaluation: 29.5 seconds.

## Differences from DCNet

DCNet uses HAT masks, an incremental basis generator, its own compensation formula, rotation augmentation, and a ResNet/long-training recipe. This pilot substitutes hard disjoint branch allocation, QR targets and simpler compaction. Routes are selected from current-task responses, not learned HAT masks. Capacity is preallocated for ten tasks; no branch reuse or efficient shared representation is demonstrated. The 128×10 architecture is larger than earlier 24×4 experiments. Its result cannot be compared directly with the published 65.40% DCNet result.

The pilot establishes a concrete task-aware/no-replay/no-inference-ID evaluation, but performs poorly. It does not establish the score of full DCNet, or show that the graph approach cannot work. Compaction gives little primary-score improvement in this recipe.

An earlier exploratory run used full-fit-set centering statistics. Its scores are superseded by the fixed-centering runs reported here; its files remain archived separately.

## Verification and files

Passed checks: orthogonal targets, disjoint allocation, exact preservation of committed rows, and evaluation state invariance. Independently reconstructed all ten routes in all six saved checkpoints and matched vectorised inference. Verified fixed centering in each checkpoint.

[Source](../../bdh_cl/scripts/dcnet_branch_pilot.py) · [Protocol](../../results/dcnet_branch_pilot_fixednorm_20261026/protocol.json) · [Stage results](../../results/dcnet_branch_pilot_fixednorm_20261026/results.json) · [Summary](../../results/dcnet_branch_pilot_fixednorm_20261026/summary.json) · [Verification](../../results/dcnet_branch_pilot_fixednorm_20261026/postrun_verification.json)

[DCNet original paper](https://ijcai.org/proceedings/2025/715).
