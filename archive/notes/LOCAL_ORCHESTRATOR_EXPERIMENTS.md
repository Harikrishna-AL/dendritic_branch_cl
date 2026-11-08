# Local dendritic orchestration: incremental experiments

## Question
Does a strongest-branch leader improve layer-specific routing beyond input-derived context gating and ordinary local communication? Does prior representation learning help?

## Experiment 1: cheap mechanism pilot
Use five sequential Split MNIST tasks: 0/1, 2/3, 4/5, 6/7, 8/9. Train a two-layer dendritic MLP on 14×14 pixels, 32 neurons with four branches per layer. All arms share initialization, data, minibatch orders, capacity, and downstream optimizer steps. Predict among all ten digits without a task ID at inference or training. No replay or explicit protection.

Compare four mechanisms:

- **Context:** each layer projects its own input to an eight-dimensional context. Branch keys convert it into sigmoid gates.
- **Local:** all active branches inject signed key-vector messages; four nearest-neighbor averaging rounds modulate gates.
- **Winner:** only the largest-activation branch per example per layer injects a message; communication remains local.
- **Fixed:** branch zero injects the message. This controls for having a single message source without choosing the most active branch.

Branches lie on a 128-node ring. Four rounds reach at most four positions in either direction. A winner therefore influences a local region, not the entire layer. Incoming messages can increase or decrease context scores through signed key similarity. Selection is a hard argmax; gradients flow through selected message content, not the discrete choice. Sigmoid gates are soft, and shared parameters remain vulnerable to forgetting. This is an operational first approximation, not a persistent leader for an entire task.

Cross all mechanisms with three initial conditions:

- **Scratch.** No prior training.
- **Autoencoder.** Reconstruct separate first-task training images before sequential learning. Encoder/context/keys learn; classification head remains at matched initialization.
- **Extra supervised.** Same extra examples and optimizer steps as the autoencoder, using their classification labels. This checks whether extra first-task exposure explains a gain, although the objectives and information differ.

Pre-training never sees later digits or test images. It uses 512 distinct first-task examples, disjoint from the downstream first-task examples. This tests narrow pre-training; it does not test broad foundation-model pre-training. Reserve training examples before runs and save indices. Three paired seeds; initial eight-epoch run followed by a 20-epoch acquisition check, with eight pre-training epochs; 200 held-out examples per task. No hyperparameter selection on test scores.

Record the entire stage-by-task accuracy matrix, final average accuracy, average accuracy immediately after each task, forgetting, branch winner concentration, and gate profiles. Report mean and sample standard deviation; three seeds support exploratory conclusions only. First run a tiny pipeline smoke check plus locality, gradient, and inference-determinism checks.

Run:

```sh
/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python3 -m bdh_cl.scripts.local_orchestrator_pilot --out results/local_orchestrator_pilot_20261010 --epochs 8 --pretrain-epochs 8
```

## How results determine the next experiment

1. If immediate task learning is poor, first establish that the small network learns individual tasks and inspect gate amplitude. Do not interpret low forgetting as success when tasks were never learned.
2. If winner improves final accuracy without sacrificing acquisition, confirm with five additional seeds and reverse task order. Compare paired winner-minus-context and winner-minus-fixed differences.
3. If mechanisms perform similarly, inspect whether messages materially change gates. Next test message strength, longer local propagation, and context-compatible rather than raw-activation selection, one factor at a time.
4. If reconstruction pre-training helps, compare frozen versus trainable encoders and add an independent broad pre-training dataset. Keep pre-training exposure explicit and downstream test data excluded.
5. Only after a reproducible routing benefit, introduce selective plasticity/protection and test retention. Then scale to larger images, more tasks, and capacity-matched dense baselines.

Do not launch a large sweep or claim task-specific circuits from accuracy alone. Gate-profile differences, context perturbations, winner swaps, and message removal at inference are follow-up causal checks.

## Completed checks

The one-epoch pipeline smoke run completed. Two mechanistic tests pass: winner messages affect only branches within the configured local radius, and every encoder parameter receives finite gradients in all four arms. Single-task controls at 20 epochs reached 97–100% accuracy for context and winner arms on seed zero. Joint training averaged approximately 92% across five tasks. The initial eight-epoch sequential runs had poor acquisition, so a 20-epoch paired run was launched before interpreting retention. Results and source snapshots are saved under `results/local_orchestrator_pilot_20261010*`.

## Two-task diagnostic completed

The follow-up is implemented in `bdh_cl/scripts/local_orchestrator_diagnostic.py`. It compares context, local, and winner mechanisms from scratch, tests gains 1 and 4, selects on disjoint training-pool validation examples, and then evaluates official test examples. It measures message removal, per-layer gate changes holding layer inputs fixed, winner concentration, and oracle within-pair accuracy. Fifteen runs and three mechanism tests completed. Results are in `results/local_orchestrator_diagnostic_20261010/summary.md`; detailed interpretation is adjacent. All arms lost global old-task accuracy despite retaining high within-pair discrimination, so shared-head/encoder isolation is the next diagnostic before scaling.
