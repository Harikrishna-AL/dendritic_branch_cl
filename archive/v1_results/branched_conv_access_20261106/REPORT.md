# Branched CNN on Split CIFAR-10: why image-only access fails, and does branch memory help?

Exploratory, 6 November 2026. Model: `conv_b4_sparse` from `branched_conv_cifar_20261106`, retrained with the identical recipe; seeds 201–203 × forward/reverse; ± is SD over seed means; paired intervals are 95% t over 3 seeds. The 6 trained models are saved here (`model_*.pt`).

## A. Why does largest-response selection fail? (test set, single images)

Per layer, "selection = trained branch" is the share of channels whose largest-|response| branch equals the route archived for the image's true task. Chance here is not 25%: routes overlap across tasks, so the fair baseline is how often *another* task's route uses the same branch.

| Layer | Selection = trained branch | Another task uses the same branch (baseline) | Task ID read from this layer's choices (chance 20%) |
|---|---:|---:|---:|
| conv1 | 40.0 | 38.9 | 21.5 |
| conv2 | 34.1 | 32.7 | 17.9 |
| conv3 | 34.0 | 29.1 | 20.6 |
| top | 35.9 | 22.4 | 27.0 |
| all layers voting | – | – | 24.9 ± 1.1 |
| *MNIST hybrid5 (one layer)* | *36.7* | *22.1* | *53.0* |

Partial oracle (true-task route at the listed layers, soft magnitude elsewhere):

| Oracle at | Test accuracy |
|---|---:|
| all layers | 68.01 ± 1.56 |
| top layer only | 63.46 ± 2.85 |
| all but conv1 | 66.60 ± 2.13 |
| conv layers only | 34.79 ± 3.09 |
| conv3 / conv2 / conv1 only | 35.2 / 21.8 / 20.8 |
| none (soft, image-only) | 18.69 ± 0.83 |

Other single-image selectors: hard largest response at every layer 19.79 ± 1.39; per-image pseudo-error (branch that most lowers CE against the model's own prediction) 18.59 ± 0.88. The latter can only confirm the current prediction, so it changes almost nothing.

Is the task identifiable from one image at all? 5-way linear task-ID probe, trained on validation images from all tasks (diagnostic, not CL-legal): 44.86 ± 0.04% on the network's soft-mode features, 38.16 ± 0.22% on raw pixels.

### Interpretation

- **In conv layers, response magnitude carries no task information.** Selection matches the trained branch no more often than another task's route does (40.0 vs 38.9 in conv1). Channels respond to generic edges and textures in every task, so "which branch fires most" is not task-specific. On MNIST, digit shapes drive different pixel subsets, and the same agreement level (36.7 vs 22.1) identifies the task 53% of the time.
- **Only the top branch layer matters for access.** Restoring the correct route there alone recovers 63.5 of the 68.0 oracle points; correct conv routes alone give 34.8. Branch retention lives in the convs, but task access is decided at the top.
- **Single-image task identity is intrinsically weak on these features.** Even a supervised task-ID probe gets only 45%: the task pairs split visually similar classes (cat/dog, automobile/truck, deer/horse). This ceiling holds for these small features only; a stronger encoder trained on all data would do better.

## B. Branch memory on streams (causal, label-free)

Streams as in `three_routes_20261106`: blocked (200-image task runs), geometric (random run length, mean 50), mixed (shuffled). Branch state is applied at every layer (conv1–3 and top). Hyper-parameters are chosen on validation by mean accuracy over all three streams.

| Method | Blocked | Geometric | Mixed | Selected config |
|---|---:|---:|---:|---|
| none (image-only) | 18.69 ± 0.83 | 18.69 ± 0.83 | 18.69 ± 0.83 | – |
| class prior (control) | 18.14 ± 1.22 | 18.13 ± 1.19 | 17.41 ± 0.90 | decay .8, strength .3 |
| pseudo-error branch state | 17.94 ± 1.74 | 17.69 ± 1.64 | 16.18 ± 1.08 | decay .9, strength .5 |
| pseudo-error, persistence-gated | 18.62 ± 0.61 | 18.63 ± 0.61 | 18.52 ± 0.85 | decay .9, strength .5, c0 .5 |
| *true-label branch state (upper bound)* | *66.65 ± 0.92* | *62.53 ± 1.21* | *19.40 ± 0.19* | decay .9, strength .9 |

Gated vs none: −0.07 [−0.79, +0.65] blocked. Gated vs prior: +0.47 [−1.07, +2.02] blocked, +1.12 [+0.84, +1.39] mixed (the prior hurts slightly on mixed streams).

### Interpretation

- **Label-free branch memory does not help on CIFAR.** The pseudo-error state builds on the model's own predictions, which are 19% correct. It reinforces wrong branches, and the persistence gate mostly stays closed because predicted pairs rarely persist. On MNIST the same rule worked because stateless predictions were already 68% correct.
- **The mechanism itself works when the feedback is good.** State built from true-label error reaches 66.65% in blocked streams, nearly the 68.0% oracle, and still 62.5% with random run lengths. Branch state can select the right configuration from a stream; the bottleneck is the quality of the label-free signal. This is the same conclusion as on MNIST (95.5% supervised), now more extreme.
- **Implication.** Without labels, branch memory needs a starting point that is better than chance at recognising the task. On CIFAR this would have to come from a stronger representation, or from occasional label or task-switch feedback. Better selection rules on these features will not close the gap.

Code: `../../bdh_cl/scripts/branched_conv_access.py`. Raw outputs: `static.json`, `streams_validation.json`, `streams_test.json`, `log.txt`.
