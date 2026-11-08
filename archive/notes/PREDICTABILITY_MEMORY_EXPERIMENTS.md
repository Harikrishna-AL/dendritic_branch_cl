# Positive mechanism test: predictive branch memory

## Why change the memory rule?

The fading activity-trace pilot did not produce useful separation. Here each branch stores a fading estimate of the temporal predictability of its own observed input: lag-one covariance and input energy. Competition selects the branch with the highest normalized correlation. This directly tests whether short-term temporal evidence, rather than activation magnitude, can route an existing predictor without task IDs.

This is intentionally a proof-of-mechanism with two preassigned sensory-channel branches, not a claim that arbitrary task-specific branch structure emerges. The rule is aligned with this task's definition (predict the slowly varying stream). No task identity is provided, but the criterion is engineered. No novelty claim is made.

## Model, controls, and criteria

Two branches each learn one slope and bias, four trainable parameters total. Input is eight observation steps from both noisy AR(1) streams, continuously generated as in the preceding memory pilot. Branch 0 observes channel 0 and branch 1 channel 1 by architecture. Fading covariance and energy update with decay0.95 from observed previous/current pairs; state is read before the current pair is committed. Neither future targets nor task IDs enter state or gate selection.

Compare averaging branches, competition using only the current observed pair, covariance-memory competition, a functionally identical centrally stored/router memory control, and a fixed branch. Central matched memory tests whether housing these statistics in dendrites adds an advantage; it should be identical under this implementation. No task cue, optimizer reset, replay, or parameter protection is used.

One Adam optimizer per run, lr0.01, updates every eight continuous observations. Stronger learning controls: single-regimeA4,096 observations, singleB4,096, and mixed repeatedA/B4,096 (32 blocks of128). Sequential training initially A1,024 then B1,024. Freeze weights and evaluate independent A/B/A/B and reverseB/A/B/A streams,256 observations per visit, without memory resets at switches. Erasure intervention resets memory every step; no output retraining occurs. Scores exclude first32 observations after changes and also retain early error separately.

Preset practical success criteria in protocol: both oldA/newB late MSE<0.30, at least20% reduction against averaging and instantaneous competition, erasure degrades, five new seeds confirm, reverse-order evaluation supports the effect. No constants selected on test data. Compare original seeds0–2 and confirmation seeds3–7. A separate longer-acquisition control uses A4,096 then B4,096 to ensure initial baseline acquisition is not the main explanation. Joint control has the original4,096 total presentations, so it should not be presented as compute-matched to this longer-acquisition sequential run.

## Interpretation boundaries

A gain demonstrates effective temporal memory and selective routing of available independent predictors. It does not establish learned sensory segregation, dendrite-specific superiority over equivalent central memory, scalable arbitrary-task inference, or a paper-ready contribution. The central-memory equality is a scientifically important result, not a control to omit.

Tests verify read-before-write, decay, no target needed for state, exact matched central-control equivalence, and frozen evaluation weights. Sources and protocols are saved in `results/predictability_memory_pilot_20261011`, `results/predictability_memory_confirm_20261011`, and `results/predictability_memory_acquisition_20261011`. Implementation: `bdh_cl/scripts/predictability_memory_pilot.py`.
