# Grant-core experiment: slow weights plus short-term branch memory

## Hypothesis

Branch-local runtime memory that fades with time, combined with branch competition, might produce stable specialization across situations and reduce continual-learning interference. This pilot directly introduces persistent state between observations rather than recomputing context separately for every input window.

## Stream and causality

Sixteen independent continuous stream lanes each contain two AR(1) signals. Regimes A and B swap their autocorrelations (0.9 and 0.1); target is the next observation of the slower signal. Input consists of the last eight noisy observations of both streams. The signal and recent history continue through regime transitions. Environment schedule is A→B→A→B. The network and optimizer receive no regime ID or boundary flag.

A single Adam optimizer (learning rate 0.001) runs through the entire training stream. It updates weights once per eight consecutive time steps. Each prediction reads the prior fast state; after the prediction, state is updated from current branch activation, without targets. Writes are detached local operations, not backpropagation through the runtime memory. State persists through weight updates and regime switches.

## Architecture and initial memory rule

One layer with 32 neurons, four branches per neuron, 16 inputs, and a fixed signed output projection. Every branch has learned slow input weights and bias. Branch short-term state is an EMA of tanh(positive branch activation), decay 0.95 per time step (half-life ~13.5 observations). Prior state biases branch-selection score and modestly scales resulting neuron activity. Competition selects one branch per neuron. Noncompetitive arms average four branches. This tests one bounded activity-trace memory rule, not all fast-weight or dendritic-state mechanisms.

Slow weight timescale is operationally controlled by learning rate and update cadence, not an explicit synaptic-consolidation model. Slow weights are shared and trainable throughout; there is no replay or protection penalty.

## Five comparisons

- slow weights only, no functional runtime state;
- slow weights plus competition, isolating hard pooling without memory;
- branch memory without competition;
- branch memory plus competition;
- neuron-level memory plus competition, with one effective scalar per neuron rather than separate branch traces.

All have identical trainable parameter counts and the same initial weights. Runtime buffers are allocated equally for convenience, but effective independent state is 128 scalars per stream for branch memory versus 32 for neuron memory and zero for slow-only. Report that difference rather than pretending functional memory budgets match. Competition changes pooling as well as sparsity; the slow-only competition arm controls for that effect. Neuron memory is a branch-architecture control, not a fully separate conventional-neuron network.

## Evaluation

At every training-regime checkpoint, freeze weights and run an independent A→B→A→B evaluation stream with runtime state active. Erase state only once at the start of this independent stream, never at regime boundaries. A matched intervention erases memory every time step. Run an additional B→A→B→A stream to check evaluation-order effects. Restore training runtime state afterward. All evaluation is input-only; labels are used by the observer for scoring.

Score first32 observations after switches separately from later observations. Record regime-specific MSE, returns to old regimes, current online prediction loss, and per-neuron branch usage. Branch-profile overlap is a diagnostic of separation, not proof of parameter or functional isolation. Zero baseline is ~1.0; perfect-regime slow-signal predictor is ~0.2, although the model must infer regime from finite history.

Initial run: three paired seeds,256 steps per regime. A separate slow-competition control run shares the initialization/data seeds. Longer acquisition check: all five arms,1,024 steps per regime, three seeds, otherwise identical constants. Settings are not tuned on final evaluation outcomes.

Tests cover causal read-before-write, correct decay, target-free detached writes, neuron-shared versus branch-specific state, memory influence on winners, continuity of observation history across switches, and frozen-weight evaluation restoring training state. Source: `bdh_cl/scripts/branch_short_memory_pilot.py`. Results: `results/branch_short_memory*_20261011`.

## Completed outcome

The short run, competition control, and extended five-arm run completed. Three tests pass. Extended final average live MSE: slow-only0.762; competition-only0.725; branch memory0.798; branch memory+competition0.761; neuron memory+competition0.763. Old regime error remains>1.0. Competitive branch profiles overlap~85%; erasing fast state slightly improves error. This particular fading activity-trace rule does not support the hypothesis yet. Acquisition is also weaker in hard-competition arms, so stronger learnability controls and a richer temporal-memory rule are needed before drawing a broad conclusion.
