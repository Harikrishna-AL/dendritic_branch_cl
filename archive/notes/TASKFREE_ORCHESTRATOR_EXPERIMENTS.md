# Task-ID-free branch orchestration

## Question

Can branch response selectivity provide internal context, and can local messages use that context to preserve computations without task identity as a network input?

## Observable regime task

Two stationary Gaussian AR(1) signals have autocorrelations 0.9 and 0.1, swapped between two regimes. Each example supplies eight steps of both noisy signals (16 continuous observations). Predict the next observation of the slow signal. No task cue, task ID, feedback label, or reward enters the model input. Training targets provide an ordinary supervised regression loss. Temporal history makes regime statistically inferable, but finite noisy histories do not perfectly identify it. This intentionally changes the ambiguous previous task into a statistically identifiable one.

Generate separate train/validation/test windows using saved seed offsets; 1024/512/1024 examples per regime. Each window starts from the stationary distribution. Train regimes sequentially for 20 epochs each, three paired seeds. Boundaries remain known to the training harness for optimizer resets; this is task-ID-free prediction, not fully boundary-free training. Joint training matches downstream presentations. Settings are fixed before test evaluation; no sweep or test-based selection.

## Mechanisms

Two layers, each with 32 neurons and four branches. Each branch receives incoming activity and produces a ReLU response. Largest branch response chooses the leader. Fixed alternating signed readout never learns. Compare dense activity, direct top16 activity gating, local messages from all branches, a single leader's local message, and a fixed message source. Routed arms recruit 16 neurons and share allocated capacity. Leader and recruitment decisions have hard forwards and soft surrogate gradients; backward paths can update forward-inactive routes.

The initial run uses a ring of 32 neuron groups and eight diffusion rounds. The stricter follow-up uses 128 individual branch compartments and 32 synchronous nearest-neighbor rounds, then aggregates received messages within each neuron and selects top16 neurons. A message's only access per round is its own and adjacent compartments. The branch source is a global argmax and recruitment uses global top-k, so this is not a decentralized election algorithm. It tests local propagation/recruitment rather than a learned semantic message. ReLU responses do not implement an independently modeled inhibitory branch state.

At inference, disabling diffusion preserves leader selection and the top16 budget but changes recruited locations. This is a causal intervention on communication-mediated selection, not a claim that the remaining circuit is otherwise unchanged. Do not interpret a favorable off intervention as successful retention without examining individual regimes.

## Measurements

Record acquisition, final old/new regime MSE, joint MSE, diffusion-removal MSE, winner distributions, and selection profiles. Zero prediction is about MSE 1; an oracle knowing the slow channel predicts 0.9 times its last observation and reaches about 0.2. The oracle baseline receives regime information only in analysis. Winner purity estimated on the same test sample is descriptive and optimistically biased; validate task association with a branch-to-regime mapping built on validation samples and applied to independent test samples.

Tests verify 16-observation inputs, reproducible separate random draws, bounded local recruitment reach, matched routed population size, finite gradients, and fixed readout. No pre-training, parameter-protection penalty, replay, or extra task-ID router is added in these runs.

Results: `results/taskfree_local_pilot_20261010` and `results/taskfree_branch_local_pilot_20261010`. Implementation: `bdh_cl/scripts/taskfree_local_pilot.py`.

## Completed outcome

Both the neuron-group and branch-compartment pilots completed. The branch-level leader gives held-out regime decoding accuracy of 69.4% / 59.1% from layer-1/2 winners; a simple temporal-history heuristic reaches 93.5%. Old-regime error is 1.523 after sequential learning, versus 0.266 immediately after acquisition; zero-output baseline is 1.000. Single-leader and all-branch messages have essentially identical retention. Thus task information in branch responses is measurable, but current leader selection/local recruitment does not produce a successful continual learner. Interpretation and independent validation-to-test decoding diagnostics are saved with the branch-level results. Temporal feature learning should be tested before scaling the communication mechanism.
