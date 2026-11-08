# Context-dependent signal prediction

## Task and question

Each example contains two independent latent values A and B, sampled uniformly from −1 to 1. Each value produces four observations with Gaussian noise of standard deviation 0.2. An explicit context cue requests either A or B. The input observations have exactly the same distribution under both contexts, and targets differ. Can layer-local orchestration preserve both context-specific computations through sequential training?

Use two dendritic layers, 32 neurons with four branches each. Each layer derives context from its incoming activity. The output is a fixed signed sum of final neurons: first 16 positive, last 16 negative, divided by four. No output weights or output bias are learned. All variants share this projection. This removes trainable classifier drift as an explanation; it does not remove internal representation interference or make the architecture biologically validated.

## Comparisons

Compare context gating alone, all-branch local messages, and strongest-activation winner messages. Communication uses four nearest-neighbor rounds on the existing ring. Test local/winner gains of 1 and 4. For each mechanism compare scratch initialization and eight-epoch input reconstruction pre-training on separate first-context examples. Pre-training does not see context 1; it adds computation and has no matched extra-training control here.

Three paired seeds; 1,024 training examples per context; 512 validation examples per context from separate generator seeds; 1,024 test examples per context from another seed. Train context 0 then context 1, 20 epochs each. A joint-training control uses both contexts for 20 epochs, matching total downstream example presentations and optimizer updates. Reset Adam at each sequential task, as in earlier pilots; joint training has no reset.

Select gain independently for each mechanism/pre-training condition using mean final sequential validation MSE across seeds, with lower gain breaking ties. Save selection before evaluating official synthetic test draws. This is an exploratory selection, not proof of a robust best gain.

## Measurements and interpretation

Save stage-by-task MSE, R², fixed-weight message-off interventions, cue-swap prediction changes, gate-change magnitudes, and winner histograms. MSE is primary and lower is better. Zero-output baseline should be approximately 1/3. A context-cued mean of the four relevant observations should have MSE approximately 0.01.

First check joint learning and immediate context-0 learning. If both work but sequential context-0 MSE deteriorates, the failure resides in trainable internal computation despite the fixed readout. If winner beats context and local mechanisms on old-context retention without harming new-context learning, confirm with additional seeds and reverse order before scaling. If pre-training helps only joint learning or acquisition, do not claim retention benefits.

Test source: `bdh_cl/scripts/orchestrator_signal_pilot.py`. Results: `results/orchestrator_signal_pilot_20261010`. Tests verify that observations are identical across contexts while targets differ, separate data seeds differ, the readout never changes, and inference interventions restore model settings.

## Outcome

All 30 configurations and joint controls completed. Acquisition and joint MSE approach the observation-average baseline (~0.01), but old-context MSE after sequential training is ~0.55 from scratch and ~0.72 after reconstruction pre-training. Fixed readout does not remove forgetting. Winner messages have very small average gate influence and no clear retention benefit. See the saved summary and interpretation. Context-dependent circuit selection, with an explicit separated-circuit positive control, is the next bounded experiment; larger-scale runs are not justified yet.
