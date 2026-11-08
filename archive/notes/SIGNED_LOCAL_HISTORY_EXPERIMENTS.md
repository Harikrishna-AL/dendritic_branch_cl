# Faithful history-only, signed local communication test

The previous task-free runs used local messages to choose hard spatial masks. This follow-up models signed excitation/inhibition directly and removes task-boundary resets from the learner.

## The four requested conditions

1. **Inputs plus recent history, no task identity:** each input contains eight observations of two noisy AR(1) streams, flattened to 16 continuous values. Regimes differ in which stream has persistent dynamics; no cue, task ID, label history, or boundary flag enters the network. Regression targets are supplied only to the training loss. Test prediction receives observations only.
2. **Orchestrator selected from branch responses:** each of two layers has 128 branches (32 neurons × four branches). Signed response is tanh of branch input. The largest absolute response selects one branch as leader; this permits a strongly suppressed branch to lead. A hard forward election uses a softmax straight-through gradient at temperature 0.2.
3. **Actual local recruitment and inhibition:** the selected branch injects signed response times a learned four-dimensional branch key. Thirty-two synchronous nearest-neighbor averaging rounds propagate this vector on a 128-branch ring. Each receiver's key compatibility yields a signed drive, shifting its sigmoid gate above or below the neutral value 0.5. Gates multiply feedforward branch activity. Single-source dilution gain is fixed at 16 and sigmoid drive gain at four. This is signed modulatory inhibition, not a biophysical inhibitory-cell simulation. Election is global, while propagation is local.
4. **Task identities for observer analysis only:** a single Adam optimizer consumes a continuous x/y stream without reset or task signals. The environment generator switches regimes; an observer records checkpoints at fixed step counts and measures old/new error. Task labels build a diagnostic branch-to-regime mapping on validation and score it on independent test windows. No such mapping enters routing or the training objective.

## Controls and scope

Compare dense neutral gates, independent signed branch gates, messages from all branches, messages from a response-selected leader, and a fixed leader. All allocate the same parameters and use the same fixed output projection; effective activity differs. Joint learning matches total examples and optimizer updates. Same data draws and seeds as the preceding task-free experiment: 1,024 training, 512 validation, and 1,024 test windows per regime; three paired seeds, 20 epochs per regime. No pre-training, replay, protection objective, or hyperparameter search. No test feedback or adaptation.

The leader carries a learned key vector plus its signed response, not a supplied task representation. Source location and message content depend on branch responses. Top-down feedback is absent; this tests whether bottom-up history and lateral interaction suffice. A task-persistent leader is not enforced. Shared trainable weights can still change outside the strongly modulated region; activity gating is not guaranteed parameter isolation.

## Measurements

Task acquisition and retention MSE; joint MSE; message-removal intervention on learned weights; excited/inhibited branch fractions; prediction changes; held-out winner-to-regime decoding at both layers. Fixed readout cannot drift. Zero-output MSE is about 1.000, and the oracle slow-channel predictor is about 0.199; a history-only heuristic is about 0.295 on the identical test draws. Noise and finite histories mean inferred regime is imperfect.

Tests check signed excitation and inhibition, bounded message influence, suppressed-branch election, removal of messages, observation/target-only stream batches, finite gradients, fixed readout, and observer checkpoints. Source: `bdh_cl/scripts/signed_local_history_pilot.py`. Results: `results/signed_local_history_pilot_20261010`.

## Outcome

The test completed: 15 configurations plus their joint controls. Signed excitation/inhibition and continuous optimizer behavior pass mechanistic checks. Winner regime decoding reaches 72.0% / 59.4% at layers 1/2, but old-task MSE rises from 0.228 to 1.786 for leader messaging (dense 1.729), so retention failed. See `results/signed_local_history_pilot_20261010/summary.md` and `interpretation.md` for the full comparison and limitations.
