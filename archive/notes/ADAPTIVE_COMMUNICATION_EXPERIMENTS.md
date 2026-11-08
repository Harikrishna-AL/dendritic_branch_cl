# Mismatch-triggered extra communication with frozen weights

Reuse the three learned-exchange models from the collective-context experiment, including first-task, final sequential, and joint checkpoints. No training or parameter adaptation occurs. Test fixed total horizons 0, 8, 16, 32, and 64, plus 32 recurrent rounds with neighbor messages replaced by self. This separates longer computation from neighboring information exchange.

Adaptive policy: run eight rounds, compute the mean local neighbor-context disagreement across branches and layers, and run a 32-round total-horizon pass for samples above a fixed threshold. Threshold is the 95th percentile on first-regime training histories using the first checkpoint; freeze it for later checkpoints. No target or regime ID enters the trigger. This is a per-input mismatch proxy with a historical calibration reference, not a demonstrated temporal change-point detector. It does not retain a recurrent context between separate input windows.

All constants and candidate horizons are specified before test evaluation. No horizon is selected based on test error; the adaptive policy always selects either 8 or 32. Official synthetic test draws and validation draws are identical to the source experiment. Observer ridge cue decoders are fitted on validation for each horizon and scored on independent test. Regime labels are observer-only.

The implementation recomputes selected examples from input at 32 rounds. Thus the conceptual policy uses 8/32 rounds, while actual cost on a triggered example is 8+32. Layers are evaluated feedforward at the chosen horizon; it is not simultaneous relaxation of every layer across time. The recurrent update is trained for eight rounds only; testing longer horizons evaluates extrapolation of existing dynamics, not a new error-correction rule. Input evidence seeds the initial context but is not re-injected at every recurrent step.

Tests verify exact agreement with the original eight-round model, zero changes to saved weights, adaptive equivalence to fixed eight/32 rounds at extreme thresholds, and equivalence of self-only inference to removed neighbor exchange. No new training is needed for this diagnostic.

Source: `bdh_cl/scripts/adaptive_communication_pilot.py`. Results: `results/adaptive_communication_pilot_20261010`.

## Completed outcome

All 63 policy/checkpoint evaluations completed. Two tests pass. Extra rounds modestly reduce old-task MSE but fail retention, and task decoding declines. The disagreement trigger has essentially equal familiar/unfamiliar activation after the first task, so it is not a successful change detector. See the saved interpretation for controls and limitations.
