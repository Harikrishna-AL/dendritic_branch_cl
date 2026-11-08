# Evidence-anchored updates trained across horizons

## Objective

The preceding extra-round experiment repeated a module trained only for eight rounds and found no old-task recovery. This pilot changes the recurrent update so observed evidence remains available at every round and trains several horizons. It isolates learned neighboring exchange from equally deep self-only computation.

## Update and comparisons

Each branch projects incoming activity into a four-dimensional evidence vector e. Initialize context c=e. At every round, compute a proposal from learned transformations of e, c, and the mean of adjacent branch contexts, then apply c ← c + 0.25(proposal − c). Tanh bounds the proposal. Evidence is reintroduced at every round; it is not allowed to disappear solely because iterations continue. Each branch reads its final context into a sigmoid activity gate. This is an evidence-conditioned latent residual correction, not a directly observed task-error signal or a proof of predictive-coding equivalence.

Four arms share all allocated parameters, initialization, inputs, and downstream example presentations:

- self-only, always trained for eight rounds;
- neighbor exchange, always trained for eight rounds;
- self-only, training horizon sampled uniformly from 4/8/16/32 per batch;
- neighbor exchange, the same sampled training horizons.

The self-only control supplies its own state in the neighbor input. The same neighbor-transform parameters are trained in every arm. Variable-horizon arms have the same number of optimizer updates but a larger expected communication-compute budget. Compare neighbor versus self within a training schedule to identify an exchange contribution.

## Data and learning

Same two AR(1) regimes and eight-step observation windows as prior pilots. No task ID, explicit context, boundary flag, or target history enters inputs. Prediction targets provide training regression loss only. Three paired seeds, 1,024 training windows per regime, 20 epochs each; one continuous Adam optimizer; no replay, parameter protection, or pre-training. Fixed alternating signed output projection. Joint controls match downstream presentations and steps, but differ in schedule. Validation/test generator seeds and counts remain identical to previous pilots (512/1,024 per regime).

## Inference and trigger

Evaluate frozen weights at 4/8/16/32/64 rounds and an adaptive 8→32 policy. At eight rounds the detector measures mean proposal-state correction magnitude across branches/layers. Freeze a threshold at the 95th percentile of initial-regime training histories under the first checkpoint. No labels enter detector calibration or inference. This is a mismatch proxy to validate, not an assumed successful change detector. Test 64 is outside variable training horizons.

Adaptive inference recomputes selected examples from input at 32 total rounds. Actual implementation therefore costs 8+32 rounds for selected examples. It has no persistent recurrent state between separate windows. Inference contains neither an optimizer nor parameter updates.

Record first/final/joint task MSE, observer cue decoding trained on validation and evaluated on independent test, mismatch magnitude, and trigger rate per regime. The cue decoder is diagnostic and never feeds the model. All settings specified before test; no test-selected best horizon.

Tests verify evidence surviving long horizons, local propagation bounds, finite gradients under variable training, fixed readout, exact adaptive equivalence to selected fixed horizons at extreme thresholds, and unchanged inference parameters. Source: `bdh_cl/scripts/anchored_communication_pilot.py`. Results: `results/anchored_communication_pilot_20261010`.

## Completed outcome

Twelve configurations plus joint controls and 216 checkpoint/policy measurements completed. Two mechanistic tests pass. Anchored variable-horizon neighbor updates have smaller residuals at long horizons and maintain cue decoding near79%, but old-task MSE remains~1.87 and self-only performs similarly. The latent residual trigger does not detect the new regime. See saved interpretation for the limits and next causal comparisons.
