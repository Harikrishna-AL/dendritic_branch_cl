# Branch-specific and shared predictive context

This experiment asks whether the useful temporal information must be stored per branch, and whether the two-layer branch model can improve. It uses the existing continuous two-channel prediction task, 384 slow trainable parameters, both task orders, no task ID or boundary flag, and one Adam optimizer without boundary resets. Weight training sees prediction targets; runtime memory writes see only current inputs or branch responses. Evaluation freezes weights and retains state across switches.

## Controls

- Baseline: each branch stores its previous response, fading lag product, and fading response energy. Hard competition selects four-way first-layer winners and a 32-way output winner.
- Neuron-shared: each first-layer neuron stores input lag/covariance matrices shared by all four of its branches. Output branches share a full eight-dimensional hidden-input context. Each branch analytically reads these matrices through its own weights.
- Layer-shared: all first-layer neurons read a common bank of eight input moment traces with different decay times. The output context is the same as the neuron-shared control.

These are strong shared controls, not an average scalar or a centrally stored branch-specific tensor. They still have branch-specific **readers**. They can reconstruct predictive statistics in the current weight coordinates, whereas local response traces remember responses formed with earlier weights. Thus the experiment compares practical context designs, not solely storage location. The first-layer inputs are identical for each neuron; equal-decay neuron contexts duplicate information.

All designs have the same parameter count and a common state cap of 265 float values per stream. Effective local baseline state is 192 floats, centered state 256, shared state 264 plus one scalar EMA-mass buffer per context. Boolean flags are additional. This is a common upper budget, not exact equality of effective memory. The shared controls receive more state than the local baseline.

## Improvement candidates and selection

Development seeds 0–2, both orders, were used to compare baseline, centered memory, linear activation and softsign activation. Validation streams use seed+2000; test streams use seed+3000. Selection was written before testing. Linear won validation; centered was second. Five fresh seeds 3–7 compare baseline, the locked linear choice, and the prespecified nonlinear centered companion. Confirmation is not used to select another variant.

Centered memory subtracts the squared fading response mean from lag and energy traces before computing their ratio. This is an approximate covariance estimate (one current-response mean, not separate lag means); it costs one extra scalar per branch. Linear changes the first-layer tanh to identity, removing the hidden nonlinear activation; a gain there does not establish improvement of the original nonlinear architecture. Softsign retains a nonlinear activation.

## Trace correction

The initial development shared controls omitted the fading mass multiplying the bias-squared term in the analytic reconstruction. This overcounted the bias before trace burn-in. The branch candidates were unaffected. A separate corrected shared-control run uses EMA mass and returns neutral scores before any observation. The uncorrected development results remain archived and must not be used for the shared comparison. A unit test checks exact reconstruction of fixed-weight branch lag/energy traces from full input moments, including startup and reset.

Results folders:
- `results/shared_context_memory_dev_20261011`: development selection; shared rows superseded.
- `results/shared_context_memory_corrected_20261011`: corrected shared controls.
- `results/branch_memory_confirmation_20261011`: fresh-seed branch confirmation and combined report.
- `results/shared_context_memory_confirmation_20261011`: shared controls on the identical five fresh seeds and both orders.

Lower MSE is better. Main summaries exclude the first 32 observations of each 256-observation evaluation visit; transition errors are preserved separately. This is a small synthetic experiment with an engineered temporal-predictability rule, not a general continual-learning benchmark or evidence of novelty.

The corrected development control-only process completed all training/checkpoints, then its old in-memory CLI failed when ranking an empty set of branch candidates. The CLI now supports control-only runs; `report_shared_context_memory.py` evaluated the existing checkpoints without retraining. This bookkeeping repair does not change model weights or settings.

In this task all neurons see the same two-dimensional input. Therefore the equal-decay neuron-shared traces are duplicates: a single layer-shared trace with branch-specific analytic readers would produce the same result using less state. The multiscale layer control additionally changes temporal horizons and combines their scores by averaging. These constraints limit any anatomical interpretation.

## Outcome

Fresh five-seed confirmation (both orders): baseline average MSE 0.265; validation-selected linear 0.243 (8.3% lower, four of five seed means better); centered nonlinear 0.258 (2.8% lower, two of five better); neuron-shared 0.281; layer-shared 0.287. Old-task MSE improves from 0.320 to 0.285 for linear. The exploratory seed-bootstrap interval for the linear gain includes zero, so this is a modest observed improvement, not a statistically established general gain. Transition MSE is essentially unchanged for linear (0.724 to 0.726), and centered worsens it (0.970).

Local baseline beats each shared control on only two of five seed means; its better aggregate is influenced by failure cases in the shared controls. Thus no consistent dendrite-specific advantage has been shown. Erasing the selected linear memory raises MSE from 0.243 to 0.785, supporting usefulness of persistent context within this model. These observations do not isolate the effect of locality or prove biological necessity.

Five targeted checks passed: exact shared trace reconstruction, reset and causal writes, parameter/state budgets, and the existing deep-memory model checks. The combined report and raw per-seed data preserve failed variants.
