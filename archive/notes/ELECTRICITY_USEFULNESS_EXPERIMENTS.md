# Delayed forecast usefulness versus activation predictability

The first electricity pilot found no branch-specific forecasting advantage. This follow-up tests a narrower causal question: does output selection improve when its fading memory tracks actual forecast usefulness instead of response autocorrelation?

## Fixed experiment

Keep1120slow parameters,25 observed inputs, the nonlinear hidden layer,24-hour residual forecasts, Adam.001, update cadence8, and first-layer predictability competition unchanged. Change output routing and training coverage separately:

| Variant | Output cue | Slow training |
|---|---|---|
| predict_winner | Fading response lag/energy | Selected output only |
| predict_all | Same predictability cue | Mean squared error of all outputs |
| quality_winner | Fading forecast error | Selected output only |
| quality_all | Fading forecast error, hard winner | All outputs |
| quality_soft | Fading forecast error, soft weights | All outputs |
| uniform_all | Uniform output average | All outputs |

Forecast-error memory keeps one EMA of squared forecast residual error per output branch and customer, decay.95. It updates only when the real electricity reading arrives24hours later. It uses saved contemporaneous counterfactual predictions, not predictions recomputed after seeing the answer. No task IDs or boundary signals enter. This is causal feedback from observed load, but it is a different rule from the earlier target-free activation-only memory. Frozen modes freeze slow weights, not factual-feedback writes. The quality soft variant uses fixed softmax(-error/.1); it is an ensemble, not sparse output selection.

The quality rule only changes output branches; hidden branch scoring remains predictability-based. All-output training is a meaningful change to learning coverage and can itself improve results. Uniform averaging and predict_all isolate that explanation. For the four all-output variants, routing does not enter the training objective, so identical initializations produce identical slow-weight trajectories. Thus their prediction differences isolate the output readout cue/aggregation, not additional trainable capacity.

Effective routing state: predictability192float/client;quality128;uniform96. Quality's pending24-hour queue additionally stores32 counterfactual outputs/client/hour. Total runtime storage is not exactly matched; the cue can use more pending-feedback storage even with smaller persistent state. Every method uses the same observed input window and timestamped actual load; quality derives an additional historical forecast-error summary from it.

## Development and locked confirmation

Development uses original16 clients, seeds0–2, training2012 and evaluation2013 only. The already-inspected original-client2014 outcomes are never evaluated in this follow-up. Settings are ranked by2013onlineMSE, then recorded before confirmation.

The overall development winner is uniform_all. The best quality-memory candidate is quality_soft, with quality_all retained as a prespecified hard-routing companion. Quality_winner is weaker in development and is not selected for confirmation. All failures are retained.

Confirmation uses a disjoint set of16 customers (eligible positions16:32 among the first64 source columns), selected solely using their first720hours2012. New initialization seeds3–7. Training/calibration is independent on this cohort; no cross-cohort pretrained weights.2013 is an online adaptation prefix and2014the new confirmation period. The cohort is disjoint, but comes from the same dataset and time period, not an independent external benchmark.

Locked confirmation arms: predict_winner,predict_all,quality_soft,quality_all,uniform_all,MLP,GRU. Standard baselines have matching observed inputs and nearly equal slow parameter counts. GRU retains the original one-step truncated-gradient limitation. No settings are retuned using confirmation.

Files: `results/electricity_usefulness_dev_20261011` and `results/electricity_usefulness_confirmation_20261011`. Development selection is in `selection.json`. Three tests check delayed causal feedback, use of stored forecasts after weight changes, and exact equality of slow weights in the all-output cue controls.

This experiment tests forecast usefulness and learning coverage. It does not yet establish reduced forgetting, task separation, or dendrite-specific superiority over generic expert averaging.

## Completed confirmation

Five fresh initializations and disjoint16customer lanes: online2014MSE original0.2245,all-output predictive routing0.2147,soft usefulness0.2099,hard usefulness0.2099,uniform averaging0.2101,MLP0.2035,GRU0.2001. Soft usefulness improves original in5/5 seeds (~6.5%), but its gain over uniform averaging is just0.00018MSE (~0.09%), in4/5 seeds. The exploratory initialization-bootstrap interval for that tiny gain is positive, but five initializations on one cohort are not independent real-world replications. Do not interpret this as a material dendritic advantage.

All dense-output variants have exactly identical2012 slow weights for each confirmation seed; training coverage and readout aggregation explain most of the improvement. Simply replacing selected-output predictability with selected-output error memory failed in development (0.4562 versus0.4554). The first-layer cue remains unchanged. The next mechanism test should assign causal usefulness credit to hidden branches, with a matched training-coverage control, and separately measure retention rather than only prediction error. An untouched new cohort/dataset is needed before selecting further changes; this confirmation cohort2014 has now been inspected.

Three targeted tests passed. Raw metrics include frozen/online, monthly, per-customer and per-initialization outcomes; the weaker candidates remain archived. This follow-up establishes a modest implementation improvement, not a general continual-learning or novelty claim.
