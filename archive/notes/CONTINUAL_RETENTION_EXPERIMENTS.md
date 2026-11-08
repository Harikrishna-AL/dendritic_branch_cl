# Direct continual-learning checks — 11 October 2026

## What is being tested

Forgetting is the change in the same older held-out error after learning a newer distribution: `old_error_after_B - old_error_after_A`. Positive means deterioration; negative means backward transfer. New-task learning is measured separately. All retention probes freeze slow weights, so improved returns cannot be attributed to connection-weight relearning during evaluation. Runtime context can adapt using observed inputs and, for usefulness memory, causal delayed forecast feedback. That is short-term learning/reconstruction, not an unchanged full internal state.

## Real electricity audit

Six fixed models, five new seeds8–12,16customer lanes from the previous confirmation cohort. A is training2012 after calibration throughSeptember; A holdout isOctober–December2012. B is analogous newer2013 training after its first720hours; B holdout isOctober–December2013. Customer selection/normalization remains fixed from the first2012month. Holdout targets never train slow weights. Same old/new holdouts are evaluated afterA,afterB andafter additional equal-budget training onA. A->B/repeatA preserves faststate and optimizer; pending horizon-crossing forecasts are excluded around holdout gaps, as explicitly documented in the run protocol.

Matched cold-context probes reset memory only at an independent audit start. Additional carry-state old-return and disabled-error-feedback probes diagnose runtime recovery. These are retrospective probes, not a natural continuously observed A-B-A timeline. No task/customer/boundary labels enter the network.

**Result: no forgetting in any model.** Older MSE improves after newer learning: originalbranch0.2235→0.1820;soft usefulness0.1888→0.1650;uniform0.1889→0.1649;MLP0.1637→0.1553;GRU0.1890→0.1624. New-period performance improves too. This is backward transfer, and does not establish a branch-specific anti-forgetting advantage. The two natural year distributions do not provide a strong conflicting-task challenge in this protocol.

## Controlled recurring regimes

Three additional seeds8–10, both training orders, same fixed competing branch rules adapted to the existing two-channelAR1task. Rho.9/.1 swap by regime; forecast the next slow-channel observation. The stream state continues across4096-step task switches. No task IDs/boundary flags or optimizer/fast-state resets. Targets are supervised environment feedback arriving one step after the forecast; this differs from the24-hour electricity horizon.

Models share nearly equal slow budgets: branch/shared/history384;MLP397;windowGRU361. MLP sees eight observations;GRU is unrolled through those eight observed steps and uses full gradients within the window. Branches see current channels and retain temporal state. Shared covariance routing uses166float/client plus masses, versus192for response-predictability memory or128for usefulness memory; effective memories and pending-feedback stores differ. An equal representation/compute budget claim would be unjustified.

Weights are evaluated on the same held-out A/B/A/Bstream afterA,afterB, andafter additional4096Aobservations. Each audit freezes slowweights; contextual/feedback writes remain causal. Both full-stream error (including transitions) and late error excluding32observations are retained. A joint-exposure control trains8192steps in alternating128-step balanced regimes, with identical hyperparameters and no IDs. Joint exposure changes switch frequency, so it is a learning-capacity control, not an optimal bound.

Initial controlled results show meaningful forgetting in baseline networks and less in branch/context models. The mechanism remains tailored to the engineered predictability task; these are not electricity forecasting gains or proof of general real-world continual learning. The generic shared router and averaging controls limit anatomical claims.

Files:
- `results/electricity_retention_pilot_20261011`: real retention probes and repeatA control.
- `results/usefulness_retention_synthetic_20261011`: controlled models, before/after and moreA.
- `results/usefulness_retention_shared_20261011`: generic shared temporal context control, same seeds/orders/settings.
- `results/usefulness_retention_joint_20261011`: balanced joint-exposure capacity control.

Four targeted tests verify untouched audit weights/live state, withheld target causality, preservation of pending one-step feedback across controlled switches, and equal slow initialization for local/shared routing. Task identities are used only to generate and score the controlled environment, never as network inputs. No settings are selected on these retention results.

## Completed controlled result

Across3fresh seeds andbothorders, oldtask MSEbefore→afterB is original0.265→0.425;soft usefulness0.224→0.307;hard usefulness0.223→0.287;uniform0.228→0.327;generic shared0.266→0.462;history0.221→0.985;MLP0.202→1.690;windowGRU0.198→1.677. Newtask afterB remains0.271/0.265forsoft/hard usefulness, versus0.273uniform,0.293original andabout0.20MLP/GRU. Thus retention gains have a modest new-task accuracy trade-off rather than complete protection.

Balanced joint-exposure meanerror is about0.44MLP,0.42GRU,0.265history and0.27memory variants, supporting substantial sequential interference while also showingarchitecture/context differences. Three seeds are limited; the task is engineered andquality feedback isone-step supervised. This supports the narrow controlled continual-learning claim, not a proven general dendrite-specific advantage. Natural electricity results remain backwardtransferforall. Combined tables and plots are in the electricity retention interpretation report.
