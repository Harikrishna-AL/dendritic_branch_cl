# Electricity predictive-memory pilot — 11 October 2026

## Purpose and data

Test transfer of the existing temporal-predictability branch rule beyond the engineered two-channel synthetic task. Dataset: UCI ElectricityLoadDiagrams20112014, https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 (Trindade,2015;CC BY4.0). The original archive is downloaded, SHA-256 recorded in the protocol, and excluded from Git along with the processed cache.

The pilot uses16 active clients selected from the first64 columns using only the first720hours of2012. Fifteen-minute observations become right-closed/right-labeled hourly means. The first month supplies each client's mean and scale; later periods do not influence selection or normalization. One all-client-zero clock-artifact hour is forward-imputed from the past. Known client lanes maintain separate state, so this is not a test of inferring unknown customer identities.

Every learned model receives the previous24 hourly observations and the observed consumption144hours ago. For a forecast24hours ahead, that latter observation is the matching hour one week before the target. Predictions are residuals added to daily persistence. No calendar, task or boundary IDs enter the model.

## Chronological protocol

First720hours2012 are calibration only. Remaining2012 is trained once in order, with targets delayed24hours.2013 and2014 are evaluated continuously in two modes: frozen weights with runtime state, and online adaptation with one continuing optimizer and target delay. Online2013 is an adaptation prefix before2014, not a held-out static validation period. Frozen2013 is diagnostic validation. All settings were fixed before either period; no validation/test hyperparameter selection occurs.

At forecast time, a training ticket stores observed inputs, current branch routing, or the GRU's previous hidden state. Only when the forecast target becomes observable is the supervised prediction recomputed using current weights and that saved routing/context. Recomputed loss does not write runtime memory. Batch updates occur after eight mature tickets, using Adam.001 and gradient clipping5. There is no replay beyond the required pending-target queue. Pending tickets and incomplete mini-batches at the historical2012 fit endpoint are not carried into independent frozen/online replays; within online2013/2014 there is no year-boundary queue reset. Runtime memory and optimizer are preserved from2012 and across2013/2014.

Forecasting errors are standardized per client using calibration scales, then equally averaged across clients and forecast origins. Cross-year targets are excluded from the yearly report for clean attribution, but online updates and runtime memory continue across years. Normalizations are neither future-fitted nor per-year reset.

Controls: daily/weekly persistence;26-parameter ridge regression fit only2012; an online26-parameter linear predictor; an1081-parameter MLP; an1121-parameter GRU;1120-parameter branch nonlinear/linear, shared context and history-routing models. The GRU has the same25-feature observed window and a10-dimensional persistent hidden state, detached everystep; this is a compact online control with one-step truncated gradients, not a tuned state-of-the-art recurrent baseline.

Branch state is192floats/client; full shared covariance context is1477floats/client plus two scalar mass buffers; GRU state10/client; history routing retains seven25-feature windows/client. The budgets differ materially. Shared context is represented in input coordinates and analytically read through branch weights; it is not merely a central tensor containing per-branch state.

## Results

Means over three initialization seeds,2014:

| Model | Frozen MSE | Online MSE |
|---|---:|---:|
| Daily persistence | .562 | .562 |
| Weekly persistence |1.080 |1.080 |
| Ridge | .458 | .458 |
| Branch nonlinear | .490 | .472 |
| Branch linear | .516 | .546 |
| Shared context | .487 | .471 |
| History routing | .482 | .469 |
| MLP | .443 | .450 |
| GRU | .461 | .444 |
| Online linear predictor | .467 | .555 |

The nonlinear branch model improves on daily persistence but does not beat shared/history routing, MLP, GRU or ridge. The linear change that helped the synthetic task does not transfer. Thus this pilot does not establish a branch-specific real-data advantage. Small differences near shared/history controls should not be overstated with only three seeds.

The forecast task does not specifically reward autocorrelated branch responses: stable/predictable activation need not indicate accurate24h-ahead forecasting. Hard winner routing also restricts effective active capacity. These are hypotheses for the observed limitation, not established causal explanations. Training-ticket routing retention, residual forecasts, historical-window input, horizon and delayed labels are explicit changes from the synthetic setup.

Three meaningful tests pass: observed-feature lag alignment; target delay and unchanged learned weights when only unseen future observations change; training-ticket recomputation without memory writes. Runtime erasure results, monthly curves, per-client errors, checkpoints, full protocol and raw seed outcomes are saved in `results/electricity_memory_pilot_20261011`. Runtime erasure forces default routes in a model trained with selected routes, so it tests dependence rather than superiority to separately trained stateless models.

This experiment is a first subset-scale forecasting transfer pilot, not a full370-client benchmark or proof of task separation/forgetting. The next justified mechanism experiment is to distinguish response predictability from forecast usefulness, selecting candidates on2013 development data and confirming on untouched clients/dates or another dataset.2014 has now been inspected and must not be reused as an untouched final test for newly tuned changes.

## Error concentration audit

The monthly plot shows a large July/August error increase shared by all models. MT_007 contributes about60% of the branch online mean squared error (client MSE4.552), and about63% for MLP/GRU. Median per-client online MSE is0.143 for nonlinear branch,0.127 for MLP and0.124 for GRU, so the baseline advantage is not solely the aggregate outlier. No clients or months were removed after seeing results. First-month normalization amplifies later changes in low-variance clients; a full-scale follow-up should preregister additional scale-robust metrics and audit meter outages/change events before drawing drift conclusions.
