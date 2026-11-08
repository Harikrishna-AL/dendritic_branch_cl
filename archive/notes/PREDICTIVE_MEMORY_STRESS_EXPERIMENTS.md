# Fixed-model robustness and larger-data options

## Locked stress test

Models: original nonlinear branch memory, validation-selected linear branch model, full-moment neuron-shared context, and eight-observation history-only routing. All have 384 slow parameters. Persistent state counts differ as documented in the shared-context experiment. No task IDs, task boundaries, optimizer resets, or target-dependent fast writes are provided.

Existing checkpoints for seeds 3–7, both 4096-step training orders, are reused for the first three arms. History-only models are trained with the identical original protocol and initial seed. This is an evaluation-only robustness test: weights are frozen during new conditions. It does not establish continual learning under the modified training schedules.

Seven conditions are fixed before outcomes:

| Condition | Visit length | AR gap | Gradual transition |
|---|---:|---:|---:|
| short | 64 | .8 | 0 |
| reference | 256 | .8 | 0 |
| long | 1024 | .8 | 0 |
| gradual32 | 256 | .8 | 32 |
| gradual128 | 256 | .8 | 128 |
| gap04 | 256 | .4 | 0 |
| gap01 | 256 | .1 | 0 |

Each evaluation stream contains eight visits, beginning with the first task the model was trained on. The underlying process continues through boundaries. Memory resets only at an independent evaluation start. Generator seeds are seed+5000 and shared across arms. Longer visits produce longer total streams, so conclusions are within paired conditions.

AR coefficients are .5±gap/2. Gradual switches linearly interpolate both coefficients and the teacher's target-channel mixture; this changes the target smoothly, not merely the input statistics. Ambiguity conditions still ask for the next slow-channel observation. The diagnostic oracle predicts the conditional mean from the true latent current state and environment parameters; it is never available to a model. It helps distinguish routing error from the changing irreducible noise floor.

Metrics include full-stream MSE, first32 post-switch MSE, late MSE, old-task revisit MSE, per16-step transition curves, output-route overlap and route churn. No first32 exclusion in the main score. Output-route overlap is the sum of minimum usage frequencies between regimes, in second-half visits; lower overlap alone is not evidence of a useful separation. Gradual transitions can occupy the second half of a visit and should be interpreted accordingly.

Three checks verify the reference generator matches the previous stream, causal prefixes/finite gradual targets, and scoring of every visit with weights frozen. Raw results retain all outcomes; no candidate selection occurs in this test.

## Larger data

The strongest next dataset fit is **ElectricityLoadDiagrams20112014**: 370 customer consumption series, 140,256 time points at 15-minute resolution, about 52 million scalar measurements. Start with a subset of clients and preserve chronological order. Predict future consumption using one common output format, compare to shared-context and sequence-model baselines, and audit whether branch separation follows real dynamics rather than customer identity. Dataset source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 . The dataset includes zeros before some clients began service and time-change artifacts; account for them explicitly.

For a visual continual-learning benchmark, **CORe50** contains 164,866 RGB-D frames of 50 objects across 11 recording sessions, with temporal coherence and established continual-learning protocols. A low-cost first version could use a frozen pretrained visual encoder shared across all arms and train the same common classifier or predict future embeddings. Session identifiers are for scheduling/analysis only, not model inputs. Future-embedding prediction would be a custom protocol, not the official object-recognition benchmark. Source: https://vlomonaco.github.io/core50/ .

For larger video scale later, **Kinetics-400** has 400 action categories and hundreds of clips per category. Predicting future pretrained embeddings is closer to the temporal mechanism than shuffled action classification. Establish the principle on smaller data first; clip boundaries and chronology need a defined protocol. Source: https://arxiv.org/abs/1705.06950 and https://github.com/cvdfoundation/kinetics-dataset .

Pretraining can reduce visual experiment cost and improve feature quality, but must be identical across arms. State explicitly what the pretrained model has seen; avoid pretraining on the future evaluation split. A frozen encoder experiment tests the branch/context module rather than end-to-end learning stability.

For the ambiguity test, models retain prediction weights learned at rho .9/.1. Changing rho therefore tests both context identification and prediction-coefficient distribution shift. The oracle comparison alone does not disentangle them. A subsequent matched-training experiment would be required for a claim specifically about learning under ambiguous context.

Real forecasting requires strong baselines: last-observation persistence, a linear autoregressor, an MLP with the identical observed window, and a GRU with comparable parameters/state. For video embeddings, test multiple future horizons and compare persistence; one-step embedding similarity alone can reward simply copying the input. Use temporally separated data and train-only normalization.

## Completed outcome

All 40 model/order/seed combinations completed, each scored under seven conditions. Full-stream linear MSE versus history-only: short0.517/0.660; reference0.318/0.638; long0.267/0.635; gradual32 0.313/0.642; gradual128 0.361/0.671; gap.4 0.685/0.923; gap.1 0.990/1.115. These are aggregate means; paired seed results and uncertainty estimates are retained.

Abrupt switch adaptation is a weakness: reference first32 MSE linear0.886/history0.631, despite better overall error. The linear improvement over original is modest and reverses in gap.1 (original0.981, linear0.990). Output-route overlap for linear rises from0.260 in reference to0.752 in gap.1, consistent with less task separation when predictive cues become ambiguous. This is diagnostic association, not proof that overlap causes the error.

The narrow result is that engineered persistent predictive context remains useful in this synthetic family, even when transition errors count. A consistent branch-specific advantage over shared context or a general real-data continual-learning result is still unestablished. An adaptive trace timescale is a motivated next mechanism test; it should receive no oracle change signal.
