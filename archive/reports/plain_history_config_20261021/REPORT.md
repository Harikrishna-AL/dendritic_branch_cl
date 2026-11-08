# Can an ordinary network recover forgotten MNIST skills through historical configurations?

Completed 21 October 2026. **A flat trace-gated MLP, with no dendritic grouping or branch pooling, recovers 96.99% under historical neuron masks at unchanged final weights and classifier. The branch reference recovers 98.06%. An ordinary fully active MLP with passively recorded traces does not recover under the tested masks.** These are task-assisted diagnostics, not automatic retrieval.

## Main confirmation

Means across three fresh initialization seeds, each tested in forward and reverse pair order. The two orders are averaged within seed before calculating seed SD. Every prediction in the historical-configuration column has all ten digit outputs eligible.

| Optimizer | Network | Latest configuration | Historical configuration | Historical old eight digits | Acquisition | Historical seed SD |
|---|---|---:|---:|---:|---:|---:|
| Adam | Ordinary MLP + passive trace | 20.28% | 20.28% | 0.00% | 99.05% | 0.02 pp |
| Adam | Flat MLP + active usefulness trace | 20.29% | **96.99%** | **96.46%** | 98.82% | 0.46 pp |
| Adam | Grouped branch reference | 20.26% | **98.06%** | **97.84%** | 98.83% | 0.19 pp |
| SGD + momentum | Ordinary MLP + passive trace | 20.30% | 20.30% | 0.00% | 99.14% | 0.04 pp |
| SGD + momentum | Flat MLP + active usefulness trace | 20.30% | **62.91%** | **53.60%** | 98.89% | 6.21 pp |
| SGD + momentum | Grouped branch reference | 20.27% | **65.95%** | **57.46%** | 98.66% | 7.62 pp |

Individual Adam flat-trace runs range from 95.08% to 98.38%; grouped runs from 97.37% to 98.54%. Thus the original 98.02% is approximately reproduced by the grouped reference, while the flat version is close, not identical. SGD flat recovery ranges from 53.40% to 72.14%. This is a small MNIST experiment, not optimizer-independent or dataset-independent evidence.

## Exactly what is a configuration?

Both flat networks are ordinary 784 -> 100 tanh -> 10 fully connected MLPs, with 79,510 learned parameters. Every hidden unit has its own ten outgoing weights. There are no four-branch groups, grouped competitions, or pooled neuron outputs.

- **Plain:** all 100 hidden units participate in every training forward. Utility traces are observed passively and cannot change any weight update. Phase-end top-k utility masks are recorded for diagnostic testing.
- **Flat trace:** only the global top 25 utility units participate in each training forward. A phase-end configuration is a 100-bit neuron-selection mask. This modifies training and limits direct gradients to selected units; Adam/SGD momentum may continue moving previously active parameters. It is an ordinary sparse gated MLP with extra trace state, not vanilla dense training.
- **Branches:** the preceding one-layer 25-neuron/four-branch architecture selects one branch per group and pools to a 25-dimensional head. It has 78,760 parameters. Its classifier shares outgoing weights across each group's branches, unlike the flat MLP. Approximately matching parameter count does not match this constraint or every aspect of capacity.

For flat traces, with raw hidden response a and masked response h, utility is:

    utility_i <- .95 utility_i + .05 [-sum_batch(dCE/dh_i * a_i)]

This is the first-order benefit of retaining a response versus zeroing it. The head supplies dCE/dh for inactive inputs too; candidate utilities can therefore be estimated even when their input weights receive no direct gradient. Traces are derived from supervised gradients, not a biologically local or target-free rule. The branch reference uses its existing gradient-estimated alternative loss EMA (.95).

At the end of each current pair, the experimenter saves a configuration. After all five pairs, the true digit pair selects that earlier configuration. **Only the mask/route changes: the hidden weights, output weights, and output bias remain at their final values.** Historical weights are saved separately for audit controls and never enter config-only evaluation. No old image is revisited during training or config-only recovery. Test labels select the oracle configuration and score predictions; no test-time learning occurs.

## Specificity and plain-network controls

Using the cyclically wrong historical configuration gives **0.00%** for both Adam trace-gated networks. Correct configuration selection matters; recovery is not produced by every arbitrary mask.

The plain MLP's validation-selected k was 100 for both optimizers, meaning no masking. To avoid hiding failures at nontrivial masks, a separate fixed-checkpoint audit evaluates all predetermined mask sizes:

| Optimizer | Top 25 historical units | Top 50 | Top 75 | All 100 |
|---|---:|---:|---:|---:|
| Adam plain MLP | 20.05% | 19.86% | 20.01% | 20.28% |
| SGD plain MLP | 12.97% | 12.98% | 12.21% | 20.30% |

These extra test readouts do not select another deployed method or trigger retuning. They show failure of this particular historical-mask family, not impossibility of recovery under every conceivable configuration.

Output masking is a separate, different task-assisted intervention. Restricting the ordinary MLP's final logits to the correct pair gives **68.55% Adam / 63.91% SGD**. Thus the plain network retains some old pair discrimination even though historical hidden masks fail; failure of our masks does not prove total erasure.

For diagnosis only, old weight checkpoints were also restored in the plain MLP:

| Intervention requiring old weights | Adam | SGD |
|---|---:|---:|
| Restore historical hidden weights, keep final head | 21.82% | 20.30% |
| Restore historical head, keep final hidden weights | 49.53% | 32.12% |
| Restore both historical hidden weights and head | 99.05% | 99.14% |

These are **not configuration-only recovery**. They confirm acquisition and show that head changes and hidden/head compatibility both matter. Partial interventions cannot uniquely localize all lost information.

## Protocol and selection

- Real full-resolution MNIST, pixels/255, five sequential digit pairs, two epochs per pair, batch128, clipping5. No pretraining, replay, task-ID input, boundary reset, usage protection or task-specific output heads.
- One continuous Adam or momentum-.9 SGD optimizer per run. Momentum is not reset at task changes.
- Development seed550: stratified 50k fitting / 10k validation from the training split. Candidate Adam rates .001/.003; SGD rates .03/.1/.3. Independently maximize historical-config validation accuracy with acquisition as tiebreak. Plain masks additionally select k25/50/75/100. This tunes a task-assisted recoverability objective, not autonomous accuracy.
- Selected Adam rates: plain .003; flat trace .001; branch .001. Selected SGD rates: plain .1; both gated models .03. These are a finite search, not exhaustive optimizer tuning.
- Fresh confirmation seeds551–553, both class orders, full60k training/full10k official test: 36 confirmation runs, plus30 development candidates. Orders are paired within three seeds, not six independent seeds.
- The two flat variants share initial weights and minibatch order for a given seed/order, but their independently selected rates differ. Branch initialization has different parameter shapes. No claim of a fully isolated single-variable architecture intervention is made.

## State and verification

Each flat MLP has 318,040 parameter bytes plus400 trace bytes, totaling318,440 numeric model bytes. A stored bank of five float32 masks occupies2,000 bytes in the actual artifacts; a bit-packed representation could use63 bytes but was not deployed. The branch model has315,440 numeric model bytes and five int64 route tensors occupy1,000 additional bytes; uint8 encoding could use125 bytes. Scalar metadata, optimizer state, temporary activations, and audit-only historical weight snapshots are excluded from inference-state counts.

Checks verify the utility derivative against finite differences, exact equality of plain versus all-ones-mask forwards, identical plain parameter updates with/without passive trace recording, and equality of masked inference to explicitly selecting hidden weight rows and matching classifier columns. Every config evaluation asserts unchanged final model state. The plain-mask/weight audit restores and checks the original final state after its separate interventions.

## What this answers

**Dendritic grouping is not necessary for near-98% historical-configuration recovery in this setup. An ordinary flat MLP trained with usefulness-based sparse selection can preserve skills that its latest mask fails to access.** Adam does not overwrite those skills completely under that training rule.

**This does not establish the same result for a vanilla dense MLP.** Recording traces passively was insufficient. Active gating changes which parameters train, so the positive result may reflect selective preservation during learning as well as access at evaluation. It does not show that arbitrary computations lost during unrestricted SGD/Adam remain recoverable, nor does it identify address drift as the cause.

Next: confirm with more seeds and an independently tuned cohort; compare identical learning rates and activation budgets; measure actual inactive-row movement under optimizer momentum; and distinguish protection during acquisition from access after training. Autonomous input-driven mask retrieval is a separate subsequent experiment. A broader vanilla-network test would need other carefully bounded historical configs, while excluding old weight restoration and retraining from config-only claims.

## Reproducibility

- [Main driver](../../bdh_cl/scripts/plain_history_config.py), [audit driver](../../bdh_cl/scripts/audit_plain_history_config.py).
- [Protocol](../../results/plain_history_config_20261021/protocol.json), [all validation candidates](../../results/plain_history_config_20261021/selection.json), [36 confirmation records](../../results/plain_history_config_20261021/results.json).
- [Mask and weight audit](../../results/plain_history_config_20261021/plain_mask_and_weight_audit.json), [verification](../../results/plain_history_config_20261021/verification.json), [completion](../../results/plain_history_config_20261021/completion.json).
