# Evidence for the next continual-learning research step

**DER++ is NeurIPS 2020; ER-ACE is ICLR 2022. They are established strong comparators, not the latest papers.** This plot package presents our measured results; it does not establish novelty or a current-SOTA claim.

## Measured strengths and open questions

- Under the matched protocol, the branch model reaches 74.76% autonomous accuracy and 97.39% with oracle route selection, with the same final weights and all ten logits eligible. The 22.63-point gap motivates work on accessing retained capabilities. Oracle access is privileged and is not a deployable score or a guarantee of achievable improvement.
- The current branch model exceeds the tested fine-tuning, SI, online EWC and Numenta variants under this protocol.
- Raw class means reach 82.32% with 0.03144 MB and dominate the current branch model when all measured methods are included. Frozen-feature class means also exceed current branch accuracy. These controls are retained in the complete plots.
- DER++ reaches 88.44% with 200 stored images and 94.52% with 1,000, at 2.285 and 2.951 MB respectively. These replay methods have extra gradient exposures and retain historical examples; their results provide concrete competitive targets.
- The auxiliary objective improves the matched CE-only branch by 2.65 pp on average, but the three-seed descriptive 95% paired interval is [-3.30, 8.61] pp. The specific contribution of the auxiliary loss remains uncertain.

A defensible research statement is: **We investigate whether improved context learning can make retained task-associated capabilities autonomously accessible under fixed memory and without historical-example replay.** Whether the mechanism is novel and whether it improves competitive accuracy require further literature review and experiments.

## Figures

1. [Complete accuracy–memory Pareto plot](01_complete_pareto.png): all matched-protocol baselines, including both class-mean controls and the exploratory Numenta activity variants. The branch model is not on this complete measured frontier. Joint is an offline reference.
2. [All baseline values](02_all_baseline_values.png): complete measured accuracy and retained state, with no class-mean exclusions.
3. [Retained capability versus access](03_retention_access_gap.png): oracle diagnostics are shown separately from autonomous results.
4. [Paired auxiliary-loss ablation](04_auxiliary_loss_ablation.png): per-seed effects and uncertainty.
5. [Online prototype-tracker pilot](05_online_context_pilot.png): original small-network protocol, kept separate.
6. [Native model and benchmark checks](06_protocol_and_native_checks.png): unmatched pilots and one-seed native-recipe sanity check, kept separate.

Each figure has an editable SVG with the same basename. [CSV](matched_baselines.csv), [JSON](matched_baselines.json), [complete frontier](complete_frontier.json), [oracle diagnostics](retention_access_gap.json), and [paired effect](auxiliary_effect.json) contain exact values.

## Interpretation and provenance

All primary scores use the same 55,000 training / 5,000 validation / official 10,000 test split, three seeds, and forward/reverse task orders. SD averages orders within each seed before measuring between-seed variability. Raw class means are deterministic. Joint training is offline and requires dataset access; its displayed state excludes that dataset. Continual models use two current-task epochs, matched preprocessing, Adam, clip5, and separately validation-selected hyperparameters. Hyperparameter trial counts and method-specific deviations are in the original reports; regularization and DER++ receive more search trials than the branch model.

Retained state includes model tensors, masks, prototypes, error traces, route archives, regularizer state, replay images/labels and stored logits where applicable. It excludes Adam, transient gradients/activations, dataset caches and Python/RNG overhead. It is not peak RAM. The original baseline report records additional optimizer state. The log memory axis is explicitly labeled. Frontiers use exact bytes and measured means; segments are visual guides and no statistical significance is inferred from membership.

Sources: [baseline suite](../../results/continual_baseline_memory_20261031/REPORT.md), [matched comparison](../../results/matched_numenta_20261031/REPORT.md), [activity controls](../../results/matched_numenta_activity_20261031/REPORT.md), [strong replay comparison](../../results/strong_replay_20261031/REPORT.md), [online context pilot](../../results/online_prototype_routing_20261031/REPORT.md), [native model pilots](../../results/numenta_native_split_mnist_20261031/REPORT.md), [Permuted MNIST check](../../results/numenta_permuted_mnist_sanity_20261031/REPORT.md).

Paper sources: [DER++ / NeurIPS 2020](https://papers.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html), [ER-ACE / ICLR 2022](https://openreview.net/forum?id=N8MaByOzUfb).

Rebuild: `python reports/continual_learning_evidence_20261031/build_report.py`.
