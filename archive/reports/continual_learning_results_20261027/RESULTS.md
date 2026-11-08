# Complete local continual-learning results

All 12 matched CNN runs and nine accepted raw-pixel runs completed. Three initialization/shuffle seeds (701–703), one fixed class order. Every final score evaluates all 100 CIFAR classes without a supplied inference task ID. No replay. ± is sample standard deviation across three seeds, not a confidence interval. Cluster jobs are outside this report.

## Matched CNN comparison

20 representation + 10 classifier epochs per task, batch 64. CNN16 channels 16–32–64; CNN32 channels 32–64–128. Three stride-2 convolutions, ReLU and global pooling. CNNs train from scratch with shared HAT protection; they are held fixed during classifier fitting. Both arms use DCNet IOE/DAC, augmentation, optimizer, projection and readout. Shared initialization and training random-stream seeds are matched. This isolates disjoint branch configurations versus learned HAT branch gating within that training framework. It is not the earlier raw-pixel pilot or published ResNet DCNet.

| Architecture | Method | Readout | Incremental avg (%) | Final (%) | Oracle (%) | Routing (%) | Continuation MB |
|---|---|---|---:|---:|---:|---:|---:|
| cnn16 | branch_config | joint | 11.22 ± 0.76 | 5.13 ± 0.09 | 31.32 ± 0.42 | 11.34 ± 0.26 | 4.098 |
| cnn16 | branch_config | prototype_cosine | 16.05 ± 0.55 | 8.63 ± 0.70 | 42.93 ± 1.02 | 14.47 ± 0.57 | 4.098 |
| cnn16 | dcnet_hat | joint | 11.87 ± 0.28 | 6.05 ± 0.15 | 34.47 ± 1.54 | 12.27 ± 0.47 | 4.138 |
| cnn16 | dcnet_hat | prototype_cosine | 16.17 ± 0.84 | 8.63 ± 0.75 | 43.62 ± 1.21 | 15.21 ± 0.92 | 4.138 |
| cnn32 | branch_config | joint | 12.34 ± 0.54 | 5.83 ± 0.02 | 35.43 ± 1.00 | 12.28 ± 0.96 | 5.316 |
| cnn32 | branch_config | prototype_cosine | 18.86 ± 0.58 | 10.01 ± 0.41 | 48.27 ± 0.33 | 15.83 ± 0.32 | 5.316 |
| cnn32 | dcnet_hat | joint | 13.30 ± 0.19 | 6.12 ± 0.21 | 37.80 ± 0.03 | 12.15 ± 0.54 | 5.355 |
| cnn32 | dcnet_hat | prototype_cosine | 18.42 ± 0.63 | 9.82 ± 0.57 | 48.07 ± 0.21 | 16.02 ± 0.76 | 5.355 |

## Acquisition and forgetting

Acquisition averages current-task accuracy immediately after learning each task, with all then-seen classes competing. Forgetting averages best earlier accuracy minus final accuracy over the first nine tasks. These are different measurements from the final oracle diagnostic.

| CNN | Method | Readout | Acquisition (%) | Forgetting (pp) |
|---|---|---|---:|---:|
| cnn16 | branch_config | joint | 13.53 ± 2.54 | 9.34 ± 2.86 |
| cnn16 | branch_config | prototype_cosine | 13.89 ± 1.70 | 5.84 ± 1.12 |
| cnn16 | dcnet_hat | joint | 14.27 ± 0.33 | 9.13 ± 0.44 |
| cnn16 | dcnet_hat | prototype_cosine | 12.93 ± 1.08 | 4.78 ± 0.38 |
| cnn32 | branch_config | joint | 14.17 ± 0.66 | 9.27 ± 0.76 |
| cnn32 | branch_config | prototype_cosine | 14.04 ± 0.25 | 4.49 ± 0.20 |
| cnn32 | dcnet_hat | joint | 17.84 ± 2.01 | 13.03 ± 2.16 |
| cnn32 | dcnet_hat | prototype_cosine | 13.17 ± 1.24 | 3.73 ± 1.18 |

## Resources for the CNN comparison

Continuation state includes model tensors, cumulative/update masks, retained class directions and concentration summaries. Inference tensor bytes are a computed subset for the primary joint readout, not measured GPU memory or a minimal deployment footprint. Peak RSS includes the dataset, Python, allocator and runtime; it is not VRAM. MB/GB use decimal units.

| CNN | Method | Parameters | Continuation MB | Primary inference tensors MB | Mean run minutes | Maximum process RSS GB |
|---|---|---:|---:|---:|---:|---:|
| cnn16 | branch_config | 686,846 | 4.098 | 0.655 | 15.46 | 3.009 |
| cnn16 | dcnet_hat | 699,646 | 4.138 | 0.694 | 15.47 | 2.422 |
| cnn32 | branch_config | 839,550 | 5.316 | 1.266 | 19.22 | 2.079 |
| cnn32 | dcnet_hat | 852,350 | 5.355 | 1.305 | 19.11 | 2.210 |

Total worker wall time for the 12 sequential CNN runs: 3.46 hours. Full contiguous update masks were needed for correct MPS protection; mask reconstruction/compression could reduce memory in another implementation.

## Every CNN seed

| CNN | Method | Seed | Readout | Incremental avg (%) | Final (%) | Oracle (%) |
|---|---|---:|---|---:|---:|---:|
| cnn16 | branch_config | 701 | joint | 12.10 | 5.14 | 31.76 |
| cnn16 | branch_config | 701 | prototype_cosine | 16.48 | 8.38 | 42.43 |
| cnn16 | branch_config | 702 | joint | 10.80 | 5.21 | 30.92 |
| cnn16 | branch_config | 702 | prototype_cosine | 15.43 | 8.10 | 44.11 |
| cnn16 | branch_config | 703 | joint | 10.76 | 5.03 | 31.28 |
| cnn16 | branch_config | 703 | prototype_cosine | 16.25 | 9.42 | 42.26 |
| cnn16 | dcnet_hat | 701 | joint | 11.66 | 5.88 | 32.82 |
| cnn16 | dcnet_hat | 701 | prototype_cosine | 15.65 | 7.88 | 42.26 |
| cnn16 | dcnet_hat | 702 | joint | 12.19 | 6.16 | 35.88 |
| cnn16 | dcnet_hat | 702 | prototype_cosine | 15.71 | 8.62 | 44.56 |
| cnn16 | dcnet_hat | 703 | joint | 11.77 | 6.11 | 34.71 |
| cnn16 | dcnet_hat | 703 | prototype_cosine | 17.14 | 9.38 | 44.04 |
| cnn32 | branch_config | 701 | joint | 12.96 | 5.84 | 35.07 |
| cnn32 | branch_config | 701 | prototype_cosine | 19.38 | 10.22 | 48.61 |
| cnn32 | branch_config | 702 | joint | 11.97 | 5.81 | 34.66 |
| cnn32 | branch_config | 702 | prototype_cosine | 18.24 | 9.53 | 48.23 |
| cnn32 | branch_config | 703 | joint | 12.10 | 5.83 | 36.56 |
| cnn32 | branch_config | 703 | prototype_cosine | 18.96 | 10.27 | 47.96 |
| cnn32 | dcnet_hat | 701 | joint | 13.40 | 6.06 | 37.80 |
| cnn32 | dcnet_hat | 701 | prototype_cosine | 17.79 | 9.43 | 48.15 |
| cnn32 | dcnet_hat | 702 | joint | 13.08 | 5.95 | 37.77 |
| cnn32 | dcnet_hat | 702 | prototype_cosine | 18.41 | 9.55 | 48.23 |
| cnn32 | dcnet_hat | 703 | joint | 13.42 | 6.35 | 37.83 |
| cnn32 | dcnet_hat | 703 | prototype_cosine | 19.04 | 10.47 | 47.83 |

## Earlier raw-pixel experiments — separate protocols

Our raw-pixel pilot used five current-task epochs, fixed orthogonal class targets, Adam, no image augmentation, disjoint hard branch allocation and either probability or cosine competition. IOE-only is its compaction-loss ablation. The raw-pixel DCNet variant used five representation plus five classifier epochs, native DCNet losses/augmentation and learned HAT gates. Both have a 3072→128×10 tanh feature layer, but their full training/readout recipes differ. These are not resource-matched and cannot isolate the effect of adding a CNN.

| Architecture | Method | Readout | Incremental avg (%) | Final (%) | Oracle (%) | Routing (%) | Continuation MB |
|---|---|---|---:|---:|---:|---:|---:|
| raw_128x10 | ioe_dac | head_probability | 20.39 ± 0.12 | 12.64 ± 0.27 | 48.02 ± 0.32 | 16.38 ± 0.43 | 15.809 |
| raw_128x10 | ioe_dac | cosine | 21.59 ± 0.20 | 13.91 ± 0.20 | 48.02 ± 0.32 | 18.53 ± 0.27 | 15.809 |
| raw_128x10 | ioe | head_probability | 20.46 ± 0.24 | 12.80 ± 0.20 | 48.15 ± 0.28 | 16.59 ± 0.36 | 15.809 |
| raw_128x10 | ioe | cosine | 21.43 ± 0.19 | 13.92 ± 0.10 | 48.15 ± 0.28 | 18.48 ± 0.41 | 15.809 |
| raw_128x10 | dcnet_hat | joint | 15.47 ± 0.40 | 8.67 ± 0.37 | 38.48 ± 0.17 | 14.62 ± 0.33 | 19.022 |

The exploratory run using full-dataset centering is superseded and excluded. The ResNet experiment was explicitly stopped before producing a complete score and is not a result.

## Interpretation

- In the matched CNN experiment, DCNet-HAT wins the primary joint readout on both sizes and every seed.
- With prototype cosine, CNN16 is tied; CNN32 gives branch configurations a mean +0.19 percentage points. That advantage occurs in only one of the three paired CNN32 seeds, so it is not a consistent win.
- Larger CNNs improve cosine scores and oracle diagnostics, but automatic task/head selection remains weak: roughly 15–16% routing accuracy against 10% chance. Both representation quality and cross-head calibration/retrieval need improvement.
- The continuation-memory saving is 39,680 bytes (approximately 0.040 MB). There is no demonstrated substantial memory or speed advantage in this implementation.
- Tiny-CNN HAT gates become heavily occupied as tasks accumulate. Per-task encoder weight movement and channel occupancy are recorded in each run’s cnn_updates.json.
- These results do not establish superiority over published DCNet or state of the art. Training budgets, architecture and one class order limit generalisation.

## Files

- [Per-seed metrics CSV](per_seed_results.csv): every completed model/readout and resource measurement.
- [All stage results CSV](all_stage_results.csv): 390 stage/readout records, including per-task accuracies.
- [All results JSON](all_results.json).
- [CNN protocol and live report](../cnn_branch_comparison_20261027/REPORT.md).
- [Raw branch pilot](../dcnet_branch_pilot_20261026/REPORT.md).
- [Raw-pixel DCNet comparison](../dcnet_raw_baseline_20261027/REPORT.md).
