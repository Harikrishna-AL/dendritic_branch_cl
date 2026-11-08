# Matched trainable small-CNN continual-learning comparison

Status: finished; 12 complete runs, 0 failures.

Question: with the same trainable CNN, branch feature layer, losses, augmentations, optimizer and task heads, does saved disjoint branch configuration differ from learned HAT branch gating?

CNNs: 16–32–64 and 32–64–128 channels; three stride-2 3×3 convolutions, ReLU, task masks and global average pooling. No normalization running statistics, external pretraining or frozen shared encoder. Both models have 128 groups × 10 tanh branches after the CNN. Seeds [701, 702, 703]; one fixed class order. 20 representation + 10 classifier epochs per task; batch 64.

Training uses known task boundaries, only current-class data, and no old-example/latent/generated replay. Inference evaluates every learned task configuration and predicts across all seen classes without a supplied true task ID. Official DCNet rotation-ensemble logits are the primary readout; prototype-cosine scores are a predeclared secondary readout, not selected on test accuracy.

| CNN | Method | Seeds finished | Readout | Incremental avg (%) | Final (%) | Oracle diagnostic (%) | Continuation MB |
|---|---|---:|---|---:|---:|---:|---:|
| cnn16 | branch_config | 3 | joint | 11.22 | 5.13 ± 0.09 | 31.32 | 4.098 |
| cnn16 | branch_config | 3 | prototype_cosine | 16.05 | 8.63 ± 0.70 | 42.93 | 4.098 |
| cnn16 | dcnet_hat | 3 | joint | 11.87 | 6.05 ± 0.15 | 34.47 | 4.138 |
| cnn16 | dcnet_hat | 3 | prototype_cosine | 16.17 | 8.63 ± 0.75 | 43.62 | 4.138 |
| cnn32 | branch_config | 3 | joint | 12.34 | 5.83 ± 0.02 | 35.43 | 5.316 |
| cnn32 | branch_config | 3 | prototype_cosine | 18.86 | 10.01 ± 0.41 | 48.27 | 5.316 |
| cnn32 | dcnet_hat | 3 | joint | 13.30 | 6.12 ± 0.21 | 37.80 | 5.355 |
| cnn32 | dcnet_hat | 3 | prototype_cosine | 18.42 | 9.82 ± 0.57 | 48.07 | 5.355 |

## Exact comparison and limits

The branch_config arm saves one unused branch index per group at each known task boundary and freezes its previous branch rows. The dcnet_hat arm learns task-specific sigmoid branch masks and protects connections with HAT. Both use identical CNN HAT masks/protection formulas, task-specific DCNet projection heads, IOE/DAC losses, LARC/SGD, current-task augmentation and classifier training. This extends the branch mechanism into the DCNet training/readout framework; it is not the earlier fixed-code/Adam raw-pixel pilot.

These are small-CNN architectural variants. They do not reproduce published DCNet’s ResNet/700+100-epoch result. Different effective gating and readout-module parameter counts are reported; the CNN, branch-layer dimensions, shared initialization, examples, optimizer and presentation budgets are matched. Hard branch configurations and soft HAT masks have different active-computation patterns. Historical CNN feature protection is HAT-style and selective, not a guarantee that all old feature values remain exactly invariant.

The CNN is trainable during representation learning and fixed during current-task classifier fitting, matching the two-phase protocol. All evaluation is read-only. The oracle columns use true tasks only after scores are computed, for diagnostics; they are not task-ID-free accuracy.

Continuation bytes count actual model tensors, cumulative/update masks, retained class-direction tensors and concentration summaries, excluding datasets/current activations and optimizer workspaces. Full contiguous update masks are used for reliable MPS convolution updates and counted. They could be reconstructed from cumulative gates to reduce memory; the reported footprint is implementation-specific. Process peak RSS includes Python, cached dataset, allocator and runtime; it is not pure model RAM. Each run is a fresh process so peak RSS is attributable to that process. Run wall time and phase times are retained in JSON, not asserted to be equal FLOPs.

Fit/unused-validation/test counts: 45,000/5,000/10,000 on the official cached CIFAR-100 archive; same fixed 10×10 class order as prior experiments. No validation or test selection. Native data-independent DCNet class directions are reused; no pretrained image representation. Model sizes/budget chosen from hardware profiling only.

## Current progress

```json
[]
```

## Failures

```json
[]
```

[Protocol](../../results/cnn_branch_comparison_20261027/protocol.json) · [Results directory](../../results/cnn_branch_comparison_20261027) · [Source](../../bdh_cl/scripts/cnn_branch_comparison.py)
