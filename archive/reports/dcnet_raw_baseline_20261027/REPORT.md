# Raw-pixel DCNet variant versus the branch pilot

Status: complete.

Official source: https://github.com/Tianqi-Wang1/DCNet at commit c3061a0492e3798d5957caa4f30fae501fd4f219.

DCNet budget: 5 representation epochs + 5 OOD-classifier epochs per task, batch 128, device mps. This is not the published 700+100-epoch recipe. Training uses the matched raw-pixel branch layer, official losses, augmentation, HAT and OOD classifier functions. DAC begins at epoch 3; warmup is shortened to one epoch.

Both experiments use the same 45,000 fit images, 10,000 official test images, ten equal class blocks and class permutation. The 5,000 validation images are unused. Task boundaries are known during training; no true task ID is supplied at inference and no old observations are rehearsed. Both feature layers have 3072 inputs and 128×10 tanh branches, with paired first-layer initialization; gates, projections, augmentation and compute differ, so this is a same-data comparison, not a resource-matched experiment.

| Method | Incremental average (%) | Final accuracy (%) | Persistent tensor state |
|---|---:|---:|---:|
| Branch pilot, head probabilities | 20.39 | 12.64 ± 0.27 | 15.81 MB |
| Branch pilot, cosine | 21.59 | 13.91 ± 0.20 | 15.81 MB |
| Raw-pixel DCNet variant, 3 completed seeds | 15.47 | 8.67 ± 0.37 | 19.02 MB |

Progress: {"seed": 703, "task": 10, "phase": "OOD", "epoch": 5}.

## Stage progress (not final benchmark scores)

| Seed | Completed tasks | Latest seen-class accuracy (%) |
|---|---:|---:|
| seed701 | 10 | 8.80 |
| seed702 | 10 | 8.25 |
| seed703 | 10 | 8.96 |

## Verification and portability

Two-task smoke training passed both official phases and HAT updates; independent official cil evaluation matched the input-only prediction wrapper. Inference state remained unchanged and model values were finite. The optimizer synchronization patch was checked against original LARC for multiple steps, clipping on/off, zero norms, weight decay and momentum.

Patches select MPS instead of CUDA, remap CUDA-saved tensor prototypes onto the selected device, use FP32 crop parameters, omit unused faiss/diffdist imports, and replace GPU-scalar Python branches in LARC with equivalent tensor operations. The exact patch is saved with the run. MPS numerical reproducibility is not the original CUDA environment.

DCNet inference combines the matching logit slices from four rotated inputs for every learned task, concatenates all task outputs and takes argmax. No per-head softmax is added. The previous branch pilot uses different scoring rules, as labelled above.

Persistent memory counts model tensors, cumulative masks, dense update masks and prototype tensors retained by this runner; peak GPU memory and optimizer workspace are separate and not measured. Checkpoints resume at completed task boundaries, preserving random states. Full result means are reported only for completed ten-task seeds.

[Runner](../../bdh_cl/scripts/dcnet_official_baseline.py) · [Run directory](../../results/dcnet_raw_short_20261027) · [Branch pilot](../dcnet_branch_pilot_20261026/REPORT.md)

This replaces the ResNet with the same first-layer dimensions as the branch pilot, not with a pretrained or CNN encoder. DCNet HAT sigmoid gates can activate multiple branches per group, whereas the pilot uses one disjoint branch per group. Its official 128→128→256 projection and learned task heads add parameters. First-layer dimensions are matched; the complete models and active compute are not identical. Raw HAT connection masks use the native min(input_gate, output_gate) rule with group input protection defined by the maximum branch gate.


## Completed-run audit

Three seeds completed. Incremental average **15.47% ± 0.40**; final **8.67% ± 0.37**; correct-task oracle diagnostic **38.48%** (not the inference-ID-free result). All checkpoints passed finite-state, no-CNN, all-100-class scoring, independent rotation-ensemble and inference-invariance checks.

The same first-layer shape and paired initialization remove the CNN-versus-raw-pixel difference. This remains a DCNet architectural ablation: HAT sigmoid gates may activate multiple branches, native projection/learned heads differ, and DCNet uses 5 feature plus 5 classifier epochs with rotation/colour/crop augmentation. The branch pilot used 5 epochs without image augmentation. DCNet sees approximately 60 augmented views per original example across the two phases, versus five presentations in the pilot; the DAC statistics pass is additional. These results cannot establish an equal-compute algorithm ranking or reproduce the published DCNet score.

[Post-run verification](../../results/dcnet_raw_short_20261027/postrun_verification.json) · [Summary](../../results/dcnet_raw_short_20261027/summary.json)
