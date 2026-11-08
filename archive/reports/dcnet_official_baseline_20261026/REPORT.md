# Official-module DCNet baseline versus the branch pilot

Status: stopped at the user’s request on 27 October 2026. No complete baseline score. Superseded by the raw-pixel comparison.

Official source: https://github.com/Tianqi-Wang1/DCNet at commit c3061a0492e3798d5957caa4f30fae501fd4f219.

DCNet budget: 5 representation epochs + 5 OOD-classifier epochs per task, batch 16, device mps. This is not the published 700+100-epoch recipe. Training uses the official ResNet, losses, augmentation, HAT and OOD classifier functions. DAC begins at epoch 3; warmup is shortened to one epoch.

Both experiments use the same 45,000 fit images, 10,000 official test images, ten equal class blocks and class permutation. The 5,000 validation images are unused. Task boundaries are known during training; no true task ID is supplied at inference and no old observations are rehearsed. Backbones, batch sizes, augmentation and compute differ, so this is a same-data comparison, not a resource-matched experiment.

| Method | Incremental average (%) | Final accuracy (%) | Persistent tensor state |
|---|---:|---:|---:|
| Branch pilot, head probabilities | 20.39 | 12.64 ± 0.27 | 15.81 MB |
| Branch pilot, cosine | 21.59 | 13.91 ± 0.20 | 15.81 MB |
| DCNet | Pending | Pending | Pending |

Progress: {"seed": 701, "task": 1, "phase": "representation", "epoch": 2, "seconds": 314.1227596660028}.

## Stage progress (not final benchmark scores)

| Seed | Completed tasks | Latest seen-class accuracy (%) |
|---|---:|---:|

## Verification and portability

Two-task smoke training passed both official phases and HAT updates; independent official cil evaluation matched the input-only prediction wrapper. Inference state remained unchanged and model values were finite. The optimizer synchronization patch was checked against original LARC for multiple steps, clipping on/off, zero norms, weight decay and momentum.

Patches select MPS instead of CUDA, remap CUDA-saved tensor prototypes onto the selected device, use FP32 crop parameters, omit unused faiss/diffdist imports, and replace GPU-scalar Python branches in LARC with equivalent tensor operations. The exact patch is saved with the run. MPS numerical reproducibility is not the original CUDA environment.

DCNet inference combines the matching logit slices from four rotated inputs for every learned task, concatenates all task outputs and takes argmax. No per-head softmax is added. The previous branch pilot uses different scoring rules, as labelled above.

Persistent memory counts model tensors, cumulative masks, dense update masks and prototype tensors retained by this runner; peak GPU memory and optimizer workspace are separate and not measured. Checkpoints resume at completed task boundaries, preserving random states. Full result means are reported only for completed ten-task seeds.

[Runner](../../bdh_cl/scripts/dcnet_official_baseline.py) · [Run directory](../../results/dcnet_official_short_20261026) · [Branch pilot](../dcnet_branch_pilot_20261026/REPORT.md)
