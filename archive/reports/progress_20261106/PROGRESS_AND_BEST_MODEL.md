# Research progress and best current version — 6 November 2026

This document records the evidence **before** the reconstruction-ranking / training-through-refinement experiment. Later results must not silently replace this baseline. These are exploratory experiments on a repeatedly inspected benchmark, not claims of novelty, SOTA, or a solved continual-learning problem.

## Recommended baseline to preserve

**Sparse four-branch MLP, supervised error-trace selection, hybrid5 auxiliary training, stateless soft-magnitude inference.**

- MNIST784 standardized inputs;168 neurons ×4 tanh branches; fixed random196-pixel subsets per branch; no per-image LayerNorm. Shared linear168→10 classifier.
-134,074 parameters, about1.593 MB persistent tensor state, including int64 sparse connection indices. No replay, pretraining, inference task ID, prototype, or archived-route lookup in its image-only mode.
- Training: five sequential digit pairs, forward/reverse order;55,000 training images and5,000 disjoint validation images. Two epochs/task, batch128, Adam .0005 continuous across tasks, clip5. Known training boundaries; no claim of boundary-free learning.
- Supervised branch-error EMA(.95), select minimum error per neuron. Auxiliary loss remains: selected CE + .5 mean single-neuron branch-substitution CE.
- Hybrid5 evaluates exact auxiliary loss and feedback every fifth global batch; other batches use first-order Taylor branch scores and a differentiable auxiliary surrogate. Residual probabilities are detached, while readout weights and branch responses remain differentiable. No second backward pass in this version.
- Inference: gate softmax(abs(raw branch responses)); each neuron outputs weighted signed tanh responses. All ten output classes compete.

### Evidence and why this is the baseline

Three seeds101–103 ×both task orders. Hybrid5 was selected on validation among gradient approximations. Image-only test **67.91% ±1.79%**, oracle **97.98% ±.77%**. Exact baseline66.70% ±1.01%, oracle98.33%. Hybrid5 minus exact paired95% t interval[-2.51,+4.95]pp: no demonstrated accuracy superiority or formal noninferiority. Its practical advantage is the local clean CPU step benchmark:1.983ms vs4.201ms, about2.12× faster. This is not a larger-network or end-to-end application benchmark.

Full evidence: ../../results/mnist_gradient_branch_scoring_20261106/REPORT.md
Checkpoints: ../../results/mnist_gradient_branch_scoring_20261106/hybrid5_{101,102,103}_{0,4}.pt
Code: ../../bdh_cl/scripts/mnist_gradient_branch_scoring.py

It retains the strongest simple architecture, keeps the auxiliary benefit, halves local step cost, and avoids the failed extra classifier/router/decoder mechanisms. It is a **best current candidate**, not a universally best method.

## Results accumulated so far

| Question | Finding | Evidence directory under results/ |
|---|---|---|
| Do sparse inputs help the MLP? | Dense raw inference56.56%;25%-input sparse63.47%. With tanh inference dense53.75%→sparse64.52%; no input LayerNorm. Sparse134k vs dense529k parameters. | mnist_sparse_reproducibility_20261105 |
| Does soft image inference help? | Hard64.52%→soft magnitude66.70%, same parameters. Learned selectors and extra image-CE training collapsed near20%. | mnist_combined_selector_20261105 |
| Can soft error-trace training replace hard selection? | No tested soft setting beat hard. Soft/uniform training often lost oracle retention. | mnist_soft_error_gates_20261105 |
| Does auxiliary loss matter? | With aux66.70%, without52.99%; oracle98.33% vs98.27%. Faster without aux, but poorer image-only access. | mnist_sparse_aux_ablation_20261106 |
| Are few-shot configurations recoverable? | Fresh search without archived routes97.60% with1 label/class, **only when query grouping supplies the same task context**. Mixed queries with tested router65.45% at10/class. Not novel-class learning. | mnist_fewshot_recovery_20261105 |
| Does label-free temporary state help? | On frozen exact-trained models, pseudo-error state78.54% vs66.70% on200-image contexts; random mixing49.46%. Resetting state removes gain; shuffling branch associations57.51%. State2688 bytes. Hybrid5 variant79.07% persistent,50.60% mixed. | mnist_short_term_context_20261105 and mnist_gradient_branch_scoring_20261106 |
| Can that state replace supervised training traces? | Tested no-state training20.20%, label-free-state training19.08%, supervised trace66.70% using stateless soft inference. | mnist_short_term_context_20261105/TRAINING_ABLATION.md |
| Where is information lost? | Validation current66.96%; true-pair-masked97.89% (task-ID diagnostic). Joint ridge on frozen mixed168 features87.80%; all672 branch features93.50%. These probes use all historical TRAIN labels jointly, **not replay-free results**. | mnist_readout_diagnosis_20261106 |
| Do stored class means fix the classifier? | Continual centroids best43.08%; recomputed joint-data means about80%. Feature drift and class geometry remain issues. | mnist_centroid_ablation_20261106 |
| Does branching the readout help alone? | Matched shared67.87%; learned branched59.69%; frozen replicated head66.97%. Oracle remains about98%. Fixed control is replicated random initialization, not an ETF head. | mnist_branched_readout_20261106 |
| Does masked reconstruction refine useful gates? |18 runs. Validation selected0 refinement steps. Decoder-only test67.91%→35.28% at10 steps although masked error.1520→.1037. Strong joint reconstruction training best validation63.82%, below66.96% baseline. | mnist_masked_gate_refinement_20261106 |

CIFAR extension: from-scratch small CNN with sparse4 branches had23.36% image-only test,80.53% oracle in a one-seed/two-epoch pilot. Larger branch counts did not consistently help, and all tested CIFAR100 branch configurations trailed the plain CNN. Do not present the MLP findings as replicated at scale. See ../../results/dendritic_architecture_screen_20261105/REPORT.md.

## What is supported, and what is not

Supported: supervised error-guided sparse branch training retains useful task configurations; auxiliary training improves access without task context; gradient approximations reduce training cost locally; temporary label-free state can exploit persistent temporal context.

Not established: autonomous context recognition on arbitrary mixed streams; reduced forgetting from label-free short-term-state training; a validated reconstruction-based verification mechanism; scalability to large architectures; novelty or superiority to published methods under native settings.

High oracle accuracy requires saved configurations and true task context. It is not the image-only score. Few-shot same-task episodes also supply context through grouping. Frozen-stream memory tests measure inference adaptation, not training-time weight forgetting.

## Next authorized experiment, specified before results

Test (a) masked reconstruction ranking that prefers lower-classification-loss configurations over genuinely harmful alternatives, and (b) one differentiable reconstruction-driven gate update followed by labeled classification loss during training. Keep the successful hybrid5 classification objective as an anchor. Include detached-decoder and ordinary joint reconstruction controls. Use small sampled configuration sets, not all joint combinations. Choose settings only on original validation; report selected-setting test plus matched baseline. Freeze weights at inference and adapt per-image gates only. Keep all negative results. The decoder itself can forget; ranking does not guarantee gradient alignment, and differentiating through an update costs extra training compute.
