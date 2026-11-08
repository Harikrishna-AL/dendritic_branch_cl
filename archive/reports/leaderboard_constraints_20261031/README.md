# Missing leaderboard methods tested under our constraints

This study extends the previous comparison. All values below are **new local measurements**, not copied leaderboard scores. MIST is a documented paper reconstruction, not an official reproduction.

| Method | Final accuracy ± seed SD | Retained payload MB | Payload range MB | Additional optimizer MB |
|---|---:|---:|---:|---:|
| PEC (ICLR 2024), size-matched | 91.35 ± 0.19% | 2.116580 | 2.116580–2.116580 | 1.425760 |
| Streaming LDA, PCA50 | 86.06 ± 0.20% | 0.543960 | 0.543960–0.543960 | 0.000000 |
| Streaming QDA, PCA50 | 95.21 ± 0.17% | 0.724032 | 0.724032–0.724032 | 0.000000 |
| RLS / ridge, PCA50 | 83.23 ± 0.15% | 0.348840 | 0.348840–0.348840 | 0.000000 |
| VFDT (River), PCA50 | 33.62 ± 2.98% | 0.911992 | 0.789973–1.035733 | 0.000000 |
| MIST-G reconstruction, PCA50 | 87.06 ± 0.36% | 0.934415 | 0.724408–1.198864 | 0.000000 |
| MIST-K, unverified diagnostic, PCA50 | 18.66 ± 2.06% | 0.934415 | 0.724408–1.198864 | 0.000000 |

Branch + auxiliary reference: **74.76 ± 0.53%, 2.136048 MB**, 529,210 parameters. Its oracle-route diagnostic remains 97.39%, giving a 22.63-point reference gap. The new controls do not erase that within-model gap, but they change comparative claims about exemplar-free superiority. **PEC is the decisive new neural control: it outperforms the branch model with essentially the same total parameter budget and slightly less numerical state.** The earlier selected-view claim that the branch model leads the measured neural no-example-storage methods is superseded by this run.

## Shared constraints

Same cached MNIST, 55,000 training / 5,000 validation indices and official 10,000 test examples as the earlier study; two epochs per digit pair, batch128 for neural updates, seeds101/102/103, each forward and reverse order. All ten classes compete at final test; no true test task IDs, pretrained external data, raw-example replay, or old validation examples used for fitting. Known training boundaries are allowed by the shared protocol. Classical/tree updates consume the same sample sequence one example at a time or through mathematically additive sufficient statistics. This is a two-epoch adaptation, **not strict single-pass Online CIL**.

## Representation and capacity

**PEC:** exact official PecMLP modules via AST import; GELU and LayerNorm, two linear layers, output99. Ten students of hidden width20 and one shared frozen teacher of width396. Students begin with identical weights, as upstream. Total529,145 parameters, including350,955 frozen teacher coefficients and178,190 trainable student coefficients; teacher counted once. The parameter gap from the branch model is only65 coefficients (0.0123%). This matches total stored size, not trainable capacity or FLOPs. Uses standardized784-pixel inputs, independent continuous Adam optimizers, gradient clipping5, no scheduler, and the shared minibatch/epoch schedule. Validation chooses LR .0005/.001/.003/.01. The native paper script uses one epoch, batch1, a linear schedule, width10 students and width5000 teacher; our smaller teacher and different update budget are material adaptations.

**SLDA/SQDA/RLS/VFDT/MIST:** PCA50 fitted once to the first1,000 presented examples of the initial training task only, then frozen. The mean and50×784 projection are retained and charged (0.319872 MB, FP64). This follows the leaderboard source paper’s compressed input protocol, but differs from the standardized-pixel neural backbones. Every transform is fitted without future-task or test data; fitting examples are not retained. These models do not have an equivalent neural parameter count; all their persistent statistics and transform are instead reported explicitly.

## Statistical baselines

SLDA and SQDA use class-wise online counts/means/full centered scatter matrices. Parallel-Welford minibatch merging equals sequential sufficient-statistic accumulation. SLDA pools within-class covariance; SQDA keeps per-class covariance. Both use (1−s)Σ+sI, empirical class priors, Cholesky precision/log-determinant evaluation; validation selects s from1e−4/.001/.01. Stored scatter, means/counts, retained precision and log-determinant caches are all charged. These are covariance-aware classifiers, not the excluded nearest-class-mean controls.

RLS uses exact ridge sufficient statistics ΦᵀΦ and ΦᵀY, with an appended bias feature and ridge applied to all coordinates. A solution can be computed from these statistics at any point; final inference solves once and retains the head. Validation selects ridge .1/1/10. No random expansion or pretrained features beyond the charged PCA transform.

## Trees and reconstruction limits

**VFDT:** unmodified River0.26.1 HoeffdingTreeClassifier, Gaussian splitters, Gini criterion, Naive Bayes leaves, tau.05, no drift resets. Validation candidates (delta,grace)=(.95,200),(1e−7,200),(.95,1000). The library max_size100 MiB is a safety ceiling, not a claimed2 MB memory match. Reported nodes and actual Python-model heap estimates are saved per run. This is a River adaptation, not the MIST authors’ implementation.

**MIST-G/K:** no official public implementation was verified. The local reconstruction follows arXiv2605.11617v2 Eq.3–5 and Algorithms1–3: McDiarmid gap radius sqrt(32ln(2dm/delta)/n), class-conditional KLL medians and CDFs, midpoint threshold candidates, alpha-discounted truncated-Gaussian child moments, and empty child sketches. Uses k64, alpha.6, delta.1, grace200, tie.05. MIST-G uses Gaussian leaves; MIST-K uses sketch-CDF likelihoods. The training tree is shared between the two predictors. KLL library5.2.0 is used directly; its C++ compaction RNG is not explicitly seedable, so the stored checkpoints record each realized state.

Implementation choices where the paper does not fully specify code behavior: test split every200 new leaf arrivals; confidence n includes inherited weighted mass; ties may split when the radius falls below.05 as listed in the paper’s configuration table; absent sketch CDFs fall back to inherited Gaussian estimates for split evaluation; absent MIST-K likelihoods contribute only class priors until observations arrive; variance floor1e−6 and small truncated-tail probability floors are used for stability. MIST-K validation selects (epsilon,beta) from(1,.5),(1e−5,.5),(1e−5,1); this is additional tuning beyond the published defaults. These choices and the two-epoch stream can materially change results. **Do not treat these runs as a verdict on official MIST performance.** In particular, MIST-K has a severe validation-to-final performance change. Its empty-sketch prediction behavior and split sensitivity are not verified against official code. Treat MIST-K as an exploratory implementation diagnostic; its low value must not be used to claim superiority over the published method. Its raw runs and summary are preserved, but it is excluded from headline competitive plots and their presentation CSV.

## Selection, resources and verification

All selections were frozen on seed113 forward validation before final-test evaluation. Selected settings: `{"mist": {"alpha": 0.6, "delta": 0.1, "grace": 200, "k": 64, "tie": 0.05}, "mist_k": {"alpha": 0.6, "beta": 0.5, "delta": 0.1, "epsilon": 1.0, "grace": 200, "k": 64, "tie": 0.05}, "pec": {"lr": 0.01}, "rls": {"ridge": 0.1}, "slda": {"shrinkage": 0.01}, "sqda": {"shrinkage": 0.01}, "vfdt": {"delta": 1e-07, "grace": 200}}`. Accuracy SD is across three seed means after averaging orders, not across six independent seeds. Final models see110,000 labeled sample presentations; PCA has an additional one-time unlabeled fit to the first1,000 current-task examples.

Memory is decimal MB of retained numerical payload: neural weights, sufficient statistics/caches, PCA, tree numerical fields and encoded KLL sketch state. River tree numeric fields are traversed explicitly; its separate Python heap estimate includes object overhead. KLL is charged by serialized payload because native allocation size is not exposed. This metric is not process RSS or peak RAM, and it must not be described as a universal hardware-memory measurement. Optimizer payload is additional. Input caches, activations/gradients, Python/RNG overhead and temporary PCA/linear-algebra workspaces are excluded, consistently with the earlier numeric-state comparison. Tree payload varies with the seed/order; both mean and range are shown. Training timings exclude the PCA fit and inference.

Five focused checks pass: PEC update/inference match actual upstream methods; online moments/QDA match batch Gaussian likelihoods; RLS matches sequential Sherman–Morrison updates; inherited Gaussian moments conserve discounted mass/mean/variance; MIST checkpoints preserve sketch-based predictions. Evaluation-state checks verify that test inference does not update models. KLL’s lazy internal sorting is canonicalized before bytewise state comparison.

## References

- [PEC paper, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/214308a2d5e3f83ef9ad2739e1cbc46d-Abstract-Conference.html); [official code](https://github.com/michalzajac-ml/pec), pinned commit3c15633bb93775b9220ce79080c98f4b1ddc86e1.
- [MIST source paper, v2](https://arxiv.org/html/2605.11617v2).
- [SLDA reference](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf).

## Artifacts

- [Expanded no-example-storage comparison](01_expanded_replay_free_comparison.png)
- [All presented methods and payload](02_all_methods_accuracy_and_payload.png)
- [Expanded accuracy–memory plot](03_expanded_accuracy_memory.png)
- [All presented values CSV](all_presented_methods.csv); editable SVG accompanies every PNG.
- Raw runs, validation trials, selections, source, provenance and checkpoints are in `../../results/leaderboard_missing_20261031/`.
