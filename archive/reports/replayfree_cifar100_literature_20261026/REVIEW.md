# Split CIFAR-100: replay-free class-incremental literature comparison

Search date: 26 October 2026. This is a broad, source-linked literature screen, not an exhaustive systematic review or a reproduced leaderboard. It includes foundational baselines, competitive recent methods, and explicit exclusions. Numerical results are reported by papers; none were reproduced for this review. Missing memory values are left missing rather than invented.

## Eligibility and interpretation

The included setting requires prediction over all observed classes without a supplied task ID at inference. No previous-task images, individual embeddings, generated images, or sampled/translated old-class features may be rehearsed. Accumulated sufficient statistics, parameter importance, model snapshots, and class means used directly for classification are allowed and counted as memory. Current-task augmentation and multiple current-task epochs are allowed. These are distinct from strict single-pass learning.

Known task boundaries during training are allowed in the main tables, but must be reported. This is weaker than the project's additional ambition of receiving no training boundary signals. Multi-head models qualify if they autonomously combine/select heads; an oracle selecting a head does not qualify.

“Paper-supported” means the accessible method and evaluation description support eligibility. It does not mean the released implementation has been audited or that experimental results have been independently verified. Unresolved methods are outside the eligible tables.

Define final average accuracy as A_final = mean_i R[T,i], where R[T,i] evaluates task i after all tasks have been learned, with all seen classes eligible. Define average incremental accuracy as A_inc = mean_t A_t, where A_t evaluates all classes observed through stage t. Papers sometimes call either quantity “average accuracy”; they must not be interchanged. For unequal tasks, authors' sample-weighted and task-weighted definitions also need checking.

Separate these protocols:

- **Equal-10:** 10 tasks of 10 classes, trained from scratch without external pretraining.
- **Equal-20:** 20 tasks of 5 classes, trained from scratch.
- **Warm-50/K:** a 50-class base task followed by K increments sharing the remaining 50 classes. K=10 means 11 total stages, not Equal-10.
- **Pretrained Equal-10:** 10 equal tasks using an externally pretrained representation; backbone/checkpoint remains a material experimental variable.
- **Online/blurry:** single-pass or overlapping-class streams; keep separate from offline block training.

## 1. Learning from scratch: equal splits

All figures are percentages. NR means not extracted/reported in the cited evidence, not zero. Rows from different papers have different augmentations, training budgets, and sometimes architectures; this is a literature comparison, not a matched experiment.

| Method / paper | Protocol / numerical source | A_inc | A_final | Retained state and memory evidence | Eligibility |
|---|---|---:|---:|---|---|
| DCNet, IJCAI 2025 | Equal-10, original Table 1 | 75.84 ± 0.52 | 65.40 ± 0.26 | ResNet-18, task masks, projection/classifier components; complete bytes NR | Paper-supported; inferred task routing |
| DCNet | Equal-20, original Table 1 | 71.52 ± 0.42 | 58.43 ± 0.36 | As above, more task-specific state | Paper-supported |
| SEED, 2024 preprint | Equal-10, original Table 1 | 61.7 ± 0.4 | NR | Five experts, shared stem, per-expert Gaussian class statistics; explicit storage formula | Paper-supported; likelihood ensemble over classes |
| SEED | Equal-20, original Table 1 | 56.2 ± 0.3 | NR | As above | Paper-supported |
| LwF + Learnable Drift Compensation (LDC), ECCV 2024 | Equal-10, original Table 1 | 59.5 ± 3.9 | 45.4 ± 2.8 | ResNet-18 + class means; teacher and learned drift map during updates | Paper-supported; NCM inference |
| LwF + Semantic Drift Compensation (SDC) | Equal-10, LDC Table 1 | 56.2 ± 3.0 | 40.6 ± 1.8 | Backbone, class means, previous model for drift estimation | Paper-supported baseline variant |
| LwF + NCM | Equal-10, LDC Table 1 | 56.2 ± 3.6 | 40.5 ± 2.7 | Backbone + means, training teacher | Paper-supported baseline variant |
| FeCAM, NeurIPS 2023 | Equal-10, reevaluated in LDC Table 1 | 48.1 ± 1.3 | 33.1 ± 0.9 | Frozen first-task ResNet-18, means/covariances | Paper-supported; cold-start weakness |
| EDD, AAAI 2026 | Equal-10 / Equal-20, Table 1 | NR | 37.24 ± 0.29 / 21.68 ± 0.10 | ResNet-18, expanding learned key/value slots; training teacher; complete bytes NR | Paper-supported; common class classifier |
| Co2In, Oct. 2026 preprint | Equal-10, narrower ResNet-18, Table V | NR | 37.54 ± 0.90 | Table XI reports 25.53 MB model/long-term memory; not peak training RAM | Paper-supported; task-agnostic inference |
| SPARC, 2025 preprint | **5 × 20 classes**, original Table 1 | NR | 49.03 ± 0.05 | 1.04M reported model parameters ≈ 4.16 MB FP32 weights; total training state NR | Paper-supported Class-IL row only |
| EWC, 2017 | Equal-10, baseline in DCNet Table 1 | 49.14 | 31.17 | Weight reference/Fisher information; implementation-dependent totals | Classical reference, not current SOTA |
| LwF, 2016/2018 | Equal-10, baseline in DCNet Table 1 | 53.91 | 32.80 | Current model and training teacher | Classical reference, not current SOTA |

DCNet uses gradient learning and HAT-style masks, not an analytic classifier update. Its CIFAR experiments use 700 training epochs, LARS, and task-organized training. Its Table 1 mixes reproduced and cited baselines; do not assume equal compute. [DCNet, Sections 4–5 and Table 1](https://ijcai.org/proceedings/2025/0715.pdf).

SEED's original result differs from the numbers quoted in DCNet (62.04/51.42 for average/final Equal-10). Preserve the provenance rather than merging these as one experiment. SEED stores Gaussian statistics for classification, not sampled old-class features for training. [SEED, Sections 3–4 and Appendix A](https://arxiv.org/html/2401.10191).

LDC fits a mapping using current-task data and applies it to old class means; nearest-class-mean inference considers all seen labels. [LDC, Sections 3–4 and Table 1](https://arxiv.org/html/2407.08536).

SPARC is relevant to compact architectural mechanisms, but its five-task result cannot rank against Equal-10. Its partitioning assumes the task count in advance. The parameter conversion above is a weight-storage estimate, excluding optimizer state, masks, temporary workspace, and framework overhead. [SPARC, Table 1 and Appendix E.3](https://arxiv.org/html/2505.21942).

EDD retains learned slots and distils the previous model on current inputs, without stored observations. [EDD, method and Table 1](https://arxiv.org/html/2511.09871).

Co2In consolidates parameters into long-term patterns. Its standard-width ResNet-18 variant reports 39.25% final accuracy and 309.32 MB; its narrower version reports 25.53 MB. These are author-defined model/long-term memory figures, not training peaks. Its task-incremental memory-efficiency plot is outside this comparison. [Co2In, Tables V and XI](https://arxiv.org/html/2610.21804).

## 2. Base-task-trained analytic/statistical methods: warm start

These methods begin without an external pretrained backbone, but train a large 50-class base task before incremental learning. They do not demonstrate the same representation-learning challenge as beginning with only 5 or 10 classes.

| Method / paper | Protocol and backbone | A_inc | A_final | Memory retained for later learning |
|---|---|---:|---:|---|
| ACIL, NeurIPS 2022 | Warm-50/10, ResNet-32, original Table 1 | 66.07 | NR | Frozen backbone, expansion weights, classifier, inverse autocorrelation/statistics matrix |
| DS-AL, AAAI 2024 | Warm-50/25, ResNet-32, original Table 1 | 66.20 ± 0.05 | 58.32 ± 0.12 | Main and compensation streams with recursive matrices |
| DS-AL | Warm-50/25, ResNet-18, original Table 1 | 68.40 ± 0.20 | 61.41 ± 0.26 | As above; different backbone |
| REAL, KBS 2026 / Dec. 2025 manuscript | Warm-50/10, ResNet-18, Table 1 | 72.41 | 65.86 | Frozen enhanced backbone, feature-fusion projections, analytic statistics |
| REAL-DS, same paper | Warm-50/10, ResNet-18, Table 1 | 72.35 | 66.74 | REAL plus compensation stream |
| FeCAM, NeurIPS 2023 | Large base, T=10 in original Table 1, ResNet-18 | 70.8 | NR | Per-class means and covariance matrices |
| FeCAM pooled covariance | Same original setting | 68.6 | NR | Means plus one shared covariance matrix |
| Euclidean NCM | Same FeCAM comparison | 64.6 | NR | Means only, beyond the frozen backbone |

ACIL uses an 8,000-dimensional expansion on CIFAR-100 in its paper. One dense FP32 8,000² matrix alone is 256 MB. [ACIL, Table 1 and method](https://papers.nips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf).

DS-AL keeps recursive state for two streams; a single-stream memory estimate cannot be presented as its complete footprint. [DS-AL, Section 3 and Table 1](https://arxiv.org/html/2403.17503).

REAL uses expansion width 15,000. One dense FP32 square matrix is 900 MB; a two-stream configuration may require multiple such matrices, plus projections and classifier state. “Feature fusion buffer” names a projection module, not an exemplar replay buffer. These are structural estimates, not measured allocator memory. [REAL, Sections 3–4 and Table 1](https://arxiv.org/html/2403.13522).

FeCAM's direct Mahalanobis classifier does not synthesize training features. Its 512-dimensional ResNet features make covariance storage significant; its pooled variant offers an explicit accuracy/storage tradeoff. [FeCAM, Table 1 and Section 4.1](https://arxiv.org/html/2309.14062).

The official analytic-learning repository additionally reports ACIL/G-ACIL 71.047/63.384 and DS-AL 71.277/64.043 average/final accuracy with width 8192. **Do not substitute those figures into the paper rows:** their exact run protocol/checkpoint must be resolved first. [Official repository](https://github.com/ZHUANGHP/Analytic-continual-learning).

## 3. Externally pretrained models: equal splits

All rows below support no oracle task ID at inference and no old-example/synthetic-feature rehearsal in the selected variant. Training boundaries generally remain known. Different checkpoints and implementations prevent a universal numerical ranking.

| Method / paper | Protocol / result provenance | A_inc | A_final | State beyond backbone |
|---|---|---:|---:|---|
| RanPAC, NeurIPS 2023 | Equal-10, original Table 1, pretrained ViT-B/16 | NR | 92.2 | Random projection, class cross-statistics, 10,000² Gram matrix, fitted classifier, optional first-session adapter |
| CoSO, NeurIPS 2025 | Equal-10, original Table 2 | 92.99 ± 0.23 | 88.77 ± 0.16 | Historical gradient subspace, current sketch/projection, compressed Adam state |
| SD-LoRA, ICLR 2025 | Equal-10, evaluated in CoSO Table 2 | 91.81 ± 0.27 | 87.30 ± 0.45 | Task LoRA directions and learned magnitudes; grows during learning |
| InfLoRA, CVPR 2024 | Equal-10, original Table 2 | 91.70 ± 0.32 | 86.51 ± 0.73 | Merged adaptation weights and retained subspace bases |
| InfLoRA-b5 | Same paper, bottom-five-block variant | 91.59 ± 0.13 | 87.06 ± 0.25 | Fewer adapted blocks plus subspace state |
| CODA-Prompt, CVPR 2023 | Equal-10, evaluated in InfLoRA Table 2 | 91.08 ± 0.28 | 86.62 ± 0.11 | Prompts, keys, attention parameters |
| DualPrompt, ECCV 2022 | Equal-10, same evaluation | 90.06 ± 0.07 | 84.42 ± 0.30 | General/expert prompts and retrieval keys |
| LAE, CVPR 2023 | Equal-10, same evaluation | 89.84 ± 0.03 | 84.15 ± 0.10 | Adapters and ensemble/EMA adapter copy |
| L2P, CVPR 2022 | Equal-10, same evaluation | 87.64 ± 0.25 | 82.48 ± 0.20 | Prompt pool and keys |
| APT, ICCV 2025 | Equal-10, original Table 1 | NR | 88.88 ± 0.65 | Shared additive prompts, stored prompt snapshot/fusion, classifier |
| APT | Equal-20, original Table 3 | NR | 86.37 ± 0.46 | Same mechanism |
| FeCAM with pretrained ViT | Equal-10, original Table 2 | NR | 85.7 | Class means and per-class covariance matrices |
| SECA, AAAI 2026 | Equal-10, CLIP ViT-B/16, Table 2 | 86.70 | 79.79 | Adapter pool, visual/text prototypes and classifiers; include text encoder/storage |
| SECA | Equal-20, same paper | 85.35 | 77.73 | Same; **SECA++ is excluded** |
| CCA-CL, CVPR 2026 | Equal-10, CLIP ViT-B, original Table 1 | 87.2 | 83.0 | Visual/text covariance and cross-covariance matrices, random Fourier projection, both encoders |
| DGASA, PLOS One 2026 | **Equal-20**, ImageNet-21K ViT-B, Table 1 | 92.11 | 86.25 | Growing adapters and class prototypes; memory arithmetic discrepancy noted below |
| sLoTh, Aug. 2026 preprint | Equal-10 CIL, pretrained QKFormer, Table 1 | NR | 81.23 ± 1.74 | 64.32M reported total parameters; prototypes, anchors/teacher also require accounting |

RanPAC's 92.2 is final accuracy, not accuracy averaged over stages. Its sufficient-statistic updates use current inputs and do not generate old inputs. [RanPAC, Algorithm 1, Section 4.5 and Table 1](https://arxiv.org/html/2307.02251).

CoSO explicitly forbids historical data and supplied inference identities. Its CIFAR backbone is ImageNet-21K pretrained and ImageNet-1K fine-tuned. Its reported GPU memory comparison is on **ImageNet-R**, not CIFAR-100; it must not be pasted into a CIFAR memory column. [CoSO, Section 3, Table 2 and Appendix C–D](https://arxiv.org/html/2505.11816).

InfLoRA's method merges past branches into weights; the historical protection uses subspace bases. The baseline numbers above are the selected implementations in that paper, not universal values. [InfLoRA, Section 3, Table 2 and Appendix B.2](https://arxiv.org/html/2404.00228). SD-LoRA optimizes current-task loss over accumulated parameter directions. [SD-LoRA, Algorithm 1](https://arxiv.org/html/2501.13198).

APT explicitly defines “Avg Acc” as accuracy averaged over tasks **after the final stage**. It reports 0.02M prompt parameters in an ImageNet-R efficiency comparison: ≈0.08 MB FP32 prompt weights, not total memory. It additionally stores old prompts for fusion. [APT, Sections 4.3–5.1 and Tables 1–3](https://arxiv.org/html/2503.07979).

SECA's paper distinguishes no-replay SECA from Gaussian-feature-replay SECA++. Text embeddings are external semantic information, not a task oracle. [SECA, method and Tables 2–3](https://arxiv.org/html/2508.01579). Conference metadata/code: [official AAAI 2026 repository](https://github.com/HHHLF/SECA_AAAI2026).

sLoTh uses all-seen-class nearest-prototype inference and current-input distillation. Its total-parameter conversion is ≈257.28 MB FP32 weights; this excludes uncounted auxiliary copies/state, not a measured total. Treat the preprint as an unreplicated recent result; its pretraining-data provenance, finer-granularity improvements and prototype drift merit verification. [sLoTh, Sections 3.3–3.4 and Table 1](https://arxiv.org/html/2608.26720).

CCA-CL accumulates statistics and computes a visual/text subspace analytically. Without first-task tuning it reports 85.4/81.0 average/final; complete persistent bytes are unresolved. [CCA-CL, Table 1 and method](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Subspace_Alignment_for_CLIP-based_Continual_Learning_via_Canonical_Correlation_Analysis_CVPR_2026_paper.pdf).

DGASA compensates prototypes for direct classification without sampling rehearsal features. Its memory formula at 20 tasks, 12 blocks, width 768, rank 16 yields 6,082,560 adapter parameters: 24.330 MB FP32 plus 6.144 MB prototypes, approximately 30.474 MB beyond the backbone. The paper instead reports 11.94 MB, apparently mixing parameter counts and bytes. This calculated discrepancy requires clarification before memory ranking. [DGASA, Table 1 and memory analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC13152188/).

## 4. Recent online / generalized settings: separate comparison

| Method | Setting / source | Metric and result | Memory interpretation |
|---|---|---|---|
| GACL, evolved from G-ACIL (2024 manuscript, later revision) | Si-Blurry CIFAR-100; disjoint ratio 50%, blurry ratio 10%; Table 2 | A_inc 56.24 ± 3.12; final 70.31 ± 0.06 | Frozen backbone, projection, recursive statistics and classifier; dense width-squared state; no inference oracle |
| MePo + DualPrompt, Feb. 2026 preprint | Online Split CIFAR-100, Sup-21K, Table 7 | A_inc 85.28; final 80.11 | Base prompt method plus refinement machinery; complete bytes NR |
| MePo + DualPrompt | Offline Split CIFAR-100, same table | A_inc 89.36; final 84.50 | Not directly comparable to online row |
| sLoTh (full) | Single-pass, boundary-free CIFAR-100, Table 3 | Reported Avg Acc 55.12 ± 1.18 | Prototypes and parameter/teacher state; complete bytes NR |
| SPIN-SGD, 20 Oct. 2026 preprint | From-scratch rehearsal-free Split CIFAR-100, Table 8 | Anytime accuracy 14.63; final 5.91 | Multiple optimizer parameter sequences; O(P) extra vectors; measured bytes NR |
| SPIN-AdamW | Same | Anytime accuracy 15.60; final 5.58 | As above, plus Adam moments |

GACL explicitly excludes historical samples and test-time task IDs, but its main Si-Blurry protocol has recurring/overlapping classes. It cannot be ranked as equal disjoint class blocks. [GACL, Sections 3–4](https://arxiv.org/html/2403.15706).

MePo refines a pretrained model through joint pseudo-task meta-training on pretraining data; its downstream rehearsal-free status does not make its upstream preparation comparable to learning from scratch. [MePo, Sections 3–4 and Table 7](https://arxiv.org/html/2602.07940).

SPIN is an optimizer usable with replay algorithms; only its explicit **rehearsal-free Table 8** variants qualify here. Its anytime metric is not A_inc measured only at block ends. [SPIN, Section 4 and Table 8](https://arxiv.org/html/2610.27835).

PROL (ICCV 2025) is another important online no-rehearsal candidate. Its accessible paper reports a ViT-Tiny result of 64.59% final accuracy with 5.6M backbone parameters, while its main method adds 0.213M parameters in the complexity comparison. These refer to different configurations: do not add them together or rank without matching the configuration. Full corresponding CIFAR result/total-memory accounting remains unresolved in this review. [PROL official paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Masum_PROL__Rehearsal_Free_Continual_Learning_in_Streaming_Data_via_ICCV_2025_paper.pdf).

## 5. Comparable memory accounting

Units: MB = 10^6 bytes. Estimates below assume dense FP32 tensors (4 bytes/scalar). Actual implementations can use FP64, triangular storage, duplicated inverses, cached features or additional copies; those must be measured. These estimates are **components**, not complete measured footprints or accuracy-matched budgets.

| Method/state | Structural storage | Concrete estimate | What is excluded |
|---|---|---:|---|
| NCM means | 4 C d | C=100,d=512: 0.205 MB; d=768: 0.307 MB | Backbone, counts/normalization, any training teacher |
| FeCAM pooled means + covariance | 4(Cd+d²) | d=512: 1.253 MB | Backbone, inverse/cached transform, workspace |
| FeCAM per-class means + full covariance | 4C(d+d²) | d=512: 105.062 MB; d=768: 236.237 MB | Backbone, inverse/cached transform, workspace |
| SEED Gaussian statistics | 4 sum_i sum_{j>=i} C_j[S+S(S+1)/2] | Five experts introduced over first five tasks of Equal-10, S=64: 3.430 MB with packed symmetric covariances | Expert networks/shared stem; dense matrices cost more |
| ACIL correlation state | 4 D² for one matrix | D=8000: 256 MB; D=8192: 268.435 MB | Projection, classifier, backbone, numerical workspaces |
| DS-AL recursive correlation state | Approximately 8 D² for two same-width matrices | D=8192 illustration: 536.871 MB | All other state; illustrative width is not an asserted paper-wide configuration |
| REAL recursive state | 4 D² per retained square matrix | D=15000: 900 MB per matrix | Fusion projections, classifier, backbone, extra stream, workspace |
| RanPAC Gram matrix | 4 M² | M=10000: 400 MB | All other state |
| RanPAC random projection | 4 L M | L=768,M=10000: 30.720 MB | Backbone and optional adapters |
| RanPAC each class cross-statistics/head matrix | 4 M C | C=100,M=10000: 4 MB per matrix | Whether cross-statistics and fitted head coexist must be measured |
| APT prompt parameters | 4 × 0.02M, paper-rounded count | ≈0.080 MB | Backbone, classifier, prompt snapshot, optimizer |
| InfLoRA additional branch parameters | 4 × (4 l d r) bytes | Depends on blocks l and rank r | Historical subspace bases, backbone, classifier, optimizer |
| SPARC reported weights | 4 × 1.04M | ≈4.16 MB | Nonparameter state, optimizer, workspace |
| sLoTh reported total parameters | 4 × 64.32M | ≈257.28 MB | Auxiliary snapshots/anchors/prototypes not included in reported parameter count |

The SEED figure follows its published packed-symmetric formula and default S=64/K=5; it is not an implementation RAM measurement. FeCAM estimates follow its stated feature dimensions. Analytic-method matrices retained only for continued learning can be discarded for deployment, so inference-only memory is lower. A fitted head and backbone still remain.

Report each experiment's (1) persistent continuation state, (2) inference-only state, and (3) peak training memory. Include all frozen weights, masks, task heads, class statistics, model snapshots, optimizer tensors, retained data/features, and actual precision. Exclude current-task dataset storage only under a shared rule across methods, and disclose caches. Peak memory additionally includes gradients, activations and matrix-solve workspace.

A prompt-only count and a Gram-matrix-only count cannot be compared as total memory. A useful experiment is accuracy versus measured continuation-state bytes at several budgets, within one protocol and representation-access setting. Include training/inference time because a 700-epoch method and a single-pass method are solving different resource-constrained tasks.

## 6. Excluded or withheld despite attractive labels

| Paper/method | Decision for the strict comparison | Evidence |
|---|---|---|
| PASS, IL2A, SSRE, PRAKA | Exclude: prototype augmentation/rehearsal or pseudo-feature training | REAL related-work discussion and original-method references |
| FeTrIL / FeTrIL++ | Exclude: translated old-class pseudo-features used to train classifier | FeTrIL official implementation |
| EFC / EFC++ | Exclude: Gaussian prototype rehearsal | EFC++ Section 4.1 explicitly samples synthetic old-class features |
| SLCA | Exclude: Gaussian-sampled old-class features for classifier alignment | SLCA Section 3.3 |
| SECA++ | Exclude; ordinary SECA qualifies | SECA explicitly labels feature-replay variant |
| RAPF, CLAP/CLAP4CLIP, PROOF variants in SECA comparison | Exclude the ER/FR-marked configurations; do not generalize to every variant | SECA Tables 1–2 |
| LoRA-minus DRS, CVPR 2025 | Withhold from strict table: old class means enter augmented triplet loss as negatives; needs an explicit decision whether training-time prototype anchors count as replay | Section 3.4; 0 KB statistics claim is not total memory |
| VPT-NSP / VPT-NSP² variants | Withhold until exact variant is checked: SECA marks a standard CLIP variant as feature replay and separately evaluates a no-replay dagger variant | Do not transfer replay status across variants/backbones |
| ORACIL, Electronics July 2026 | Withhold: “no raw-image replay” does not establish no feature replay; router uses feature-derived distance representations | Full router historical-state/update audit needed; do not admit 95.77% on title alone |
| LACL, Neural Networks August 2026 | Recent analytic candidate; complete numerical/memory and cache audit unresolved | Accessible abstract plus official code provide insufficient verified comparison data |
| D-VLAM, ECCV 2026 | Recent candidate; no accuracy admitted here because primary full paper could not be retrieved | Conference listing confirms title; secondary summary is insufficient for protocol certification |
| UMIM | Withhold under strongest “no reuse of buffered observations” interpretation | New-expert sample buffer; distinguish this from conventional old-task rehearsal |
| Task-IL PackNet/PNN/NISPA/HAT results | Exclude task-oracle evaluations | Can qualify only with separately evaluated autonomous inference |

Exclusion sources: [FeTrIL code/method](https://github.com/GregoirePetit/FeTrIL), [EFC++](https://link.springer.com/article/10.1007/s11263-026-03018-4), [SLCA](https://arxiv.org/html/2303.05118), [LoRA-minus DRS](https://arxiv.org/html/2503.18985), [SECA](https://arxiv.org/html/2508.01579), [ORACIL](https://www.mdpi.com/2079-9292/15/13/2941), [LACL publisher](https://www.sciencedirect.com/science/article/pii/S0893608026002236), [LACL code](https://github.com/origi6615/lacl), [ECCV 2026 listing](https://eccv.ecva.net/Conferences/2026/AcceptedPapers), [UMIM](https://d-nb.info/1350286508/34).

## 7. What this establishes for the project

Replay-free class-incremental learning is already an established feasible setting. DCNet is an important from-scratch equal-split accuracy reference; LDC, SEED and SPARC provide different statistical/architectural baselines. ACIL/DS-AL/REAL test whether sufficient-statistic classifiers already solve the proposed output-retention problem under base-task-trained features. RanPAC/FeCAM and parameter-efficient methods are separate pretrained references.

The literature does not supply one reliable universal accuracy-versus-total-memory ranking. Important missing measurements are complete DCNet state, exact pretrained backbones/checkpoints, analytic precision/workspace, and teacher/optimizer copies. Reproduction with instrumentation is needed for that ranking; paper tables alone cannot supply it.

A defensible experiment would fix Equal-10 and Equal-20, train from scratch, compare no-replay methods with all-class inference, and separately test boundary-free operation. Report A_final and A_inc, acquisition, class/order seeds, persistent continuation bytes, inference bytes, training peak bytes, and compute. Add an analytic/NCM classifier on the same features as a diagnostic. The project's contribution must be demonstrated relative to this evidence, not inferred from its absence of replay.

## Search coverage and limits

Searches combined Split CIFAR-100, class-incremental, rehearsal/replay-free, exemplar-free, analytic, memory, and 2025/2026 terms. Accessible original papers, official proceedings, and author code were used for results. Reference chains were followed through DCNet, FeCAM, LDC, REAL, InfLoRA, APT, CoSO and SECA. Preprints through 20 October 2026 were screened. Discovery also surfaced inaccessible or unresolved recent papers; they are explicitly withheld above. No claim of covering every paper or establishing an absolute latest SOTA is made. No source's “exemplar-free” label alone was sufficient for admission.
