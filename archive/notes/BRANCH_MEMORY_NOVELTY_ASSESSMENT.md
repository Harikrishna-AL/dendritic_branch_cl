# Preliminary novelty assessment — 11 October 2026

A positive electricity forecasting result would be useful evidence of transfer, not proof of novelty or of reduced catastrophic forgetting. This is a targeted literature check, not an exhaustive novelty search. Conference acceptance cannot be predicted from the idea alone.

## Existing ingredients

- **Active Dendrites (Iyer et al., 2022)** already studies dendritic modulation and sparse context-dependent routing to reduce interference: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846219/full . The broad claim that dendrites help continual learning is established territory.
- **DendSN/DBG (Huang et al., 2024)** explicitly uses an additional task context signal for dendritic branch gating: https://arxiv.org/html/2412.06355v1 . The current candidate differs in computing its cue from observed branch response history rather than receiving a task context signal. That difference alone does not establish firstness.
- **Few-shot Continual Learning: a Brain-inspired Approach (Wang et al., 2021)** uses interdependent fast/slow weights and consolidation: https://arxiv.org/abs/2104.09034 . Fast/slow memory is not itself novel.
- **HTM sequence memory (Cui et al., 2015/2016)** studies continuous unsupervised Hebbian-like sequence prediction with sparse temporal codes on synthetic and real streams: https://arxiv.org/abs/1512.05463 . This is important prior art when claiming that temporal context can emerge without task IDs.
- **MemDA (2023)** uses memory-based drift adaptation for urban time-series forecasting: https://arxiv.org/abs/2309.14216 . Memory-based adaptive forecasting is also existing territory.
- **Continuous Evolution Pool (2025, updated 2026)** identifies recurring concepts with lightweight statistics and maintains specialized forecasters without historical raw replay: https://arxiv.org/abs/2506.14790 . This is a particularly relevant competitor for statistical context, specialization and replay-free recurring drift.
- A recent neuroscience paper, **Assembly-based computations through contextual dendritic gating of plasticity (2026)**, is an additional overlap to read in full: https://www.sciencedirect.com/science/article/pii/S0896627326005830 . Search metadata describes contextual inhibition and dendrite-specific plasticity; full text was inaccessible in this check, so do not assert finer differences yet.

## Candidate contribution to investigate

A compact branch-specific runtime state, updated locally from target-free predictive response statistics, could form a task cue that selects computation and limits interference in slow weights without task IDs or supplied boundaries. The interesting contribution would be the exact update/competition mechanism and its demonstrated behavior—not the labels "dendritic", "short-term memory", or "task-free" by themselves. The current pilot uses competing predictive traces, not the original proposed neighbor-message communication/orchestrator mechanism.

We have not established that this exact mechanism is new. Nor have we established a consistent advantage over a comparably capable shared router. A central tensor storing the same per-branch statistics is equivalent; no anatomical necessity should be claimed. Earlier shared comparisons also differ in coordinate representations and effective state counts.

## Evidence that could justify a major-conference submission

1. State one precise claim and compare the actual mechanism to the closest algorithms, including recurrent memory, statistics-driven routing/MoE, fast-weight memory, and online forecasting/drift methods—not only a plain MLP.
2. Separate predictive context from retention: show reduced interference/forgetting on controlled recurring regimes, alongside naturally evolving streams. Better forecast MSE alone can arise from improved temporal prediction without protecting prior knowledge.
3. Isolate the memory rule, branch specificity, sparsity, and local communication if included. Match parameters, effective state, observed history, tuning effort and compute; report transition errors and failure cases.
4. Confirm on several independent datasets/settings and fresh seeds, with standard protocols or clearly specified custom streaming protocols. Scale enough to show the effect is not unique to a tiny hand-designed model.
5. Supply a convincing explanation of why it works and where it fails. Theory is one route; careful causal ablations and diagnostic evidence are another. A simple mechanism with robust evidence can be a contribution; unnecessary complexity is not required.

Current status: promising synthetic evidence for persistent predictive context; original-vs-linear improvement modest; shared memory advantage inconsistent; abrupt transition costs remain; no major-conference-strength novelty claim yet. If electricity is negative, retain the result and investigate which synthetic assumptions failed rather than searching test settings until a favorable number appears.
