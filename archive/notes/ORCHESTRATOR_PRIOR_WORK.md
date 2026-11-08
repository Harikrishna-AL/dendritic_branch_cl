# Preliminary novelty assessment

A successful experiment would establish a mechanism's value, not by itself establish novelty. This is a targeted initial literature check, not an exhaustive review.

## Closest work

[Active Dendrites / Avoiding Catastrophe (Iyer et al., 2022)](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.846219/full) already combines dendritic context gating with sparse subnetworks and continual learning. It includes context inferred using online input clustering when task information is absent. Thus dendritic context inference and task-specific subnetworks alone cannot support a new contribution.

[Context association in pyramidal neurons (Baronig and Legenstein, 2024)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1276706/full) derives a local apical plasticity rule that associates context with sensory features, clusters co-active inputs, and segregates contextual patterns. Continual performance also uses protection of previously used branches. This overlaps with learned branch selectivity and context-dependent recruitment, though it supplies a separate contextual stream.

[Functional Task Networks](https://arxiv.org/abs/2604.24637) is particularly close to the spatial recruitment portion: it optimizes component masks, spatially smooths them, and selects a fixed-capacity subnetwork. The [OpenReview manuscript](https://openreview.net/pdf?id=m415rVhE4i) describes task recovery from a labeled support batch. This must be compared carefully with any task-ID-free local recruitment claim. The initial arXiv listing and OpenReview manuscript use different titles; version correspondence needs checking before a full novelty claim.

[Local dendritic balance (2021)](https://doi.org/10.1073/pnas.2021925118) develops locally available dendritic coding-error signals for representation learning. Internal local errors themselves are not new.

## Candidate distinctive contribution

An input/history-selective branch acting as a persistent or stable layer leader, with explicit branch-to-branch local communication that recruits a circuit during both training and inference, may be a distinguishable combination. This is a hypothesis about novelty, not a verified novelty claim. Need to show an advantage over independent context gating, all-branch local communication, direct sparse gating, fixed/random leaders, and spatially smoothed masks, under matched resources. Also distinguish learning selection from direct task supervision and from protection of old parameters.

Current experiments do not establish these claims. Hard argmax, top-k, supervised backpropagation, and a fixed ring are engineering abstractions; avoid presenting them as biologically demonstrated circuitry.
