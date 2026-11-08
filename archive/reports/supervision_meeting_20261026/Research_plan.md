# Two projects, one research direction

Harikrishna Pillai | Discussion with Subhendu | 26 October 2026 | Proposal for agreement

> **Meeting objective:** agree the remaining FOSTER deliverables, a distinct 15-credit semester research question, supervision, and a written allocation of paid versus assessed work.

## The common scientific question

Can separating computation and short-lived state across dendritic branches help a network learn new situations without disrupting earlier skills? The current evidence motivates this question, but does not establish a uniquely dendritic continual-learning advantage.

| Project | Purpose | Scope boundary |
| --- | --- | --- |
| **FOSTER**; Paid grant; reported 6 h/week over 4 months | Implement and evaluate the original short-term branch-memory hypothesis. | Baseline implementation, controlled evaluation and honest documentation of positive and negative findings. |
| **PRO research project**; 15 credits; one complete semester | Investigate a new mechanism of selective learning beyond the agreed funded baseline. | Design and causally test branch-local plasticity; develop an independent scientific contribution and paper-oriented report. |

## What the experiments currently establish

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Fading activity memory [E1] | Synthetic MSE: competition only 0.725; branch memory + competition 0.761. Lower is better. | This tested short-term rule did not improve the control. It does not refute every branch-memory rule. |
| Historical configuration [E2] | Split MNIST: 19.40% automatic; 98.02% with historical gates and unchanged final weights/head. | Old skills were accessible under assisted configurations. True digit-pair selection and stored gates were required. |
| Automatic retrieval [E3] | MNIST: local cues 76.39%; + neighbors 77.48%; direct ridge 85.61%. | A small local-communication effect; extra memory is required. No corresponding gain on Fashion MNIST. |
| Input-driven branches [E4] | Joint: 96.30%; sequential: 16.26%. Matched MLP sequential: 20.14%. | End-to-end Adam training works jointly; sequential acquisition is sometimes unstable and retention is not improved. |

**Important:** these rows describe different models/protocols, not a single improvement curve. Fading state, cumulative input prototypes, optimizer moments and temporary gates are different forms of state. A retrieval gain from prototypes is not proof that fading branch memory reduces forgetting.

Evidence IDs: E1 = fading-state pilot (11 Oct); E2 = configuration/readout diagnostic (11 Oct); E3 = locked retrieval studies (24 Oct); E4 = input-driven inhibition study (26 Oct). Full local records are indexed in the editable plan.

**Proposed direction:** stop expanding inference-time routing searches. First separate stable learning from retention, then test whether branch-local control of plasticity has an advantage over equally sparse ordinary networks.


---

# FOSTER: complete the funded hypothesis

Working title: Learning without forgetting - testing short-term memory in neural networks

## Aim and minimum scientific deliverable

Evaluate whether fading branch state changes specialization, adaptation and interference relative to long-term weights alone. A well-controlled negative result is a legitimate scientific outcome; whether the proposed deliverables satisfy the grant must be agreed with the supervisors.

> **Keep the original hypothesis explicit:** short-term state must actually evolve during the stream and fade. Input prototypes or saved configurations may be diagnostic extensions, but cannot substitute for the grant-core fading-memory test.

| Work package | Deliverable and evidence |
| --- | --- |
| F1. Audit and define the baseline | Document the slow weights, fading state, update timing and available feedback. Verify read-before-write, no task-ID/boundary input, and what persists during evaluation. Freeze a reproducible baseline. |
| F2. Consolidate the core comparisons | State on/off and competition on/off; branch-specific versus matched shared state; ordinary-network controls. Use existing synthetic prediction and chronological electricity experiments, plus Split MNIST diagnostics where relevant. Rerun only unresolved checks. |
| F3. Explain outcomes and costs | Report acquisition, adaptation after switches and retention separately. Count trainable parameters, fast state, auxiliary storage and runtime. Mark any oracle routing or old-label readout probe as assisted analysis. |
| F4. Handover and dissemination | Short technical report, a small set of clear figures, runnable commands and experiment index. Record supported claims, limitations and which new research questions remain open. |

## Provisional closure effort - subject to remaining paid hours

| Task | Illustrative remaining effort |
| --- | --- |
| Baseline/provenance audit | 6 hours |
| Only essential control checks | 8 hours |
| Results and figures | 6 hours |
| Report and handover | 4 hours |
| Total | 24 hours = 4 weeks at 6 hours/week, only if available in the grant balance. |

The original reported commitment is four months at six hours per week. This is not a request to restart or extend that period. Actual funded dates, work already charged and remaining hours are still to be checked. Reduce or reorder closure tasks if less time remains.

## Acceptance and scope to agree

Request agreement on a concrete completion checklist rather than a positive accuracy target. Confirm whether the compact benchmark suite is sufficient under the approved proposal; larger CIFAR/CORe50 commitments should be explicitly prioritized or revised, not silently dropped. Do not treat the current research history as automatic proof that FOSTER is complete.

The new selective-weight-update mechanism is proposed for PRO only if the supervisors agree it lies beyond the funded scope; otherwise the task split or research question must change.


---

# PRO: selective learning in branched networks

Proposed 15-credit semester project | A scientific extension beyond the agreed FOSTER baseline

> **Primary research question:** Can a branch-local plasticity mechanism preserve input-selective computations during task-ID-free continual learning better than an ordinary network with comparable sparsity, memory and compute?

## Proposed contribution

Develop and test one bounded family of branch-local learning controls, rather than add another configuration retriever. Short-lived activity/error state would regulate **where learning is written**, alongside slower weights. The exact rule should be selected after a focused literature review. A scalar activity-based slowdown already failed to show a specific benefit, so it remains a negative control rather than the proposed novelty.

Use global backpropagation as the initial teaching baseline. Replacing all credit assignment with a biologically local rule is outside the minimum project. Biological inspiration is a design motivation; any plausibility claim must specify the actual signals available to each branch.

## Three linked experiments

| Stage | Question and discriminating comparison |
| --- | --- |
| P1. Establish stable learning | Compare Adam and momentum-free SGD with separately selected learning rates; check initialization/normalization of the soma. Match joint and current-task learning before interpreting forgetting. Keep all failed runs. |
| P2. Test selective plasticity | Choose at most two rule candidates. Compare branch-local state with shared state, shuffled assignments and state off. Include an equally sparse/gated ordinary network. Measure old-loss change at a matched amount of new learning. |
| P3. Test transfer and access | Confirm locked settings on fresh seeds and both orders. Measure retention and automatic inference; use probes or historical configurations only as explicitly assisted diagnostics of what remains accessible. |

## Benchmarks and controls

**Mechanism:** a controlled sequential feature-combination task with adjustable overlap. **Primary image setting:** task-ID-free domain-incremental MNIST/Fashion MNIST, keeping the output labels fixed to reduce new-class readout confounds. **Secondary transfer:** strict Split MNIST and, if feasible, Split Fashion MNIST with all ten outputs eligible. Report domain- and class-incremental claims separately.

Use cached data and small models initially. A larger image benchmark or shared pretrained encoder is optional after the mechanism passes; any pretraining exposure must be declared and identical for controls. Frozen/random-feature and direct-classifier references remain important wherever comparable.

## Decision criteria before scaling

**Learning gate:** target acquisition within two percentage points of the matched control on validation data. **Mechanism gate:** a retention/interference benefit over shared, shuffled and ordinary sparse controls, not merely over unrestricted Adam. **Confirmation:** lock settings before five fresh seeds and both orders; give paired seed-level intervals and at least one transfer condition. These are proposed criteria for agreement, not achieved results.

**Outputs:** a new rule/protocol, causal ablations, full resource accounting, reproducible code and a paper-oriented report. A precise limitation or negative mechanism result is the fallback scientific outcome; publication acceptance is not a promised deliverable.


---

# Semester milestones and agreement

Dates, module requirements and the task allocation require supervisor confirmation

| Core weeks | Milestone / review decision |
| --- | --- |
| 1-2 | Literature map, baseline handover, hypotheses and frozen evaluation protocol. Agree what is new PRO work. |
| 3-5 | Stable acquisition: optimizer and soma controls. Go/no-go before interpreting retention. |
| 6-9 | At most two plasticity candidates; local/shared/shuffled and sparse-network pilots. Stop uninformative sweeps. |
| 10-12 | Locked fresh-seed/order confirmation and one dataset/setting transfer. |
| 13-15 | Causal analysis, parameter/state/compute accounting and stress tests. |
| 16 onward | Draft course report and manuscript; use the remaining semester for feedback, revisions and any justified follow-up. Formal module deadline takes precedence. |

## Proposed task boundary - not a retroactive allocation

| Work | Allocation to confirm |
| --- | --- |
| Existing implementation/pilot history | Background with recorded provenance. Confirm actual funded tasks/hours; do not simply relabel already paid work as assessed work. |
| Baseline audit, funded comparisons, FOSTER report | FOSTER work packages F1-F4, within agreed paid hours. |
| New rule design and independent causal study | PRO work packages P1-P3, after or alongside FOSTER only with explicit task separation. |
| One eventual paper using both sets of findings | Possible shared scientific output; identify each contribution and funding acknowledgement. Paid work is not submitted again as new assessed work. |

## Decisions requested in this meeting

1. Will Subhendu supervise the 15-credit project, and what are its report, presentation and submission requirements?
2. What remains necessary to finish FOSTER, and how many paid hours remain?
3. Is selective plasticity the right PRO question, or should the earlier retrieval-focused proposal be retained instead? Choose one primary direction.
4. Which tasks, start/transition date and separate work logs document the allocation?
5. Agree a regular check-in, available compute and a paper target after the first mechanism review.

## Selected starting literature

Poirazi & Papoutsi (2020), dendritic computation: [doi:10.1038/s41583-020-0301-7](https://doi.org/10.1038/s41583-020-0301-7)
Legenstein & Maass (2011), branch-specific plasticity and binding: [doi:10.1523/JNEUROSCI.5684-10.2011](https://doi.org/10.1523/JNEUROSCI.5684-10.2011)
Cichon & Gan (2015), branch-specific plasticity/interference: [doi:10.1038/nature14251](https://doi.org/10.1038/nature14251)
Ashley et al. (2021), optimizer effects on forgetting: [arXiv:2102.07686](https://arxiv.org/abs/2102.07686)
Chavlis & Poirazi (2024), parameter-efficient dendritic networks: [arXiv:2404.03708](https://arxiv.org/abs/2404.03708)


## Evidence index for the meeting

E1 - Grant-core fading-state pilot: [BRANCH_SHORT_MEMORY_EXPERIMENTS.md](../../BRANCH_SHORT_MEMORY_EXPERIMENTS.md).

E2 - Historical configuration and readout diagnostics: [summary](../../results/split_mnist_retention_diagnostic_20261011/summary.md). This also reports that an MLP frozen-feature readout can recover 82.15% using all old training labels; random features reach 83.43%. Do not claim that an ordinary MLP necessarily lost all information.

E3 - Confirmed automatic retrieval and costs: [24 October report](../overnight_branch_research_20261024/MORNING_REPORT.md). Separate cohorts support the low-memory cue gain (59.69% to 66.29%) and the small communication gain (76.39% to 77.48%). They are not a single paired improvement.

E4 - Direct input-driven inhibition with Adam: [26 October report](../input_driven_inhibition_20261026/REPORT.md).

Other bounded negative directions: [recurrent gate updater](../recurrent_branch_gate_20261026/REPORT.md), [energy-ranking pilot](../branch_energy_landscape_20261026/REPORT.md), [activity-regulation controls](../branch_binding_interference_20261025/REPORT.md).

These local reports remain the evidence records. No claim of new independent replication is made by this planning document.
