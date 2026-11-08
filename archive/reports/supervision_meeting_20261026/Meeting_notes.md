# Meeting preparation: FOSTER and 15-credit PRO

## What to achieve

Leave with (1) Subhendu's supervision decision, (2) a FOSTER completion checklist, (3) one agreed PRO research question, and (4) written separation of paid and assessed tasks, dates and hours. Do not try to convince him that the project has already solved continual learning.

## A 60-second opening

“My FOSTER project asks whether short-term state in dendritic branches can help separate situations and reduce interference. I have implementations and a set of controlled experiments, but I cannot yet claim a branch-specific continual-learning advantage.

“The clearest diagnostic is that historical branch configurations can recover earlier skills using unchanged final weights. Automatic retrieval is possible with extra input memory, but simpler controls remain strong. Several local-memory, recurrent and energy-based rules did not establish the mechanism we wanted.

“I would like to finish FOSTER with a reproducible evaluation of the original hypothesis, including its negative results. For the 15-credit project, I propose a distinct study of branch-local selective learning, with matched sparse-network controls and clear stopping criteria. I would like your advice on that question and on the formal division of work.”

## Five-minute presentation order

1. **Motivation, 30 seconds:** separate within-neuron computation and short-term state may influence interference. This is a hypothesis, not an established benefit of our implementation.
2. **Evidence, 90 seconds:** fading-state core result is not yet positive; oracle recovery suggests an access problem in one model; cue retrieval improves but costs memory and does not beat the direct classifier; direct inhibition is trainable but does not retain and sometimes fails acquisition. Use the first PDF page, not a long chronological list of experiments.
3. **FOSTER, 60 seconds:** baseline audit, consolidated comparisons, explicit state/feedback accounting, report and runnable handover. Ask what is still required and how many paid hours remain.
4. **PRO, 90 seconds:** investigate selective plasticity beyond generic slowing. First stabilize acquisition, then compare local/shared/shuffled state and ordinary sparse controls. Confirm on fresh seeds/orders and a transfer setting. One rule family, not multiple open-ended algorithm searches.
5. **Agreement, 30 seconds:** supervision, scope, assessment deadlines, task allocation and next review date.

## Answers to likely questions

**“What is the new contribution?”**

Not “adding branches.” The proposed contribution is a tested learning mechanism and causal evidence about when branch-local plasticity reduces interference beyond ordinary sparsity, additional state or slower learning. That advantage has not yet been demonstrated. Literature review must establish novelty before presenting it as a paper contribution.

**“Have you shown the grant hypothesis?”**

Not conclusively. The original activity-trace fading-memory test did not beat its competition-only control. Prototype memories, cumulative covariance and oracle route banks are different mechanisms; they cannot be substituted as evidence for short-term fading state.

**“What does the 98% result show?”**

In the original experiment, old configurations restored 98.02% while final weights/head remained fixed, versus 19.40% under the final configuration. Selection used the true digit pair and saved gates. It shows accessible earlier skills under assisted configurations, not autonomous continual learning or superiority to ordinary MLPs. Do not generalize that exact number across later model variants.

**“Does the MLP lose all its features?”**

No. Earlier supervised probes recovered 82.15% from final MLP features using old training labels, and random features reached 83.43% in that condition. Both are assisted diagnostics. We must distinguish information availability, useful learned representation and automatic access.

**“Could Adam explain the failures?”**

Possibly part of the behavior, but we have not isolated that. Adam can train the architecture jointly. Sequential optimization also produced acquisition failures. Compare Adam and momentum-free SGD with separate validation selection and matched new-task learning; a slower learner can appear to forget less simply because it learns less.

**“Why not focus on the energy idea?”**

The bounded prediction-energy pilot did not beat the simpler cue baseline; no energy qualified for descent under the predeclared screen. Energy-based retrieval remains an alternative research question, but should not run in parallel with selective plasticity inside this project. Ask Subhendu to choose the direction he considers scientifically strongest.

**“What would count as a negative outcome?”**

A stable, properly controlled study showing that shared regulation or an ordinary sparse network explains the gains. The academic report would then document the mechanism limit and the conditions causing failure. Whether this supports a publication depends on the strength and generality of the findings; paper submission is the goal, acceptance is not guaranteed.

**“Are the two projects genuinely separate?”**

This is a proposed allocation, not a retroactive declaration. Confirm actual paid tasks and dates. Reuse the code as disclosed background if agreed, but do not bill PRO work to FOSTER or submit paid deliverables again as new assessed work. If any proposed new mechanism is already covered by the funded scope, change the allocation or the PRO question with the supervisors.

## Decisions to write down

- Supervisor and whether Prof. Färber should approve the final scope.
- Exact PRO title, start/end date, assessment requirements and workload expected for 15 credits.
- Funded dates, paid hours already used and remaining hours.
- Named FOSTER deliverables and acceptance criteria.
- Named PRO work packages and the transition date, or separate concurrent work logs.
- Main setting: same-label/domain-incremental mechanism study, with strict class-incremental transfer clearly separated.
- Initial compute allowance and cadence of supervision (suggest a short weekly/fortnightly meeting).
- First milestone: a literature map, frozen baseline and a small acquisition-stability protocol.

## After-meeting allocation record

| Task ID | Concrete task/deliverable | FOSTER or PRO | Actual dates | Actual hours | Approval / provenance |
|---|---|---|---|---|---|
| F1 | Baseline audit and documented state/feedback | Proposed FOSTER | To agree | To record | To confirm |
| F2 | Remaining funded comparisons | Proposed FOSTER | To agree | To record | To confirm |
| F3 | Grant report, figures and handover | Proposed FOSTER | To agree | To record | To confirm |
| P1 | New experimental protocol and stability study | Proposed PRO | To agree | Separate record | To confirm |
| P2 | New selective-plasticity rule and controls | Proposed PRO | To agree | Separate record | To confirm |
| P3 | New confirmation, analysis and course report | Proposed PRO | To agree | Separate record | To confirm |

Count actual human work according to the agreed rules; automated run time is not automatically paid working time. Keep the baseline commit/run IDs and contribution provenance.

## What to bring

- The four-page PDF handout.
- The editable plan and these notes.
- The original approved grant proposal, funding dates/hour records and PRO module description, if available.
- The local evidence reports only as backup; do not spend the meeting explaining every failed variant.

No email, upload or submission has been made.
