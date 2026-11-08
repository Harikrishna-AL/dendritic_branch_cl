# Context-driven circuit recruitment

## Hypothesis and operational model

The strongest activation may indicate a common input feature rather than the task. Test whether selecting a leader by compatibility with an explicit context recruits more distinct circuits and preserves sequential computations.

Use the same noisy two-signal task and fixed readout as the preceding pilot. Two layers have 32 neurons and four branches each. The fixed output projection alternates positive and negative signs so any contiguous 16-neuron circuit contains equal positive/negative coordinates. All arms use the same new projection, initialization, allocated parameters, training examples, and minibatch order. This projection differs from the previous pilot's first-half-positive arrangement; use only within-experiment comparisons for causal conclusions.

## Four arms

- **Dense:** all 32 neurons participate; capacity control, twice the forward-active neurons of routed arms.
- **Separated:** explicit cue selects one of two disjoint 16-neuron groups in each layer. Positive control; circuit identity is supplied, not learned. Inactive rows receive zero gradients, and Adam resets at each task boundary. No separate protection objective.
- **Raw leader:** strongest activating branch selects its neuron as leader; recruits a contiguous neighborhood of 16 neurons.
- **Context leader:** learned branch-to-cue affinities choose the leader; recruits the same-size neighborhood. The explicit cue is supplied to each layer. Affinity scores depend only on cue, so the leader is persistent for a context once weights are fixed, rather than selected anew from input content.

Recruitment uses fixed ring neighborhoods extending seven positions in one direction and eight in the other. These could be reached with eight nearest-neighbor hops; implementation applies the resulting mask directly. This is a static recruitment abstraction, not a simulation of messages spreading in time. It replaces the previous sigmoid message-amplitude mechanism. The forward route has one actual winner; a straight-through softmax at temperature 0.2 supplies gradients to learn selection. This surrogate can update parameters associated with forward-inactive routes in learned arms. Hard activity masks also restrict ordinary activation gradients; the separated arm provides implicit plasticity isolation. Do not describe this as a pure messaging-only experiment.

All arms allocate identical branch weights and affinity tables, but dense and raw/separated arms do not use learned cue-affinity parameters to select circuits. A later direct-context-mask control is required to establish that a single orchestrator helps beyond learned context gating.

## Protocol

1,024 training, 512 validation, and 1,024 test examples per context, using generator seed offsets 1000/2000/3000. Scratch initialization only. Twenty epochs per context, Adam 0.003, fresh optimizer at each task boundary. Joint control matches total example presentations and updates, without a boundary reset. No tuning or selection after seeing test scores.

Initial three paired seeds (0/1/2), then five independent confirmation seeds (3–7) with identical settings, plus reversed-order runs on initial seeds. Record task-specific stage MSE, joint MSE, cue sensitivity, neuron-selection profiles, and winner distributions. Circuit overlap is overlap of mean selection profiles divided by selected population size. Compare old-task retention without sacrificing new-task acquisition.

Tests verify disjoint positive-control routes, exact preservation of old predictions after new-context training in the separated arm, fixed readout, exactly 16 forward-selected neurons for learned routes, and nonzero cue-affinity gradients only for the currently supplied cue.

Results are under `results/context_circuit_pilot_20261010*`. Source is `bdh_cl/scripts/context_circuit_pilot.py`.

## Completed result

Initial, confirmation, and reversed-order runs completed: 44 configurations plus their joint controls. Context-compatible leaders reduce old-task error while preserving new-task learning in both orders. Forward eight-seed old-task MSE is 0.0697 versus 0.5057 for raw leaders; reverse three-seed MSE is 0.0453 versus 0.5265. Explicit separation stays near 0.010. One learned-leader seed collapses to the same first-layer circuit and performs poorly. This motivates a direct context-mask control and a separately identified allocation intervention before larger tasks. See `results/context_circuit_pilot_20261010/confirmation.md` and `interpretation.md`.
