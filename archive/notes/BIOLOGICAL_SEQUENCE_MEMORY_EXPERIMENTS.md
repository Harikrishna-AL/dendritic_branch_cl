# Sequential memory in published fly models

Status: initial Incentive Circuit retention pilot completed, 2026-10-17.
Ten independent input-noise seeds, both punished-odor assignments, four arms,
blank/extinction/reversal conditions, and activity/weight resets. See
`reports/biological_memory_20261017/REPORT.md` and
`bdh_cl/scripts/biological_memory_pilot.py`. Full-connectome and learned-order
experiments below remain unrun.

## Question

After a sequence of sensory events, what information remains accessible, for how
long, and in which state variables? Separate fading activity, learned associations,
and memory of event order. Recognition of A after A→B is not evidence that the
model remembers the order A→B.

## Published models

1. Shiu et al. (Nature, 2024): connectome-constrained whole-brain leaky
   integrate-and-fire model. Use the original fixed-weight model to test activity
   memory before adding any plasticity. Anatomical constraints do not imply
   detailed biophysics or comprehensive biological memory mechanisms.
   Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/
   Code: https://github.com/philshiu/Drosophila_brain_model
2. Huang, Luo et al. (Nature, 2024): mushroom-body model with interacting γ1,
   α2 and α3 learning modules. First candidate for mechanistic short/long memory
   experiments; reproduce its native conditioning results before extending it.
   Paper: https://www.nature.com/articles/s41586-024-07819-w
   Code: https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model
3. Gkanias et al. (eLife, 2022): incentive circuit with acquisition,
   consolidation and forgetting mechanisms. Independent model for checking
   whether findings depend on a particular memory rule.
   Paper: https://elifesciences.org/articles/75611
   Code: https://github.com/InsectRobotics/IncentiveCircuit

The Eon embodied demonstration builds on the Shiu-style brain model and
body-level controllers. Embodiment is optional for these initial memory assays:
https://eon.systems/updates/embodied-brain-emulation

The supplied Cell Reports Methods URL could not be accessed or independently
identified during this review. Do not infer its model or results from its URL:
https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(26)00339-5

## Experiment 1: intrinsic history and sequence order

Deliver A→B and B→A with matched pulse counts, amplitudes, durations and total
elapsed time. Use published sensory populations and native stimulus settings
where available; document synthetic input encodings separately. Also run A-only,
B-only and no-stimulus trials.

After a blank delay, deliver an identical probe P. For an initial LIF sweep use
delays 20, 50, 100, 200, 500, 1000 and 5000 ms; validate timestep convergence and
extend delays only if signal persists. These are proposed assay settings, not
published physiological retention times.

Read out population spikes and native circuit outputs during both the blank and
probe periods. Include A→B versus C→B with matched B and matched time since B:
this isolates information about earlier stimuli from identity of the last input.
Counterbalance stimulus identities and use a second probe identity to check
probe specificity.

An observer decoder may quantify information but is not part of the memory
mechanism. Fit it on separate training trials; select settings on validation
trials; score held-out trials with fresh stimulus noise. Split complete trials,
not overlapping time bins. Fit all feature scaling only on training trials.
Report decoder performance separately from changes in native model output.

## Experiment 2: associative retention and interference

Use each plastic model's native odor and reinforcement interface. Begin with
its published acquisition/retention protocol, then train A+, B− sequentially.
Test A and B without reinforcement on separate copies of the same checkpoint.
Compare a blank interval with an equally long C+, D− interference block.
Then reverse the original associations (A−, B+) and measure old-association
retention, new acquisition, and recovery after another blank interval.

Use model-native time units. A trial index is not a biological hour. Compare
massed versus spaced training with matched exposures and reinforcement, and
include a control matched for total elapsed time. Do not call persistence
biological long-term memory unless the model represents the relevant timescale
and mechanisms; otherwise report durable simulated synaptic retention.

## Experiment 3: learned order, beyond odor association

Train AB+ and BA− with the same constituents and total stimulation. At test,
present full sequences without reinforcement and also test isolated A and B.
Counterbalance rewarded order. A model must receive separable temporal events;
do not encode the sequence label directly into its input. If it cannot acquire
this task, report that outcome rather than adding a sequence-aware readout and
crediting the biological circuit for learning it.

## Causal controls

- Reset fast state after acquisition while preserving learned synapses. Reset
  voltages, synaptic currents, pending spike queues, refractory state and any
  relevant activity traces; state exactly which variables are reset.
- Restore baseline synapses while preserving compatible fast state. Apply at a
  defined point and report intervention transients.
- Disable plasticity during acquisition, using otherwise matched stimulation.
- Remove reinforcement or shuffle its pairing, preserving reward counts.
- Ablate the published intermodule pathway implicated in memory transfer, after
  reproducing its native effect; include a matched control intervention.
- For the connectome model, compare against sign/degree-preserving rewiring,
  checking firing rates and stability so a silent control is not interpreted
  as a selective memory deficit.

Run probes from cloned checkpoints. If normal model plasticity stays active
during a probe, quantify its effect on a separate branch; repeated testing can
cause extinction or further learning. Freeze only learning updates for an
observer-only diagnostic and identify this intervention explicitly.

## Reporting and execution order

First reproduce native results with pinned source revisions and recorded data
versions. Audit dependencies, runtime and state access before adapting code.
Then run the inexpensive mushroom-body retention pilot, followed by whole-brain
activity assays and order learning. Keep published models separate from any
new hybrid model with added plasticity.

Use a 3-seed smoke pilot to check acquisition, stability and resource use;
confirmatory runs should use at least 10 independent realizations where the
model is stochastic, with paired conditions and confidence intervals over
independent realizations. For deterministic models, vary meaningful input or
parameter realizations rather than counting identical reruns as replicates.
Keep a single connectome's trial uncertainty distinct from between-animal
uncertainty. Predefine delays, metrics and comparisons before confirmatory runs.

Save acquisition curves; retention versus delay and interference count;
old/new association performance; order discrimination; activity and weight
trajectories; reset/ablation effects; runtime and peak memory. Quantify forgetting
relative to each condition's own acquisition level, alongside absolute scores.

The useful link to this repository is a comparison of where memory lives:
ongoing activity, temporary synaptic state, durable weights, or interacting
circuit compartments. Match tasks and information access before comparing to
BDH branch mechanisms; do not treat neuron counts as a sufficient budget match.
