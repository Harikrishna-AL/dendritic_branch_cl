# Fly memory pilot

Completed simulation, not a whole-brain or MaleCNS experiment.

Published Incentive Circuit revision `1610c80072fe8bb59bd397e7a61f716393a509b9`; 10 noise seeds, both punished-odor assignments per seed.

## Protocol and checks

Six A/B conditioning cycles; three native timesteps per trial. Shock coincides with the final odor timestep. Original Python equations and default manuscript-example parameters are unchanged. Trials are model units, not biological hours.

All three original reversal/extinction/unpaired routines execute. The custom stimulus driver matches their complete activity and weight histories exactly. This verifies the driver, but is not an independent quantitative reproduction of the paper's biological results.

Probes use separate clones with plasticity disabled; assertions check frozen weights and source-checkpoint preservation. Positive contrast means stronger avoidance of the original punished odor; negative means the opposite. This is a circuit-response contrast, not percent correct or a calibrated behavioral probability.

## Endpoints

Mean contrasts across counterbalanced odors and seeds:

| Arm | Phase | Susceptible | Restrained | Long-term compartment |
|---|---|---:|---:|---:|
| full | acquisition | 0.463 | 0.786 | 2.059 |
| full | blank | 0.000 | 0.093 | 2.049 |
| full | extinction | 0.000 | 0.082 | 2.532 |
| full | reversal | -0.445 | -0.655 | 1.693 |
| no_ltm | acquisition | 0.463 | 0.816 | 0.000 |
| no_ltm | blank | 0.000 | 0.096 | 0.000 |
| no_ltm | extinction | 0.000 | 0.000 | 0.000 |
| no_ltm | reversal | -0.445 | -0.790 | 0.000 |
| no_plasticity | acquisition | 0.000 | 0.000 | 0.000 |
| no_plasticity | blank | 0.000 | 0.000 | 0.000 |
| no_plasticity | extinction | 0.000 | 0.000 | 0.000 |
| no_plasticity | reversal | 0.000 | 0.000 | 0.000 |
| unpaired | acquisition | -0.000 | 0.005 | -1.917 |
| unpaired | blank | 0.000 | -0.006 | -1.893 |
| unpaired | extinction | -0.000 | -0.082 | -2.532 |
| unpaired | reversal | 0.000 | -0.069 | -1.549 |

![Retention](retention.png)

## Interpretation and limits

In the full model, the susceptible contrast fell from 0.463 after acquisition
to effectively zero after 256 blank trials. The long-term compartment contrast
was 2.059 after acquisition and 2.049 after the blank interval (seed-bootstrap
95% interval 1.651–2.387 at the latter endpoint). Activity reset left this
contrast essentially unchanged; restoring baseline synapses reduced it to zero.
This supports synaptic storage in this simulated circuit.

After 256 reversed A/B cycles, the susceptible and restrained contrasts were
negative (−0.445 and −0.655), while the long-term compartment retained a positive
contrast (1.693). Thus different compartments expressed conflicting old and new
associations. This does not establish that the model's overall behavior is
optimal or that it remembers event order.

Four checkpoint-continuation and probe-isolation tests pass. All three native
schedule equivalence checks pass inside the experiment runner. The downloaded
upstream checkout has no tracked modifications.

Inspect compartments separately: an average can conceal opposite memories after reversal. Removing the long-term pathway also removes a direct contributor to the average, so that average alone is not evidence for a consolidation mechanism.

Unpaired shock is a timing manipulation, not a guaranteed zero-memory baseline: it can produce opposite-sign associations in this model. Plasticity-disabled acquisition is the no-learning control.

Activity-reset probes retain learned synapses; weight-reset probes preserve current activity. Both use baseline state as defined by the implementation. Reset results and seed-bootstrap 95% intervals are in summary.csv. Intervals describe input-noise realizations of this circuit, not biological animals or uncertainty in the model's mechanism.

Blank intervals include the upstream weak spontaneous KC noise and native synaptic updates. Repeated unrewarded odor exposure can change memory, so extinction is not equivalent to passive delay. Blank counts are single trials; extinction/reversal counts are two-trial A/B cycles and should not be compared as equal elapsed time.

This first pilot measures learned odor association retention, not learned event order. It uses a reduced biologically motivated circuit, not detailed neuronal biophysics or the Google/Janelia wiring map. Follow-up: validate paper figure trajectories, test matched-duration interference with additional odors, inspect causal pathway interventions, and then attempt AB versus BA learning.

Sources: [model and original code](https://github.com/InsectRobotics/IncentiveCircuit), [eLife paper](https://elifesciences.org/articles/75611), [Google MaleCNS release](https://www.research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).

Runtime: 79.9 s. Raw metrics, native histories, acquisition checkpoints and provenance are in `/Users/harikrishna/Documents/lossfunk/graph_cl/q_graph_cl/bdh-d/results/biological_memory_20261017`.
