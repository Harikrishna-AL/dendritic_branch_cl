# Two-layer scaling check for predictive branch memory

## Architecture and controls

First layer: eight neurons, four branches each, random learned input mixtures, hard predictability competition, tanh of selected output. Second layer:32learned predictor branches receiving the eight first-layer outputs, hard competition producing one scalar. All arms have384trainable parameters.

Compare branch-local lag-product/energy memory, stateless eight-observation history-based routing, and the same predictor computation with centrally stored memory. The central router is a re-expression of the same computational graph, not an independently designed RNN or general dense network. Identical branch/central predictions are expected by construction and verified in tests; this checks whether memory location alone adds an advantage.

Branch and central arms have identical effective runtime state:3scalar traces perbranch×64branches×16stream lanes=3,072FP32values,12,288bytes, plus64seen flags. History-only allocates the same buffers for software convenience but has no functional persistent state. That inherent functional-state difference is the variable under test and must not be described as matched effective memory between stateful and stateless arms.

## Protocol

Same continuous AR(1) two-regime task,16independent lanes, latest eight observations available. No task ID, boundary flag, or optimizer reset. Both A→B and B→A training orders; three paired seeds;4,096steps perregime. One Adam,lr0.003,update everyeightobservations,gradient clipping5. Fast memory reads before target-free writes and persists through switches and weight updates. Fixed constants, no test tuning.

Strong learning controls:singleA4,096,singleB4,096,and joint repeatedA/B4,096. Joint has fewer total presentations than the8,192step sequential run. The same single/joint controls are repeated under each order for bookkeeping, not independent evidence.

At frozen checkpoints run independent A/B/A/B and reverse B/A/B/A evaluation streams,256steps pervisit. Reset memory only at independent stream start. Report first32steps separately, then late error. Erasure resets both memory layers every step. Record first-layer per-neuron branch usage and second-layer predictor usage; measure overlap across regimes.

Stateless second-layer histories are reconstructed with current weights and a first-layer gate inferred from the entire observed eight-step window. This uses only observations already available at the prediction time, not future targets, but is not the exact historical trajectory of runtime first-layer routing. Stateful deeper memory instead tracks actual prior outputs, including representation changes caused by learning.

## Cost measurements

Warm CPU inference with two threads, batch16stream lanes. Original benchmark includes Python memory tracing; a separate post-training benchmark removes profiling/tracemalloc, measures three300-step repeats percheckpoint, and saves medians. Native CPU profiler memory timeline on three inference steps reports tracked peak live tensor allocations; this excludes allocator/process/Python overhead and may not track every preexisting allocation. Neither is a GPU or training-memory benchmark. Exact buffer bytes are reported separately.

Current code recomputes historical projections even in memory arms where they are not used for selection. It is unoptimized; timing cannot establish deepGPU-scale overhead. Cost comparison concerns this toy implementation only.

Tests verify384parameter equality, identical effective branch/central state budgets and numerical outputs, causal writes, finite gradients reaching both learned layers, and erasure of both states. Source: `bdh_cl/scripts/deep_predictive_memory_pilot.py`;profilers: `profile_deep_memory.py` and `benchmark_deep_memory.py`. Results: `results/deep_predictive_memory_pilot_20261011`.

## Completed outcome

Eighteen configurations plus learning controls completed;three tests pass. Branch-memory old-task MSE0.405/0.375 in the two training orders versushistory1.098/1.080. Central memory matches exactly. First-layer route overlap~7–8%;second~24–33%. Frozen memory erasure worsens error. Untraced CPU inference~26%slower in this unoptimized toy;state12,352bytes. Retention advantage survives depth, but seed variation and incomplete old-task recovery remain. See the saved interpretation and profiling metadata.
