# Short-term branch state: causal frozen-network evaluation

Exploratory inference-only test on six frozen sparse-MNIST MLP checkpoints (three seeds × two training orders). All 10,000 official test images appear once per stream. Models remain frozen. This evaluates inference-state context retention and switch recovery, not training-time weight forgetting.

| State policy | Intervention | Persistent context accuracy | Random mixed accuracy |
|---|---|---:|---:|
| none | retain | 66.70 ± 1.01% | 66.70 ± 1.01% |
| magnitude | retain | 68.21 ± 2.28% | 61.43 ± 2.61% |
| magnitude | reset_each | 66.70 ± 1.01% | 66.70 ± 1.01% |
| magnitude | shuffle_branches | 64.52 ± 1.32% | 64.47 ± 1.15% |
| pseudo | retain | 78.54 ± 1.73% | 49.46 ± 0.92% |
| pseudo | reset_each | 66.70 ± 1.01% | 66.70 ± 1.01% |
| pseudo | shuffle_branches | 57.51 ± 2.18% | 58.48 ± 2.13% |
| supervised | retain | 95.50 ± 0.27% | 59.98 ± 0.45% |
| supervised | reset_each | 66.70 ± 1.01% | 66.70 ± 1.01% |
| supervised | shuffle_branches | 56.99 ± 2.11% | 58.50 ± 2.10% |

## Selected state parameters

- magnitude: decay=0.9, state strength=0.5; validation persistent-context accuracy=67.57%.
- pseudo: decay=0.9, state strength=0.9; validation persistent-context accuracy=77.51%.
- supervised: decay=0.9, state strength=0.9; validation persistent-context accuracy=95.75%.

## Adaptation after context switches

| Policy, state retained | First 20 queries after switch | Queries 101–200 after switch | Mean gate overlap between contexts |
|---|---:|---:|---:|
| none | 66.43% | 66.65% | 0.887 |
| magnitude | 64.32% | 68.58% | 0.890 |
| pseudo | 63.15% | 80.06% | 0.812 |
| supervised | 75.90% | 97.66% | 0.722 |

Pseudo-error state paired gain in persistent streams: **+11.85 pp**, descriptive seed-level 95% t interval [+8.51, +15.18] pp.

## How it works

Current-image baseline produces magnitude-soft branch weights and a ten-class prediction. For each neuron, replace only that neuron’s mixture with each branch and measure cross entropy against the baseline’s own predicted class (not the true label). Center and scale the four errors within that neuron, then softmax the negative standardized errors to obtain branch preferences. EMA stores these preferences after prediction. On the next image, combine its current-image weights with the previous EMA: gate = .1 current + .9 state for the selected pseudo policy. No task boundary, task ID, prototype, query label or stored example enters this label-free state. Fixed stateless probes define update preferences; probes are not recomputed under the current state.

Magnitude state instead averages recent magnitude-soft branch weights. Supervised control uses true-label counterfactual errors after each prediction: its accuracy is not label-free. It is a matched feedback control, not the exact original training error-trace algorithm, because it stores standardized branch preferences.

Persistent stream rotates 200-image blocks from each digit pair; context recurs. Mixed stream randomly permutes the same images. Labels construct benchmark order and score predictions but are not provided to label-free state. Retain keeps the causal state. Reset_each removes it before every prediction, recovering the stateless baseline exactly. Shuffle_branches independently permutes each neuron’s historical branch assignments before use while maintaining the unshuffled underlying EMA update.

State has 168×4 float32 values: **2,688 bytes**. It requires no new trainable parameters and stores no examples. Counterfactual probes add compute, not included in the state-memory count. Core model remains 134,074 parameters and about 1.593 MB of persistent tensors. Implementation precomputes independent per-image probes for experimental efficiency; streaming deployment can compute each probe on arrival. The algorithm only reads the current probe and previous state.

## Limits

- All checkpoints were trained with supervised error-trace selection and the auxiliary loss. This is not a comparison of networks trained without traces versus label-free traces. It isolates the additional inference-time state mechanism.
- No new training/weight-forgetting experiment was conducted. Early-versus-late switch accuracy measures adaptation, not changes in long-term weights. Context gate overlap is mean histogram intersection (1 means identical average gates), not proof of isolated subnetworks.
- State parameters were selected on the original validation split for persistent streams. Mixed-stream scores use those same parameters without retuning. The beneficial setting exploits temporal context persistence; random mixing is a failure case, not something hidden by the summary.
- Pseudo-labels can be wrong, and this does not establish correctness of the inferred context or novel-class learning. Three seeds and one stream construction per checkpoint leave uncertainty; alternative context durations and datasets remain untested.
- Accuracy ± SD averages the two orders within each of three seeds. Repeated streams share query images. Stateless accuracy rounds to 66.70%, matching the earlier image-only result; comparisons use the matched scoring implementation.
- Three tests passed: state reset equals stateless predictions, label-free policies ignore supervised feedback arrays, and each stream contains every image exactly once. All original model tensors were checked unchanged.
