# Learning branch specialization from random sensory mixtures

Follow the positive preassigned-channel proof of mechanism with no sensory channel assigned to a branch. Thirty-two branches begin with random trainable mixtures of both current inputs, plus bias:96parameters in each arm. A selected branch predicts the next slow-process observation. Branch-local fast state stores previous output, lag covariance, and energy (three scalars per branch/lane). Memory writes use current branch outputs only, not targets. Gate competition selects the largest normalized lag correlation. This predictability criterion is still hand designed and aligned with the task, but branch-to-regime roles are not supplied.

Compare equal averaging, full-eight-step stateless window competition, and persistent predictive-memory competition. All learn prediction weights via identical supervised losses, one Adam lr.01 throughout, update everyeight observations. Hard selection restricts direct loss gradients to selected branch parameters; Adam momentum can continue moving temporarily inactive parameters. The averaging control's winner usage is not an actual sparse route and its diagnostic overlap should not be interpreted as one.

Train A4,096 thenB4,096, with no boundary flag/reset; include singleA4,096, singleB4,096, and joint repeatedA/B4,096. Frozen evaluation streams revisit A/B/A/B and reverseB/A/B/A,256steps pervisit, with no fast-state reset at transitions. Erase fast state each step as an intervention. Report late error after32warmup steps separately from early error. The longer memory has more information than a stateless eight-step window; that is the intended state advantage, not matched observation horizons.

Use initial seeds0–2 and independent confirmation seeds3–7 without changing constants. Practical preset criteria: mean old/new MSE<0.30, ≥20% improvement over the full-window gate, erase worsens, and replication. Report per-seed failures rather than interpreting favorable means as universal success. No test tuning.

Observe branch-selection overlap across contexts and retained old performance. Low overlap alone is insufficient. This is a small linear-branch, hard-competition model; it does not show unique advantages of biological dendrites over an equivalent mixture of predictors or neuron-state architecture. A matched-state ordinary-neuron control is still necessary before a dendrite-specific claim. No novelty or paper-ready claim is made.

Tests confirm initial input weights mix both channels, state reads/writes are causal and detached, and direct prediction gradients are confined to selected branch rows. Results: `results/learned_predictive_branches_20261011` and `results/learned_predictive_branches_confirm_20261011`.

## Confirmed outcome

Eight-seed mean old/new MSE0.296/0.196 for predictive memory versus0.592/0.223 for the full-window gate. Mean functional route overlap~0.3%. Erasure raises mean error to1.142. Reverse evaluation supports the average-error benefit. Two old-task seed failures remain above0.30, so uniform reliability is not established. This is a positive synthetic mechanism result with random sensory mixtures; matched non-dendritic state controls and broader tasks remain necessary.
