# Matched training ablation

Same sparse 168x4 MLP, fixed training recipe, three seeds x two orders. All reported final inference is stateless soft-magnitude combination, isolating learned weights. No tuning on these query metrics.

| Training selection | Final image-only test accuracy | Forgetting pp |
|---|---:|---:|
| Supervised error trace (existing matched baseline) | 66.70 ± 1.01% | See original trajectory report |
| no_state | 20.20 ± 0.01% | 98.65 |
| label_free | 19.08 ± 0.11% | 93.76 |

No-state training selects each image’s greatest-magnitude branch per neuron, without history. Label-free training selects each neuron’s highest EMA pseudo-error preference: branch substitution CE against the model’s own current predicted class, standardized across four branches; EMA decay .9, updated only after each training minibatch. Labels still train the common CE+aux weights, but do not form that state. State persists across batches/tasks; no boundary resets. Batch-mean updates during training approximate stream context, unlike single-image updates in the frozen inference test. All alternatives retain the same auxiliary loss and optimizer.

This is a fixed-recipe ablation, not proof that every unsupervised-state training rule must fail. Extra context buffer is 2,688 bytes; original error buffer is unused in these alternatives. State is not used during final stateless inference. No-state versus label-free gradient flow differs through their chosen branches; that is the intervention.
