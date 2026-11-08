# Local branch memory: accuracy, retention, storage and compute

The experiments support response-derived context retrieval through learned local coactivity. They do not establish an overall accuracy, compute or storage advantage over an independently tuned ordinary MLP or compact central activity memory. Increasing branches per neuron improves the eight-domain result modestly, at almost unchanged persistent storage. All unsuccessful comparisons are retained below.

## What was tested

MNIST and Fashion MNIST were each presented as four or eight consecutive domains: ordinary images followed by fixed random pixel rearrangements. Every domain contains the same ten classes. The learner receives inputs and supervised labels during training, with no domain IDs or boundary notifications. At inference it receives inputs only. Each current domain receives two training passes; earlier domain images are never replayed. As in standard permuted-MNIST, the same underlying source images appear transformed in later domains. This is domain-incremental learning, not class-incremental learning or a language-model experiment.

The primary comparison has three fresh seeds per dataset, paired across models and domain counts. The grouping follow-up has three additional seeds per dataset, paired across its models. Validation uses a stratified 50,000/10,000 training split, excluding all held-out images from every domain. A fixed 2,000-example validation subset selects the recorded candidates separately for each dataset/domain count. Confirmation trains on all 60,000 training images. Final accuracy uses all 10,000 official test images per domain. No test-based choice of hyperparameters is made.

## Branch mechanism

The original model has 64 neurons with 4 branches each. Each branch computes tanh of its own image-weighted input. Fixed balanced 64-dimensional class codes supply the desired response at each neuron during supervised learning. One branch per neuron is selected using its previous-minibatch response error. Only selected input synapses are updated, using presynaptic activity multiplied by local desired-minus-actual response and the tanh derivative. Successful usage reduces subsequent plasticity. Neighboring successfully responding branches increment a local 4 × 4 coactivity table.

Inference considers every possible output class, selecting branch combinations that both match its fixed code and have compatible learned neighbor connections. Exact chain max-sum computes that score. No true class, domain ID, stored image, input prototype or archived task configuration is supplied at inference. Coactivity counts ARE explicit distributed memory state. The previous-minibatch error is short-term training state; inference uses longer-term weights and coactivity. This does not demonstrate a short-term-memory benefit during inference, an anatomically realistic dendrite-to-dendrite connection, or biological implementation of the decoder.

## Controls and storage accounting

- Ordinary MLP 256:784→256→10, supervised cross-entropy, no-momentum SGD; separately validated ReLU/tanh and learning rates. Stored weights/biases 814,120 bytes versus 812,480 branch bytes, a 0.20% difference.
- Ordinary MLP 85:784→85→10, Adam; independently validated learning rate. Persistent weights plus Adam moments/steps 811,036 bytes; deployed weights alone 270,340 bytes.
- Central activity memory:256 locally taught units,64 active, continually learned activity prototypes with independently validated novelty threshold and 4/8 prototype capacity. No input archive or supplied domain ID. State 820,240 bytes for 4 prototypes or 824,352 for 8, within 1.5% of the original branch model.
- Local class-count classifier:31,400 bytes of Bernoulli class-associated counts, no covariance or replay. This is deliberately a strong inexpensive control, not a matched-budget claim.

The branch 812,480 bytes comprise 803,840 input-weight bytes,2,560 fixed-code bytes,1,024 recent-error bytes,1,024 usage bytes and 4,032 neighbor-coactivity bytes. The additional recent-error/usage/coactivity state totals 6,080 bytes. Storage figures include persistent numeric parameters, statistics and optimizer state where present; they exclude gradients, activations, temporary working arrays, Python overhead and the common dataset. They are not peak hardware RAM measurements. For MLPs, a fully allocated gradient buffer would add another parameter-sized allocation during learning.

## Primary results

Final accuracy is averaged across all domains and three seeds. Forgetting is the best previously measured accuracy minus final accuracy on each old domain, averaged across old domains, using a fixed stratified 2,000-example test subset. Low forgetting must be interpreted alongside acquisition and final old-domain accuracy: learning less initially can produce low forgetting.

| Dataset | Domains | Model | Final accuracy | Acquisition | Old-domain accuracy | Forgetting (pp) |
|---|---:|---|---:|---:|---:|---:|
| mnist | 4 | branch | 82.04% | 78.70% | 82.49% | 1.28 |
| mnist | 4 | central | 82.25% | 79.62% | 82.49% | 2.51 |
| mnist | 4 | sgd_mlp | 90.56% | 95.05% | 88.56% | 8.43 |
| mnist | 4 | adam_mlp | 87.35% | 93.65% | 84.70% | 11.86 |
| mnist | 4 | local_nb | 80.46% | 77.08% | 80.58% | 2.87 |
| mnist | 8 | branch | 75.48% | 73.61% | 76.28% | 3.58 |
| mnist | 8 | central | 79.01% | 77.07% | 79.16% | 3.28 |
| mnist | 8 | sgd_mlp | 83.10% | 95.27% | 81.17% | 15.43 |
| mnist | 8 | adam_mlp | 73.86% | 93.96% | 70.80% | 24.85 |
| mnist | 8 | local_nb | 75.11% | 74.56% | 75.04% | 5.55 |
| fashion_mnist | 4 | branch | 64.48% | 70.31% | 64.89% | 7.34 |
| fashion_mnist | 4 | central | 68.19% | 71.85% | 68.57% | 4.96 |
| fashion_mnist | 4 | sgd_mlp | 78.44% | 82.97% | 77.39% | 3.59 |
| fashion_mnist | 4 | adam_mlp | 75.39% | 85.64% | 72.66% | 11.50 |
| fashion_mnist | 4 | local_nb | 68.12% | 70.00% | 68.00% | 2.56 |
| fashion_mnist | 8 | branch | 58.33% | 63.09% | 59.35% | 5.47 |
| fashion_mnist | 8 | central | 64.67% | 67.22% | 65.38% | 3.26 |
| fashion_mnist | 8 | sgd_mlp | 68.97% | 82.93% | 67.19% | 14.97 |
| fashion_mnist | 8 | adam_mlp | 62.44% | 85.64% | 59.44% | 25.54 |
| fashion_mnist | 8 | local_nb | 64.95% | 68.02% | 64.96% | 3.06 |

![Accuracy and retention](/Users/harikrishna/Documents/lossfunk/graph_cl/q_graph_cl/bdh-d/reports/local_branch_resource_audit_20261013/accuracy_and_retention.png)

On MNIST the branch model forgets less than the MLP but still has lower final old-domain accuracy. On Fashion MNIST it also loses to the small local class-count control; low forgetting does not consistently favor branches over SGD. These results do not support a general anti-forgetting advantage.

## Communication ablation with unchanged final data weights

| Dataset | Domains | Communication enabled | Disabled | Paired gain (pp) |
|---|---:|---:|---:|---:|
| mnist | 4 | 82.04% | 73.93% | +8.11 |
| mnist | 8 | 75.48% | 70.28% | +5.20 |
| fashion_mnist | 4 | 64.48% | 63.15% | +1.34 |
| fashion_mnist | 8 | 58.33% | 55.85% | +2.49 |

This paired intervention shows useful information in learned branch compatibility. It does not isolate the role of communication during learning, because communication is used for inference and its counts are merely accumulated during learning. Earlier scrambling controls also damaged retrieval; see the previous controls archive.

## More branches within the same input-weight budget

The follow-up compares 64 neurons×4 branches with 32 neurons×8 branches, both 256 input-response units and 803,840 input-weight bytes. Each grouping independently selects rate.3/1, usage timescale 10,000/40,000, and communication weight 0/.1/1 on the same validation split. Eight branches were selected for both datasets. The K4 reference uses its own best recorded validation setting and is evaluated on exactly the same fresh streams as K8.

| Dataset | Tuned four branches | Tuned eight branches | Paired gain (pp) | Central memory | MLP 256 |
|---|---:|---:|---:|---:|---:|
| mnist | 76.08% | 78.07% | +1.98 | 79.05% | 82.89% |
| fashion_mnist | 59.20% | 61.88% | +2.67 | 68.36% | 71.44% |

K8 uses 815,104 bytes, only 2,624 bytes more than K4. This is a modest reproducible improvement on these three paired streams. It remains behind the MLP and does not consistently match central memory. Grouping also changes fixed-code length 64→32, active branches 64→32 and chain topology; it is not a pure isolation of branch count. Three seeds do not establish broad statistical reliability.

## Measured compute

Two CPU threads; training time excludes validation/testing and includes current-domain input indexing and learning. Inference latency is the median of 15 warmed repetitions for batch sizes 1 and 128. This is equal current-data access, not equal compute, FLOPs, hardware RAM or a GPU benchmark.

| Four-domain MNIST model | Learning time (s) | Batch 1 latency (ms) | Batch 128 latency (ms) |
|---|---:|---:|---:|
| branch | 1.84 | 1.040 | 4.114 |
| central | 1.91 | 0.036 | 0.258 |
| sgd_mlp | 1.17 | 0.011 | 0.057 |
| adam_mlp | 1.37 | 0.010 | 0.033 |
| local_nb | 0.83 | 0.061 | 0.129 |

![Resources](/Users/harikrishna/Documents/lossfunk/graph_cl/q_graph_cl/bdh-d/reports/local_branch_resource_audit_20261013/resources.png)

The current sequential decoder is the main inference implementation bottleneck. A mathematically equivalent tree reduction was checked on a representative previously trained K4 model:

| Batch | Sequential decoder (ms) | Parallel tree decoder (ms) |
|---:|---:|---:|
| 1 | 1.034 | 0.281 |
| 128 | 3.860 | 7.952 |

The tree reduces sequential operations and improves batch 1 latency, but uses more arithmetic/temporary storage and worsens batch 128 latency. This diagnostic uses random inputs of matching dimensions; it is not a new accuracy cohort or a claim that branches outperform MLP inference. Prediction agreement and score agreement within floating-point tolerance were verified.

For N neurons, K branches, D input dimensions and C output classes, input responses cost O(DNK), and chain decoding costs O(CNK²). Connection storage is O(NK²). This class-hypothesis decoder would require reconsideration for a large language vocabulary; the MNIST method cannot simply be assumed to scale efficiently to transformers.

## What now stands out—and what does not

The supported finding is that a small set of local coactivity connections contains information useful for selecting compatible branch responses, without an external input-to-task/configuration archive. The fresh MNIST4-domain ablation improves 73.93%→82.04% at unchanged final input weights, and the neighbor tables occupy only 4,032 bytes. Eight branches also improve the eight-domain result over an independently tuned four-branch reference at almost equal stored size.

The stronger claim fails at present: the branch system does not outperform a similarly sized ordinary MLP in final accuracy or measured CPU time. A compact central activity memory remains stronger, especially on Fashion MNIST. Matching a weak ordinary baseline or reporting only low forgetting would conceal these limitations.

The next scientifically useful direction is to improve local feature acquisition while preserving the response-derived retrieval mechanism. The current local coded learner acquires lower classification accuracy than the cross-entropy MLP; the cause requires further ablations. A mechanism study should also cross short-term branch selection and neighbor communication, include protection-off controls, and use the same feedback resources in the strongest local flat controls. More domains alone have not made the result conference-ready.

## Reproduction and verification

- Primary: `bdh_cl/scripts/local_graph_resource_benchmark.py`, `results/local_graph_resource_benchmark_20261013/`.
- Grouping: `bdh_cl/scripts/local_graph_grouping_benchmark.py`, `results/local_graph_grouping_benchmark_20261013/`.
- Paired K4 reference: `bdh_cl/scripts/local_graph_grouping_reference.py`, `results/local_graph_grouping_reference_20261013/`.
- Decoder diagnostic: `bdh_cl/scripts/local_graph_decoder_profile.py`, `results/local_graph_decoder_profile_20261013/`.
- Previous communication scrambling/central controls: `results/graph_domain_controls_20261013/`.

Each experiment archives its protocol, candidates, selection, per-seed results and driver source. Generalized K4 training tensors and predictions were checked to exactly reproduce the original implementation on a small supervised sequence. Chain recurrence matched exhaustive toy enumeration, and inference did not mutate the branch state. Baseline optimizer/storage counts were checked after a real update.

Python: `/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python`; PyTorch 2.8, cached real datasets, two CPU threads. No external submission or email was sent.
