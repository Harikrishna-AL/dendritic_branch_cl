import json,statistics as st,re
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
rows=json.loads((root/'results/local_graph_resource_benchmark_20261013/results.json').read_text())
group=json.loads((root/'results/local_graph_grouping_benchmark_20261013/results.json').read_text())
ref=json.loads((root/'results/local_graph_grouping_reference_20261013/results.json').read_text())
profile=json.loads((root/'results/local_graph_decoder_profile_20261013/results.json').read_text())
lines=['# Local branch memory: accuracy, retention, storage and compute','',
'The experiments support response-derived context retrieval through learned local coactivity. They do not establish an overall accuracy, compute or storage advantage over an independently tuned ordinary MLP or compact central activity memory. Increasing branches per neuron improves the eight-domain result modestly, at almost unchanged persistent storage. All unsuccessful comparisons are retained below.','',
'## What was tested','',
'MNIST and Fashion MNIST were each presented as four or eight consecutive domains: ordinary images followed by fixed random pixel rearrangements. Every domain contains the same ten classes. The learner receives inputs and supervised labels during training, with no domain IDs or boundary notifications. At inference it receives inputs only. Each current domain receives two training passes; earlier domain images are never replayed. As in standard permuted-MNIST, the same underlying source images appear transformed in later domains. This is domain-incremental learning, not class-incremental learning or a language-model experiment.','',
'The primary comparison has three fresh seeds per dataset, paired across models and domain counts. The grouping follow-up has three additional seeds per dataset, paired across its models. Validation uses a stratified 50,000/10,000 training split, excluding all held-out images from every domain. A fixed 2,000-example validation subset selects the recorded candidates separately for each dataset/domain count. Confirmation trains on all 60,000 training images. Final accuracy uses all 10,000 official test images per domain. No test-based choice of hyperparameters is made.','',
'## Branch mechanism','',
'The original model has64neurons with4branches each. Each branch computes tanh of its own image-weighted input. Fixed balanced64-dimensional class codes supply the desired response at each neuron during supervised learning. One branch per neuron is selected using its previous-minibatch response error. Only selected input synapses are updated, using presynaptic activity multiplied by local desired-minus-actual response and the tanh derivative. Successful usage reduces subsequent plasticity. Neighboring successfully responding branches increment a local4x4coactivity table.','',
'Inference considers every possible output class, selecting branch combinations that both match its fixed code and have compatible learned neighbor connections. Exact chain max-sum computes that score. No true class, domain ID, stored image, input prototype or archived task configuration is supplied at inference. Coactivity counts ARE explicit distributed memory state. The previous-minibatch error is short-term training state; inference uses longer-term weights and coactivity. This does not demonstrate a short-term-memory benefit during inference, an anatomically realistic dendrite-to-dendrite connection, or biological implementation of the decoder.','',
'## Controls and storage accounting','',
'- Ordinary MLP256:784→256→10, supervised cross-entropy, no-momentum SGD; separately validated ReLU/tanh and learning rates. Stored weights/biases814,120bytes versus812,480branch bytes, a0.20% difference.','- Ordinary MLP85:784→85→10, Adam; independently validated learning rate. Persistent weights plus Adam moments/steps811,036bytes; deployed weights alone270,340bytes.','- Central activity memory:256locally taught units,64active, continually learned activity prototypes with independently validated novelty threshold and4/8prototype capacity. No input archive or supplied domain ID. State820,240bytes for4prototypes or824,352for8, within1.5% of the original branch model.','- Local class-count classifier:31,400bytes of Bernoulli class-associated counts, no covariance or replay. This is deliberately a strong inexpensive control, not a matched-budget claim.','',
'The branch812,480bytes comprise803,840input-weight bytes,2,560fixed-code bytes,1,024recent-error bytes,1,024usage bytes and4,032neighbor-coactivity bytes. The additional recent-error/usage/coactivity state totals6,080bytes. Storage figures include persistent numeric parameters, statistics and optimizer state where present; they exclude gradients, activations, temporary working arrays, Python overhead and the common dataset. They are not peak hardware RAM measurements. For MLPs, a fully allocated gradient buffer would add another parameter-sized allocation during learning.','',
'## Primary results','',
'Final accuracy is averaged across all domains and three seeds. Forgetting is the best previously measured accuracy minus final accuracy on each old domain, averaged across old domains, using a fixed stratified2,000-example test subset. Low forgetting must be interpreted alongside acquisition and final old-domain accuracy: learning less initially can produce low forgetting.','',
'| Dataset | Domains | Model | Final accuracy | Acquisition | Old-domain accuracy | Forgetting (pp) |','|---|---:|---|---:|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 for d in [4,8]:
  for arm in ['branch','central','sgd_mlp','adam_mlp','local_nb']:
   rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,d,arm)];m=lambda key:100*st.mean(r[key] for r in rs)
   lines.append(f'| {ds} | {d} | {arm} | {m("accuracy"):.2f}% | {m("acquisition"):.2f}% | {m("old_accuracy"):.2f}% | {m("forgetting"):.2f} |')
lines+=['',f'![Accuracy and retention]({out}/accuracy_and_retention.png)','',
'On MNIST the branch model forgets less than the MLP but still has lower final old-domain accuracy. On Fashion MNIST it also loses to the small local class-count control; low forgetting does not consistently favor branches over SGD. These results do not support a general anti-forgetting advantage.','',
'## Communication ablation with unchanged final data weights','',
'| Dataset | Domains | Communication enabled | Disabled | Paired gain (pp) |','|---|---:|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 for d in [4,8]:
  rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(ds,d,'branch')];a=st.mean(r['accuracy'] for r in rs);b=st.mean(r['no_communication_accuracy'] for r in rs)
  lines.append(f'| {ds} | {d} | {100*a:.2f}% | {100*b:.2f}% | {100*(a-b):+.2f} |')
lines+=['','This paired intervention shows useful information in learned branch compatibility. It does not isolate the role of communication during learning, because communication is used for inference and its counts are merely accumulated during learning. Earlier scrambling controls also damaged retrieval; see the previous controls archive.','',
'## More branches within the same input-weight budget','',
'The follow-up compares64neurons×4branches with32neurons×8branches, both256input-response units and803,840input-weight bytes. Each grouping independently selects rate.3/1, usage timescale10,000/40,000, and communication weight0/.1/1 on the same validation split. Eight branches were selected for both datasets. The K4 reference uses its own best recorded validation setting and is evaluated on exactly the same fresh streams as K8.','',
'| Dataset | Tuned four branches | Tuned eight branches | Paired gain (pp) | Central memory | MLP256 |','|---|---:|---:|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 a=[r for r in ref if r['dataset']==ds];b=[r for r in group if r['dataset']==ds and r['arm']=='grouped'];c=[r for r in group if r['dataset']==ds and r['arm']=='central'];m=[r for r in group if r['dataset']==ds and r['arm']=='sgd_mlp']
 v=lambda rs:100*st.mean(r['accuracy'] for r in rs)
 lines.append(f'| {ds} | {v(a):.2f}% | {v(b):.2f}% | {v(b)-v(a):+.2f} | {v(c):.2f}% | {v(m):.2f}% |')
lines+=['','K8 uses815,104bytes, only2,624bytes more than K4. This is a modest reproducible improvement on these three paired streams. It remains behind the MLP and does not consistently match central memory. Grouping also changes fixed-code length64→32, active branches64→32 and chain topology; it is not a pure isolation of branch count. Three seeds do not establish broad statistical reliability.','',
'## Measured compute','',
'Two CPU threads; training time excludes validation/testing and includes current-domain input indexing and learning. Inference latency is the median of15warmed repetitions for batch sizes1and128. This is equal current-data access, not equal compute, FLOPs, hardware RAM or a GPU benchmark.','',
'| Four-domain MNIST model | Learning time (s) | Batch1 latency (ms) | Batch128 latency (ms) |','|---|---:|---:|---:|']
for arm in ['branch','central','sgd_mlp','adam_mlp','local_nb']:
 rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==('mnist',4,arm)]
 lines.append(f'| {arm} | {st.mean(r["train_seconds"] for r in rs):.2f} | {st.mean(r["latency"]["1"]["median_ms"] for r in rs):.3f} | {st.mean(r["latency"]["128"]["median_ms"] for r in rs):.3f} |')
lines += ['',f'![Resources]({out}/resources.png)','',
'The current sequential decoder is the main inference implementation bottleneck. A mathematically equivalent tree reduction was checked on a representative previously trained K4 model:','',
'| Batch | Sequential decoder (ms) | Parallel tree decoder (ms) |','|---:|---:|---:|']
for r in profile['timing']:lines.append(f'| {r["batch"]} | {r["sequential_ms"]:.3f} | {r["tree_ms"]:.3f} |')
lines+=['','The tree reduces sequential operations and improves batch1latency, but uses more arithmetic/temporary storage and worsens batch128latency. This diagnostic uses random inputs of matching dimensions; it is not a new accuracy cohort or a claim that branches outperform MLP inference. Prediction agreement and score agreement within floating-point tolerance were verified.','',
'For Nneurons,Kbranches,Dinput dimensions and Coutput classes, input responses cost O(DNK), and chain decoding costs O(CNK²). Connection storage is O(NK²). This class-hypothesis decoder would require reconsideration for a large language vocabulary; the MNIST method cannot simply be assumed to scale efficiently to transformers.','',
'## What now stands out—and what does not','',
'The supported finding is that a small set of local coactivity connections contains information useful for selecting compatible branch responses, without an external input-to-task/configuration archive. The fresh MNIST4-domain ablation improves73.93%→82.04% at unchanged final input weights, and the neighbor tables occupy only4,032bytes. Eight branches also improve the eight-domain result over an independently tuned four-branch reference at almost equal stored size.','',
'The stronger claim fails at present: the branch system does not outperform a similarly sized ordinary MLP in final accuracy or measured CPU time. A compact central activity memory remains stronger, especially on Fashion MNIST. Matching a weak ordinary baseline or reporting only low forgetting would conceal these limitations.','',
'The next scientifically useful direction is to improve local feature acquisition while preserving the response-derived retrieval mechanism. The current local coded learner acquires lower classification accuracy than the cross-entropy MLP; the cause requires further ablations. A mechanism study should also cross short-term branch selection and neighbor communication, include protection-off controls, and use the same feedback resources in the strongest local flat controls. More domains alone have not made the result conference-ready.','',
'## Reproduction and verification','',
'- Primary: `bdh_cl/scripts/local_graph_resource_benchmark.py`, `results/local_graph_resource_benchmark_20261013/`.','- Grouping: `bdh_cl/scripts/local_graph_grouping_benchmark.py`, `results/local_graph_grouping_benchmark_20261013/`.','- Paired K4 reference: `bdh_cl/scripts/local_graph_grouping_reference.py`, `results/local_graph_grouping_reference_20261013/`.','- Decoder diagnostic: `bdh_cl/scripts/local_graph_decoder_profile.py`, `results/local_graph_decoder_profile_20261013/`.','- Previous communication scrambling/central controls: `results/graph_domain_controls_20261013/`.','',
'Each experiment archives its protocol, candidates, selection, per-seed results and driver source. Generalized K4 training tensors and predictions were checked to exactly reproduce the original implementation on a small supervised sequence. Chain recurrence matched exhaustive toy enumeration, and inference did not mutate the branch state. Baseline optimizer/storage counts were checked after a real update.','',
'Python: `/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python`; PyTorch2.8, cached real datasets, two CPU threads. No external submission or email was sent.','']
body='\n'.join(lines)
body=re.sub(r'(\d)([a-zA-Z])',r'\1 \2',body)
body=re.sub(r'([a-z])(?=\d)',r'\1 ',body)
body=body.replace('MLP256','MLP 256').replace('MLP85','MLP 85').replace('4 x 4','4 × 4')
for old,new in [('batch1','batch 1'),('batch128','batch 128'),('Nneurons,Kbranches,Dinput dimensions and Coutput classes','N neurons, K branches, D input dimensions and C output classes'),('PyTorch2.8','PyTorch 2.8')]:body=body.replace(old,new)
(out/'REPORT.md').write_text(body)
