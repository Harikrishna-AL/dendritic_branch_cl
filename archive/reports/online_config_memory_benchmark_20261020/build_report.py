import json,statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;base=ROOT/'results/online_config_memory_benchmark_20261020'
r=json.loads((base/'results.json').read_text());f=json.loads((base/'no_bank_followup/results.json').read_text());audit=json.loads((base/'key_drift_audit.json').read_text());selection=json.loads((base/'selection.json').read_text())
assert len(r)==len(f)==len(audit)==6
assert json.loads((base/'completion.json').read_text())['complete'] and json.loads((base/'no_bank_followup/completion.json').read_text())['complete']
assert json.loads((base/'audit_verification.json').read_text())['all_6_backbones_exactly_match_independent_original_training']
METHODS=['local','local_communication','central96','central144','rls'];KEYS=['fixed_pixels','frozen_responses','evolving_responses']
NAMES={'local':'Branch-local','local_communication':'Branch-local + communication','central96':'Central 96 centroids','central144':'Central 144 centroids','rls':'RLS activity associations'}
def vals(data,key,method,metric='accuracy'):
 return { (x['seed'],x['order']):next(z for z in x['methods'] if z['key']==key and z['method']==method)[metric] for x in data}
def grouped(v):return [st.mean(a for (s,o),a in v.items() if s==seed) for seed in sorted(set(s for s,o in v))]
def mean(data,key,method,metric='accuracy'):return 100*st.mean(grouped(vals(data,key,method,metric)))
def sd(data,key,method):return 100*st.stdev(grouped(vals(data,key,method)))
def diag(name):return 100*st.mean(x['diagnostics'][name]['accuracy'] for x in r)
direct=100*st.mean(x['direct_class_rls']['accuracy'] for x in f)
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
ax=axes[0]
for i,m in enumerate(METHODS):
 for offset,data,color,label in [(-.18,r,'#236e9e','With saved bank'),(.18,f,'#9ab8ca','No saved bank (follow-up)')]:
  ax.bar(i+offset,mean(data,'fixed_pixels',m),.34,color=color,label=label if i==0 else None)
  ax.scatter([i+offset]*3,[v*100 for v in grouped(vals(data,'fixed_pixels',m))],s=13,color='#152634',zorder=3)
ax.axhline(direct,color='#477d4e',linestyle='--',label='Direct class RLS');ax.set_xticks(range(5),['Local','Local +\nmessages','Central\n96','Central\n144','RLS']);ax.set_ylim(0,100);ax.set_ylabel('All-ten-digit test accuracy (%)');ax.set_title('Fixed pixel keys');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=7,frameon=False,loc='upper left')
ax=axes[1]
for i,k in enumerate(KEYS):
 for off,data,color,label in [(-.18,r,'#236e9e','With saved bank'),(.18,f,'#9ab8ca','No saved bank')]:ax.bar(i+off,mean(data,k,'rls'),.34,color=color,label=label if i==0 else None)
ax.axhline(diag('ordinary'),color='#b15745',linestyle=':',label='Final automatic route');ax.set_xticks(range(3),['Fixed\npixels','Frozen initial\nresponses','Changing\nresponses']);ax.set_ylim(0,100);ax.set_title('RLS: stable versus changing keys');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8,frameon=False)
fig.savefig(OUT/'online_retrieval.png',dpi=180);fig.savefig(OUT/'online_retrieval.svg');plt.close(fig)
lines=['# Online acquisition of configuration retrieval memory','',
'Completed on 20 October 2026. **Online current-input/current-branch associations recover old skills without replaying historical images. Stable keys work much better than changing network-response keys. However, branch-local memory does not beat the stronger RLS association baseline, communication adds no consistent gain, and a smaller direct RLS classifier performs best.**','',
'## Main results','',
'All results below use fixed 64-dimensional pixel keys, three fresh backbone seeds and two paired class orders. Every prediction competes across all ten digits.','',
'| Method | With externally saved configuration bank | No external bank — exploratory follow-up |','|---|---:|---:|']
for m in METHODS:lines.append(f'| {NAMES[m]} | {mean(r,"fixed_pixels",m):.2f}% | {mean(f,"fixed_pixels",m):.2f}% |')
lines+=['',f'The same backbones achieve {diag("ordinary"):.2f}% using their final automatic configuration and {diag("oracle"):.2f}% using the historical-route oracle. The direct fixed-key class RLS control reaches **{direct:.2f}%**, without needing the branch network. These are fresh paired follow-ups, not the original three-seed 98.02% cohort.', '',
'![Online retrieval, no-bank follow-up and key drift comparison](online_retrieval.png)','',
'Bars are seed means, with the two class orders averaged within each seed. Dots in the left panel are these three seed-level values. No significance claim follows from three seeds.','',
'## What was learned online','',
'The backbone remains the original hard-routed 24-neuron, four-branch network, trained for two passes per current digit pair with Adam 0.001, gradient clipping 5, factual cross-entropy plus 0.5 mean counterfactual cross-entropy, and error EMA 0.95. Current digit labels train the backbone as usual. The routing sidecars do not affect its gradient updates.','',
'Before each training update, a memory receives only the current input key and the currently active branch in each neuron. It receives no digit label, task ID, task boundary or future end-of-task configuration. Once a class leaves the stream, its images are never revisited for association learning. The original two within-current-task passes are retained; no historical replay is added.','',
'Five end-of-phase configurations are recorded externally by the harness. In the primary test they are provided only at inference; every memory must select from this same bank. This isolates retrieval, not autonomous configuration storage. Importantly, memory targets are the contemporaneous branches rather than the eventual saved route: future routes were not leaked backward into association training.','',
'## Memory mechanisms','',
'- **Branch-local:** each branch accumulates its coactive key sum and count. Cosine matches to these conditional means provide branch evidence. A ring of 24 neighboring neurons accumulates branch coactivity. The optional communication condition passes neighbor-belief messages for three iterations.','- **Central centroids:** a bounded bank of 96 or 144 online key centroids, each tagged with the full contemporaneous 24-branch pattern. Novelty threshold is 0.85. When full, a new route replaces the least-supported entry. These are explicit summary associations, not an individual-example reservoir; a newly initialized centroid can initially represent one observation.','- **RLS association baseline:** accumulate key covariance C and key-to-branch cross-statistics B from 96 one-hot branch-activity targets. Ridge strength is 1. The inferred map W = B C⁻¹ predicts branch preferences. This is an online least-squares/RLS associative baseline inspired by DeltaNet-PC’s memory rule, **not a reproduction of its predictive-coding architecture, learned addressing or settling procedure**.','',
'RLS uses global covariance statistics and is not a branch-local biological mechanism. Its statistics are updated causally; no old keys or images need to be retained. A transient linear solve at readout recovers the map. The local associative sums also persist across tasks: they are not newly demonstrated fading short-term memory. The backbone still contains the original recent-error state.','',
'## Key controls','',
'Every key has 63 normalized features plus a constant 0.2 coordinate, renormalized to dimension 64. Fixed pixel keys use deterministic 7×9 average pooling. Response keys use all 96 first-layer tanh responses, pooled to 63 coordinates. One condition holds the initial first-layer encoder fixed; the other uses the current changing encoder. Those two response encoders begin identically.','',
'The frozen-response control requires an extra copy of the initial encoder and is not equal-storage to the other key conditions. Pixel and response representations are different; their comparison alone cannot isolate drift. Comparing frozen versus evolving response keys, plus the measured movement below, supports drift as a substantial problem but does not prove it is the only source of failure.','',
'## Validation and fresh confirmation','',
'Development uses seed 510, forward order, a stratified 50,000/10,000 fitting/validation split from training data. Routers write associations only for the fitting stream; held-out images never write memory. Each key/method independently selects its allowed write stability (every batch or after ten consecutive identical active configurations), neighbor strength when applicable, and central bank-compatibility penalty. All validation candidates are archived. No official test labels select settings.','',
'Confirmation uses seeds 511/512/513 and forward/reverse pair orders, full 60,000 training images and the official 10,000-image test set. Orders are paired repeats, not six independent seeds. Retrieval memory and final backbone stay fixed during test inference. The optional ten-batch stability gate uses whole-route agreement and is an engineering aid, not strictly local communication.','',
'## Complete primary results: saved bank','',
'| Key | Method | Final accuracy ± seed SD | Old eight classes | Newest pair |','|---|---|---:|---:|---:|']
for k in KEYS:
 for m in METHODS:lines.append(f'| {k} | {NAMES[m]} | {mean(r,k,m):.2f}% ± {sd(r,k,m):.2f} pp | {mean(r,k,m,"old_accuracy"):.2f}% | {mean(r,k,m,"new_accuracy"):.2f}% |')
lines+=['',
'## Fresh oracle and order sensitivity','',
'| Seed | Order | Final automatic accuracy | Historical-route oracle |','|---|---|---:|---:|']
for x in r:lines.append(f'| {x["seed"]} | {x["order"]} | {100*x["diagnostics"]["ordinary"]["accuracy"]:.2f}% | {100*x["diagnostics"]["oracle"]["accuracy"]:.2f}% |')
lines+=['',
'The original 98.02% oracle was not universal: this fresh cohort ranges from 88.27% to 97.79%. This is additional evidence that retained accessibility depends on initialization and class order. The oracle is a diagnostic choice among recorded routes, not a strict upper bound over every possible branch configuration.','',
'## Key movement diagnostic','',
'Test class mean keys were measured immediately after each class was learned and again with the final encoder. These test means are reporting artifacts only; neither the original network nor associative memories receive them. The table averages squared mean-key movement over the old eight classes, then orders and seeds.','',
'| Key type | Old-class mean-key movement (MSE) |','|---|---:|']
for k in KEYS:
 v=st.mean(st.mean(z for c,z in a['classmean_key_drift'][k].items() if int(c) not in a['pairs'][-1]) for a in audit);lines.append(f'| {k} | {v:.8g} |')
lines+=['',
'Changing-response retrieval is close to the final-route baseline even with RLS, while both fixed key families retrieve substantially more. This is consistent with old associations becoming stale as their addresses change. A future control should hold the final feature quality and association targets fixed to quantify exactly how much loss is due to address drift.','',
'## Exploratory follow-up: remove the external bank','',
'After the primary result, the same cohort was rerun to obtain and save association states. Every repeated backbone exactly reproduced its primary final tensors. All primary validation-selected settings were kept fixed; the no-bank readout was **not independently tuned** and is not an independent confirmation.','',
'At inference, local memories choose a branch independently in each neuron (with or without messages). RLS takes each neuron’s largest predicted branch value. Central centroids return the observed route stored at the nearest centroid. The external five-route bank never enters these predictions; it is used only for historical-match analysis afterward. Central centroids still explicitly store online full-route patterns, whereas local and RLS states represent associations through per-branch quantities.','',
'| Key | Method | No-bank accuracy ± seed SD | Old eight classes | Newest pair |','|---|---|---:|---:|---:|']
for k in KEYS:
 for m in METHODS:lines.append(f'| {k} | {NAMES[m]} | {mean(f,k,m):.2f}% ± {sd(f,k,m):.2f} pp | {mean(f,k,m,"old_accuracy"):.2f}% | {mean(f,k,m,"new_accuracy"):.2f}% |')
lines+=['',
'With fixed pixel keys, factorized local memories outperform the whole-route centroid controls in this no-bank follow-up. This is a useful structural lead: composing branch choices can be more robust than retrieving a complete observed configuration. But primary settings were selected with the bank present, which can favor different stability settings; the comparison needs independent no-bank validation and fresh confirmation before a strong advantage claim. Larger central centroid capacity does not establish an adequately optimized central baseline.','',
'## Ordinary direct classifier control','',
'A direct class RLS model receives the identical fixed pixel keys and current digit labels, rather than branch-activity targets. It accumulates covariance plus ten class cross-statistic rows during the same current-only two-pass sequence. It reaches '+f'**{direct:.2f}%**'+ ' at 18,944 bytes of numeric persistent state, without requiring the branch network or a route bank.','',
'This stronger, smaller predictive baseline prevents claiming a general branch advantage. Its teacher is explicitly the digit label, whereas association memories see branch activity; it is a complete-method control, not a pure architecture intervention. Given fixed features and identical data, its final sufficient-statistics solution is essentially invariant to seed/order; the equal accuracies are not independent random-feature replications.','',
'## Persistent storage','',
'| Component | Numeric bytes |','|---|---:|','| Original final backbone and error buffer | 302,824 |','| External five-route bank, packed uint8 | 120 |','| Branch-local sums/counts/ring links | 26,496 |','| Central 96 centroids + observed routes/counts | 27,264 |','| Central 144 centroids + observed routes/counts | 40,896 |','| RLS association covariance/cross-statistics | 40,960 |','| Optional frozen initial key encoder | 301,440 |','| Direct fixed-key digit RLS (no backbone needed) | 18,944 |','',
'Local and central96 are close in memory (central96 is 2.9% larger). RLS has about 55% more associative state than local; central144 nearly matches its budget. Without a bank, the fixed-key branch+RLS pipeline totals 343,784 bytes versus 329,320 for branch-local, before scalar metadata. A stability-10 updater additionally needs approximately 32 bytes for its previous route and streak counter. The frozen-response key condition adds the initial encoder cost.','',
'These are per-method persistent numeric budgets. Experiment files contain multiple competing memories for paired measurement, not a claim that all are deployed together. Common Adam training state, scalar metadata and temporary gradients, solves, activations and key caches are excluded. No peak-RAM, runtime or equal-FLOP advantage is claimed.','',
'## Interpretation and next decision','',
'1. Online cue-to-branch associations can recover useful skills without old-image replay; the RLS no-bank follow-up also avoids an external configuration archive.','2. Key stability is a major practical constraint. Addresses based on a changing encoder are unreliable under these rules.','3. Local memory is competitive with similarly sized centroid memory under the bank and better in the exploratory no-bank readout, but RLS is stronger. Communication has no consistent positive effect.','4. A direct, much smaller classifier beats every retrieval pipeline, so task-ID-free retrieval alone is insufficient for a branch-specific scientific contribution.','',
'The next focused test should independently validate no-bank storage/readout rules, then compare a more discriminative local cue update with equivalent shared controls. Avoid adding further task archives or global statistics while claiming locality. A stronger visual encoder can later test whether branch-held nonlinear skills offer something the simple fixed-key classifier cannot, with every comparator using the same encoder and honest storage accounting.','',
'## Reproducibility and audits','',
'- [Primary driver](../../bdh_cl/scripts/online_config_memory_benchmark.py), [protocol](../../results/online_config_memory_benchmark_20261020/protocol.json), [all development candidates](../../results/online_config_memory_benchmark_20261020/development.json), [selection](../../results/online_config_memory_benchmark_20261020/selection.json), [90 primary readouts](../../results/online_config_memory_benchmark_20261020/results.json).','- [Independent backbone/key-drift audit](../../bdh_cl/scripts/online_config_memory_audit.py), [audit results](../../results/online_config_memory_benchmark_20261020/key_drift_audit.json), [verification](../../results/online_config_memory_benchmark_20261020/audit_verification.json).','- [No-bank driver and direct RLS control](../../bdh_cl/scripts/online_config_no_bank_followup.py), [follow-up protocol](../../results/online_config_memory_benchmark_20261020/no_bank_followup/protocol.json), [90 follow-up readouts and six direct controls](../../results/online_config_memory_benchmark_20261020/no_bank_followup/results.json), [saved association arrays](../../results/online_config_memory_benchmark_20261020/no_bank_followup/memories).','- Checked online ridge against an offline synthetic reference; all inference methods leave persistent association arrays unchanged on a synthetic read-only audit; all six backbones exactly match independent original training; frozen current model tensors do not change during evaluation; and all expected result counts and numeric budgets are complete.','- Six underlying confirmation backbones are paired across methods; independent audit and no-bank reconstruction runs repeat these same conditions and do not increase the independent sample count.','']
(OUT/'REPORT.md').write_text('\n'.join(lines));print('Verified90primary+90followupreadouts,sixdirectcontrols,andallindependentaudits.')
