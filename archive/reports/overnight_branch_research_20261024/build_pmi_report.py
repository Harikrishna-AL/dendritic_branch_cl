import json,statistics as st,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
def pretty(text):
 out=[]
 pattern=r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])|(?<=%)(?=[A-Za-z])|(?<=\dD)(?=[a-z])'
 for line in text.splitlines():
  parts=re.split(r'(\[[^\]]*\]\([^)]+\)|`[^`]*`)',line)
  line=''.join(part if i%2 else re.sub(pattern,' ',part) for i,part in enumerate(parts))
  out.append(line)
 return '\n'.join(out)+'\n'
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RAW=ROOT/'results/overnight_branch_research_20261024/pmi_confirmation';runs=json.loads((RAW/'results.json').read_text());assert json.loads((RAW/'completion.json').read_text())['complete'] and len(runs)==10;profiles=json.loads((RAW/'profile.json').read_text())
keys=[(m,d) for d in [16,64] for m in ['no_message','learned_pmi','shuffled_pmi']]+[(m,None) for m in ['original_local','global_ridge','direct_ridge']]
def entries(key):return [dict(next(r for r in run['methods'] if (r['method'],r.get('dimension'))==key),seed=run['seed']) for run in runs]
def values(key,field='accuracy'):return [100*st.mean(r[field] for r in entries(key) if r['seed']==seed) for seed in range(602,607)]
def avg(key,field='accuracy'):return st.mean(values(key,field))
def delta(a,b):
 v=[x-y for x,y in zip(values(a),values(b))];m=st.mean(v);h=2.776445105*st.stdev(v)/math.sqrt(5);return f'{m:+.2f} pp (95% paired-seed t interval {m-h:+.2f} to {m+h:+.2f})'
fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for d,ax in zip([16,64],axes):
 for i,m in enumerate(['no_message','learned_pmi','shuffled_pmi']):
  v=values((m,d));ax.bar(i,st.mean(v),color=['#a4b5c4','#237e83','#b78369'][i]);ax.scatter([i]*5,v,color='#152634',s=12,zorder=3);ax.text(i,st.mean(v)+1,f'{st.mean(v):.2f}%',ha='center')
 ax.set_xticks(range(3),['No messages','Learned PMI','Shuffled PMI']);ax.set_ylim(0,100);ax.set_ylabel('All-ten-class accuracy (%)');ax.set_title(f'{d}D cues, four prototypes per branch');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(OUT/'pmi_confirmation.png',dpi=180);fig.savefig(OUT/'pmi_confirmation.svg');plt.close(fig)
lines=['# Overnight result: learned local neighbor compatibility','',
'Five fresh seeds and both Split MNIST orders confirm a **small, configuration-dependent communication gain**. Full-dimensional four-cue retrieval improves from76.39% to77.48% with learned neighbor compatibility. Shuffling the learned edge potentials gives75.22%. The same rule hurts the lower-dimensional memory condition. A direct input-memory classifier remains stronger at85.61%.','',
'## All locked conditions','', '| Method | Cue dimension | Automatic accuracy | Seed SD | Earlier four pairs | Newest pair | Extra memory |','|---|---:|---:|---:|---:|---:|---:|']
for key in keys:
 z=entries(key)[0];lines.append(f"| {key[0]} | {key[1] or '—'} | {avg(key):.2f}% | {st.stdev(values(key)):.2f} pp | {avg(key,'old_accuracy'):.2f}% | {avg(key,'new_accuracy'):.2f}% | {z['extra_bytes']:,} B |")
lines+=['',
'Primary full-dimensional contrasts:','', '- Learned minus no-message: **'+delta(('learned_pmi',64),('no_message',64))+'**.','- Learned minus shuffled: **'+delta(('learned_pmi',64),('shuffled_pmi',64))+'**.','- Learned minus global ridge routing: '+delta(('learned_pmi',64),('global_ridge',None))+'.','',
'Both class orders are averaged within seed. Intervals use five independent seed means and a t distribution (four degrees of freedom); no multiple-comparison adjustment is applied. These are small-sample, single-dataset effects, not universal guarantees. The direct classifier uses fixed image features and sufficient statistics, so its zero seed SD is expected.','',
'![Communication confirmation](pmi_confirmation.png)','',
'## How the local communication works','',
'Place the25neurons on a fixed5×5lattice, with40neighbor edges. During the original training stream, each edge counts which pair of branches is simultaneously selected. It sees current activity only: no task IDs, class labels, boundaries or future saved configurations. At inference, local input-prototype matches supply branch preferences. Three damped message rounds encourage combinations supported by edge co-occurrence beyond what their individual frequencies predict (pointwise mutual information). All ten class outputs remain eligible.','',
'Unit Laplace smoothing, unary temperature0.05, three rounds, damping0.5 and coupling0.01 are fixed. The shuffled control permutes edge potentials with a fixed seed, preserving the pool of potential values but breaking their correspondence to particular neighboring neurons. It is a control for this learned compatibility structure, not a proof that only dendrites can implement it. The lattice is an engineering choice, not a measured biological circuit.','',
'## Memory and computation','',
'Forty4×4FP32co-occurrence tables add2,560bytes. Four64Dprototypes per branch already require104,000bytes; communication therefore needs106,560bytes of auxiliary state, plus315,440bytes of neural model state (422,000bytes total). Global ridge routing uses41,984extra bytes plus the same model; the direct classifier uses18,944bytes and no neural model at deployment. Counts and prototypes are explicit memory. No individual examples are replayed, and no external historical configuration bank enters automatic predictions.','',
'| Method | Cue dimension | Batch1 inference | Batch128 inference |','|---|---:|---:|---:|']
for p in profiles['rows']:lines.append(f"| {p['method']} | {p['dimension']} | {p['inference_ms']['1']:.3f} ms | {p['inference_ms']['128']:.3f} ms |")
lines+=['',profiles['scope'],'',
'The additional communication is not free. Its numerical benefit should be assessed against this latency and memory overhead. No claim of GPU efficiency or lower training cost follows.','',
'## Protocol and limits','',
'- Development600/601 used stratified50kfit/10kheld-out training images. Coupling candidates0/.01/.05/.2 were retained with learned and shuffled potentials for both16D and64Dfour-cue memories.','- Coupling0.01 was locked before confirmation: it was the smallest positive value within0.1percentage points of the best mean validation score. The0.05 candidate was only0.035points higher in mean and less consistent across development seeds. This parsimony decision was made after development, not pre-registered before the pilot, and never used official test outcomes.','- Confirmation602–606, forward and reverse orders, full60ktraining/full10ktest, two current epochs per pair, batch128, Adam0.003, clip5. Prototype decays16D0.995/64D1.0 were fixed from earlier development. All90automatic conditions are retained. No subsequent coupling adjustment or seed search.','- The neural network remains supervised with current labels. The association and communication memories receive current keys/branch activity only. The model is task-ID-free at inference, not unsupervised.','- Model/memory checksums verify no evaluation writes. Gamma0matches the unary route; uniform co-occurrence produces zero PMI effect. Each learned edge records exactly120,000current-image presentations. Historical snapshots are used only for a separately labeled oracle diagnostic after predictions.','- The16Dnegative control deteriorates with communication. No global superiority to input-memory classifiers or unique dendritic mechanism is established. The memory gain in MULTICUE_REPORT.md and this communication effect are separate cohorts; do not subtract their headline numbers as a paired effect.','',
'## Reproducibility','',
'- [Confirmation driver](../../bdh_cl/scripts/overnight_pmi_confirmation.py), [pilot](../../bdh_cl/scripts/overnight_pmi_message_pilot.py), [profile driver](../../bdh_cl/scripts/overnight_pmi_profile.py).','- [Locked protocol](../../results/overnight_branch_research_20261024/pmi_confirmation/protocol.json), [all results](../../results/overnight_branch_research_20261024/pmi_confirmation/results.json), [checks](../../results/overnight_branch_research_20261024/pmi_confirmation/verification.json), [CPU profiles](../../results/overnight_branch_research_20261024/pmi_confirmation/profile.json), [saved association states](../../results/overnight_branch_research_20261024/pmi_confirmation/memories).','- [Development candidates](../../results/overnight_branch_research_20261024/pmi_message_pilot/results.json), [earlier capacity report](MULTICUE_REPORT.md).']
(OUT/'PMI_REPORT.md').write_text(pretty('\n'.join(lines)));print('PMI report complete')
