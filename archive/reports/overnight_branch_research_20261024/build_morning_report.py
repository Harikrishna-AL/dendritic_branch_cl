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
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RAW=ROOT/'results/overnight_branch_research_20261024'
def load(name):
 p=RAW/name;assert json.loads((p/'completion.json').read_text())['complete'];return json.loads((p/'results.json').read_text())
mnist=load('pmi_confirmation');fashion=load('pmi_fashion_transfer')
keys=[('original_local',None),('no_message',16),('no_message',64),('learned_pmi',64),('global_ridge',None),('direct_ridge',None)]
labels=['Original local average','Four cues, 16D','Four cues, 64D','Four cues + local messages','Global ridge routing','Direct input ridge classifier']
def entries(rows,key):return [dict(next(x for x in run['methods'] if (x['method'],x.get('dimension'))==key),seed=run['seed'],order=run['order']) for run in rows]
def seedvalues(rows,key):return [100*st.mean(x['accuracy'] for x in entries(rows,key) if x['seed']==seed) for seed in sorted({x['seed'] for x in rows})]
def avg(rows,key):return st.mean(seedvalues(rows,key))
def difference(rows,a,b):
 v=[x-y for x,y in zip(seedvalues(rows,a),seedvalues(rows,b))];mean=st.mean(v);half=2.776445105*st.stdev(v)/math.sqrt(5);return f'{mean:+.2f} pp (95% paired-seed t interval {mean-half:+.2f} to {mean+half:+.2f})'
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for rows,ax,title in [(mnist,axes[0],'Split MNIST'),(fashion,axes[1],'Split Fashion MNIST: locked transfer')]:
 for i,key in enumerate(keys):
  val=avg(rows,key);ax.bar(i,val,color=['#a4b5c4','#6ca7a1','#40877f','#237e83','#ab865f','#7a6a9b'][i]);ax.scatter([i]*5,seedvalues(rows,key),s=10,color='#152634',zorder=3);ax.text(i,val+1.3,f'{val:.1f}',ha='center',fontsize=8)
 ax.set_xticks(range(6),['Original\nlocal','Four cues\n16D','Four cues\n64D','Four cues\n+ messages','Global ridge\nrouting','Direct ridge\nclassifier']);ax.tick_params(axis='x',labelsize=8);ax.set_ylim(0,100);ax.set_ylabel('All-ten-class automatic accuracy (%)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.savefig(OUT/'overnight_summary.png',dpi=180);fig.savefig(OUT/'overnight_summary.svg');plt.close(fig)
lines=['# Overnight research report — 24 October 2026','',
'**There are two bounded positive findings, not a breakthrough:** multiple input cues per branch improve automatic retrieval at approximately fixed auxiliary-memory budget, and a learned local compatibility rule gives a small additional MNIST gain. The communication gain does not transfer to Fashion MNIST. A simpler direct input-memory classifier remains better on both datasets.','',
'## Automatic task-ID-free results','', '| Method | MNIST | Fashion MNIST | Extra associative state | Total inference numeric state |','|---|---:|---:|---:|---:|']
for key,label in zip(keys,labels):
 e=entries(mnist,key)[0]['extra_bytes'];total=e+(0 if key[0]=='direct_ridge' else 315440);lines.append(f'| {label} | {avg(mnist,key):.2f}% | {avg(fashion,key):.2f}% | {e:,} B | {total:,} B |')
lines+=['',
'The table uses matched controls within each dataset cohort: five fresh seeds and both task orders. Fashion uses the MNIST settings unchanged, with no Fashion-specific validation or retuning. All ten labels compete at prediction time. Current training labels supervise the neural network and the direct classifier; the routing/communication memories receive current image features and branch activity only. Task IDs, task-boundary notifications, old-task replay, external configuration banks and test-label feedback are absent from automatic predictions.','',
'![Overnight summary](overnight_summary.png)','',
'## Finding 1: multiple cues help local memory','',
'A separate locked confirmation (seeds562–566, both orders) compared one64Dinput average per branch against four16Dcues at nearly the same storage cost. Accuracy improved **59.69% → 66.29%**; the paired difference was **+6.60pp**, 95%t interval **+2.22 to +10.98pp**. Extra memory increased from26,000to27,200bytes (4.6%). The method trades feature resolution for multimodal capacity; it is not an exact-byte equality or proof that slot count alone caused the gain.','',
'Four full64Dcues reached75.95% in that cohort, at104,000bytes of additional memory. These point estimates belong to a different cohort from the table above and should not be subtracted as paired effects. [Full capacity report](MULTICUE_REPORT.md).','',
'## Finding 2: learned neighbor compatibility helps modestly on MNIST','',
'For the64Dfour-cue model, learned compatibility improved **76.39% → 77.48%** in a new five-seed confirmation. The learned tables encode which branches tend to be active together along40edges of a fixed5×5neuron lattice. Three message rounds combine current input evidence with these learned pair preferences.','',
'- Learned versus no messages: **'+difference(mnist,('learned_pmi',64),('no_message',64))+'**.','- Learned versus shuffled edge tables: **'+difference(mnist,('learned_pmi',64),('shuffled_pmi',64))+'**.','- Fashion transfer, learned versus no messages: **'+difference(fashion,('learned_pmi',64),('no_message',64))+'**.','',
'The shuffled control shows that arbitrary coupling is insufficient in this MNIST condition. However, messages hurt16Dmemory and provide no detectable benefit on Fashion. This is a restricted mechanism result, not a general communication advantage. No matched ordinary-network implementation of the same rules was tested, so uniquely dendritic credit remains unestablished. [Communication report](PMI_REPORT.md).','',
'## Fashion transfer is an important limitation','',
'All settings, including optimizer, prototype decays, message strength and rounds, transferred unchanged from MNIST. Fresh seeds610–614 and both orders used the full60k/10k split. Local four-cue retrieval improves on the original local average, but remains below global ridge routing and the direct classifier. Historical-route oracle recovery averages92.92%, whereas automatic four-cue retrieval is58.18%. Thus useful skills remain accessible under task-assisted configurations, but the current cue mechanism still fails to retrieve many of them. The oracle uses true-pair selection of phase-end snapshots and must never be presented as automatic class-incremental performance.','',
'Fashion no-message and learned-message all-condition records, including the16Dnegative control, are in [the transfer report](FASHION_TRANSFER.md). No attempt was made to rescue the transfer result by retuning Fashion settings.','',
'## Memory and compute are real costs','',
'The one-layer network has78,760trainable parameters and400bytes of fading usefulness state: 315,440bytes total. Four full-dimensional cue prototypes add104,000bytes; communication adds another2,560bytes of edge counts. The direct ridge classifier needs18,944bytes and no neural backbone at deployment. These are explicit aggregate input/statistical memories, not a system that stores nothing extra. No individual examples are replayed, but input information is retained in prototypes.','',
'On two-thread CPU, within the same fresh profile the64Dfour-cue model took0.686ms per batch128; adding messages took1.857ms (about2.7×). Single-image latency increased0.105→0.447ms. The earlier memory-capacity profile also found extra write/inference cost for multicomponent cues. These microbenchmarks do not establish GPU scalability or full-training speed. State counts exclude transient workspace, optimizer state and Python overhead.','',
'## Negative and diagnostic work retained','',
'- Restricted patch cues and per-image usefulness teaching did not improve the initial validation pilot over whole-image activity associations.','- A bounded local delta-rule study failed badly: best validation29.98% versus61.16%for the average-based control; all40candidates and the unstable setting were retained.','- Centered prototype metrics produced at most sub-percentage-point gains in the single-cue condition and hurt or inconsistently changed multicue retrieval. No further confirmation was pursued.','- Local persistence traces for gating writes produced small, seed-dependent gains and did not consistently help low-memory retrieval. No threshold search was expanded.','- TRAIN-holdout analysis found that projecting predictions onto an external historical-route bank recovered only a few points. Those assisted diagnostics were never used as autonomous results.','- The local communication pilot retained all gamma and shuffled-control candidates. A small coupling was locked before confirmation; no test-driven adjustment was made.','',
'## What can be claimed','',
'With the trained branched backbone held to a fixed learning protocol, the design of additional local cue memory changes how effectively earlier skills are retrieved without task IDs. Multiple cues give a reproducible local-memory improvement, and learned local compatibility gives a small MNIST-specific gain beyond the same no-message model.','',
'What cannot be claimed: a breakthrough, top-conference readiness, superiority to simple memory-based classifiers, uniquely dendritic causation, broad transfer of the communication effect, or a full biologically local learning rule. The neural network still trains by backpropagation; the associations and messages are engineered mechanisms.','',
'## Recommended next research decision','',
'Prioritize stable, informative cue representations and the quality of online associations, rather than adding more message rounds. The large Fashion gap between oracle and automatic recovery makes this a sharper target than further coupling sweeps. If pretraining is explored, specify its dataset/label access and give the same encoder to ordinary-memory controls. Keep the direct classifier baseline, include complete memory/compute costs and confirm on held-out seeds/datasets before making a general claim.','',
'## Evidence and reproducibility','',
'- [Capacity study](MULTICUE_REPORT.md), [communication study](PMI_REPORT.md), [Fashion transfer](FASHION_TRANSFER.md).','- [Research status and decision log](../../results/overnight_branch_research_20261024/STATUS.md).','- [Raw capacity confirmation](../../results/overnight_branch_research_20261024/multicue_confirmation), [raw communication confirmation](../../results/overnight_branch_research_20261024/pmi_confirmation), [raw locked Fashion transfer](../../results/overnight_branch_research_20261024/pmi_fashion_transfer).','- All development candidates, negative pilots, protocols, sources, models and association states are preserved under results/overnight_branch_research_20261024.']
lines+=['','## Final verification','','A saved-state audit verified all30confirmation backbones and300automatic readouts, seed/order completeness, immutable evaluation state, 120,000current presentations and memory counts. Independent NumPy forwards reproduced a saved MNIST and Fashion no-message result exactly, with100%agreement with Torch predictions. Image-key extraction was shared; no training or settings changed. Protocol/source/result/data file hashes are preserved in [audit.json](../../results/overnight_branch_research_20261024/audit.json).']
(OUT/'MORNING_REPORT.md').write_text(pretty('\n'.join(lines)))
fashionlines=['# Locked transfer to Split Fashion MNIST','', 'Five fresh seeds610–614, both orders, no Fashion-specific tuning. All settings are transferred from MNIST. The backbone learns sequentially from current labels; association memories receive only current inputs/activity. All automatic outputs compete across ten classes.','', '| Method | Dimension | Accuracy | Seed SD | Extra memory |','|---|---:|---:|---:|---:|']
allkeys=[(m,d) for d in [16,64] for m in ['no_message','learned_pmi','shuffled_pmi']]+[(m,None) for m in ['original_local','global_ridge','direct_ridge']]
for key in allkeys:fashionlines.append(f"| {key[0]} | {key[1] or '—'} | {avg(fashion,key):.2f}% | {st.stdev(seedvalues(fashion,key)):.2f} pp | {entries(fashion,key)[0]['extra_bytes']:,} B |")
fashionlines+=['','Learned messages minus no-message,64D: '+difference(fashion,('learned_pmi',64),('no_message',64))+'.','',
'No evidence of transfer of the MNIST communication gain. The16Dcommunication condition also deteriorates. A source-trained fixed protocol may be suboptimal on Fashion; this study deliberately does not tune it and does not establish an intrinsic impossibility. No external route bank, task-ID or old-label revisits enter automatic predictions. The separately computed historical oracle is task-assisted.','',
'[Locked protocol](../../results/overnight_branch_research_20261024/pmi_fashion_transfer/protocol.json), [all90automatic readouts](../../results/overnight_branch_research_20261024/pmi_fashion_transfer/results.json), [saved states](../../results/overnight_branch_research_20261024/pmi_fashion_transfer/memories), [source driver](../../bdh_cl/scripts/overnight_pmi_confirmation.py).']
fashionlines+=['','## Class-order detail','','| Method | Forward order | Reverse order |','|---|---:|---:|']
for key in [('no_message',64),('learned_pmi',64),('global_ridge',None),('direct_ridge',None)]:
 v={order:100*st.mean(x['accuracy'] for x in entries(fashion,key) if x['order']==order) for order in ['forward','reverse']};fashionlines.append(f"| {key[0]} | {v['forward']:.2f}% | {v['reverse']:.2f}% |")
fashionlines+=['','Fashion messages hurt forward order and help reverse order; averaging orders does not establish a robust communication benefit. Both orders were fixed before confirmation.']
(OUT/'FASHION_TRANSFER.md').write_text(pretty('\n'.join(fashionlines)));print('Morning and Fashion reports built')
