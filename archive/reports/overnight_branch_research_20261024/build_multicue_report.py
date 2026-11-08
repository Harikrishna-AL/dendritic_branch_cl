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
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RAW=ROOT/'results/overnight_branch_research_20261024/multicue_confirmation';runs=json.loads((RAW/'results.json').read_text());assert json.loads((RAW/'completion.json').read_text())['complete'] and len(runs)==10
profiles=json.loads((RAW/'profile.json').read_text())['rows'];settings=json.loads((RAW/'selection.json').read_text());shapes=[(16,1),(16,4),(32,1),(32,2),(64,1),(64,2),(64,4)];keys=[('original_local',None,None)]+[('multicue',d,s) for d,s in shapes]+[('global_ridge',None,None)]+[('direct_ridge',d,None) for d in [16,32,64]]
def matches(r,key):return (r['method'],r.get('dimension'),r.get('slots'))==key
def entries(key):return [dict(next(r for r in run['methods'] if matches(r,key)),seed=run['seed'],order=run['order']) for run in runs]
def seedvalues(key,field='accuracy'):return [100*st.mean(r[field] for r in entries(key) if r['seed']==seed) for seed in range(562,567)]
def mean(key,field='accuracy'):return st.mean(seedvalues(key,field))
def label(key):
 m,d,s=key
 return f'Local {d}D × {s} cue'+('s' if s!=1 else '') if m=='multicue' else f'Direct ridge {d}D' if m=='direct_ridge' else 'Original local average' if m=='original_local' else 'Global ridge routing'
def difference(a,b):
 v=[x-y for x,y in zip(seedvalues(a),seedvalues(b))];m=st.mean(v);half=2.776445105*st.stdev(v)/math.sqrt(5);return f'{m:+.2f} pp (95% paired-seed t interval {m-half:+.2f} to {m+half:+.2f})'
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for key in keys:
 z=entries(key)[0];mem=z['extra_bytes']/1000;v=mean(key);color='#237e83' if key[0]=='multicue' else '#7a6a9b' if key[0]=='direct_ridge' else '#a66e4d';ax.scatter(mem,v,s=45,color=color);ax.annotate(label(key).replace('Local ','').replace('Original local average','Original local'),(mem,v),xytext=(5,-12) if key[0]=='original_local' else (5,4),textcoords='offset points',fontsize=7)
ax.set_xscale('log');ax.set_xlabel('Additional associative memory (KB, logarithmic scale)');ax.set_ylabel('All-ten-class automatic accuracy (%)');ax.set_title('Confirmed local cue capacity versus memory');ax.set_ylim(35,92);ax.set_xlim(1.4,175);ax.grid(alpha=.2);fig.savefig(OUT/'multicue_capacity.png',dpi=180);fig.savefig(OUT/'multicue_capacity.svg');plt.close(fig)
lines=['# Overnight result: multimodal branch-local retrieval','',
'Confirmed on five fresh seeds and both Split MNIST orders, 24 October 2026. **Four lower-dimensional input cues per branch improve automatic accuracy from 59.69% to 66.29% with nearly the same associative-memory budget. Four full-dimensional cues reach 75.95%, comparable in mean to global ridge routing at 75.90%, but require more auxiliary memory. The direct input classifier remains stronger and smaller.**','',
'## All locked confirmation conditions','', '| Method | Automatic accuracy | Seed SD | Earlier four pairs | Newest pair | Extra memory | Total inference numeric state |','|---|---:|---:|---:|---:|---:|---:|']
backbone=runs[0]['diagnostics']['model_bytes']
for key in keys:
 z=entries(key)[0];extra=z['extra_bytes'];total=extra+(0 if key[0]=='direct_ridge' else backbone);lines.append(f"| {label(key)} | {mean(key):.2f}% | {st.stdev(seedvalues(key)):.2f} pp | {mean(key,'old_accuracy'):.2f}% | {mean(key,'new_accuracy'):.2f}% | {extra:,} B | {total:,} B |")
lines+=['', 'Primary contrast, local16D×4 minus original local: **'+difference(('multicue',16,4),('original_local',None,None))+'**.','',
'Orders are averaged within each of five independent seeds before SD/interval calculation. Intervals describe this dataset/protocol and are not multiplicity-adjusted. Seven capacity shapes were declared and all are reported; the primary comparison was specified before confirmation. The direct ridge classifier is effectively deterministic because its sufficient statistics use fixed keys and the same examples; its zero seed SD is not five independently learned representations.','',
'![Memory versus accuracy](multicue_capacity.png)','',
'## Mechanism','',
'The earlier local association mechanism compressed every branch’s input history into one average. Different digit styles and inputs can therefore be merged into a single unrepresentative cue. The new mechanism allocates several input prototypes to each branch. During the original stream, current fixed image features are assigned to the nearest prototype belonging to the currently selected branch; that prototype’s sum and count are updated. At inference, each branch is scored by its best-matching prototype, and each neuron selects its branch. No complete historical configuration is supplied.','',
'The four-cue16D condition trades feature resolution for multiple modes: it uses27,200bytes, versus26,000bytes for one64Daverage. The additional4.6% includes the extra counts. Feature dimensions16/32/64 and one-cue controls separate resolution from multimodal capacity; this is not an exact-byte equality or an isolated slot-count intervention. Full64Dfour-cue memory uses104,000bytes. Input cues are explicit memory, initialized from current batch features and updated as aggregates; claiming that no input information is stored would be false. They are never replayed into the neural learner.','',
'The neural backbone stays the same one-layer25×4 model (78,760trainable parameters plus400bytes of fading usefulness). It is trained using current labels, factual cross-entropy and gradient-estimated usefulness. The cue memories see only current fixed keys and pre-update selected branch indices. No classlabels, taskIDs, task-boundary signals or future phase-end routes enter association writes. Predictions use current image features and final model weights, with all ten digits eligible. This remains task-ID-free class-incremental inference under a benchmark stream constructed from digit pairs.','',
'## What failed before this result','',
'- Whole-image activity cues remained better than restricted patch cues in the initial validation pilot. Replacing activity targets with per-image best single-neuron alternatives also failed to improve that pilot.','- A local MSE/softmax delta-rule association pilot reached at most29.98%validation accuracy versus61.16%for cumulative/fading averages. All40delta candidates are retained, including one unstable setting; no additional learning-rate search was pursued.','- Local multicomponent cues improved on two development seeds before confirmation. Settings were locked by mean validation performance on560/561. They were not chosen from the official test outcomes.','',
'## Computational costs','', '| Method | Batch1 inference | Batch128 inference | Accepted memory write, batch128 |','|---|---:|---:|---:|']
for key in keys:
 p=next(p for p in profiles if matches(p,key));z=p['inference_ms'];lines.append(f"| {label(key)} | {z['1']:.3f} ms | {z['128']:.3f} ms | {p['accepted_write_ms_batch128']:.3f} ms |")
lines+=['',json.loads((RAW/'profile.json').read_text())['scope'],'',
'Counts exclude optimizer state, transient working memory, Python overhead and checkpoints. Total inference state includes315,440bytes of the backbone for branch-retrieval methods, but none for direct classifiers. Ridge timings solve the statistic system on each inference call; no derived-weight cache is counted. The combined all-candidate training runtime is not the deployed cost of one method.','',
'## Protocol and limits','',
'- Development: seeds560/561, separate stratified50kfit/10kvalidation splits of training data. All28multicue settings per seed retained. Seven predeclared dimension/slot combinations each select decay1/.995 and route stability1/10 by mean validation accuracy.','- Confirmation: fresh562–566, forward/reverse five digit pairs, full60ktraining/full10ktest. Two passes per current pair, Adam.003, batch128, gradientclip5. No old-task replay, pretraining, inputtaskIDs, test-label feedback or external route bank in predictions. All120automaticreadouts retained.','- The data loader returns both raw splits, but development code uses only training arrays; official test metrics/labels do not select settings. Phase identities construct the stream and analysis only. Stability gates never reset at boundaries.','- Evaluation checks unchanged network and associative states. Historical banks are stored only for an explicitly task-assisted oracle after automatic predictions; they are not available to retrieval.','- Main evidence is improved branch association at an approximately fixed extra-memory budget. There is no matched ordinary-network sidecar ablation here. Another architecture implementing the same factorized memory can in principle behave similarly. No unique biological or dendritic mechanism, top-conference readiness, or breakthrough is established.','',
'## Reproducibility','',
'- [Locked protocol](../../results/overnight_branch_research_20261024/multicue_confirmation/protocol.json), [settings](../../results/overnight_branch_research_20261024/multicue_confirmation/selection.json), [all confirmation results](../../results/overnight_branch_research_20261024/multicue_confirmation/results.json), [saved states](../../results/overnight_branch_research_20261024/multicue_confirmation/memories).','- [Confirmation driver](../../bdh_cl/scripts/overnight_multicue_confirmation.py), [prototype pilot](../../bdh_cl/scripts/overnight_multicue_pilot.py), [timing driver](../../bdh_cl/scripts/overnight_multicue_profile.py).','- [Activity/usefulness and patch pilot](../../results/overnight_branch_research_20261024/cue_pilot), [local delta pilot](../../results/overnight_branch_research_20261024/local_delta_pilot), [multicue development560](../../results/overnight_branch_research_20261024/multicue_pilot), [development561](../../results/overnight_branch_research_20261024/multicue_pilot_seed561).']
(OUT/'MULTICUE_REPORT.md').write_text(pretty('\n'.join(lines)));print('report_complete')
