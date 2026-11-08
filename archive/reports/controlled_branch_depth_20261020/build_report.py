import json, statistics as st, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;RAW=ROOT/'results/controlled_branch_depth_20261020'
rows=json.loads((RAW/'results.json').read_text());done=json.loads((RAW/'completion.json').read_text());assert done['complete'] and len(rows)==90
arms=['mlp1','mlp2','branch1_exact','branch2_exact','branch1_proxy','branch2_proxy'];names={'mlp1':'MLP, one layer','mlp2':'MLP, two layers','branch1_exact':'Branches, one layer, exact','branch2_exact':'Branches, two layers, exact','branch1_proxy':'Branches, one layer, gradient estimate','branch2_proxy':'Branches, two layers, gradient estimate'}
profiles={a:json.loads((RAW/f'profile_{a}.json').read_text()) for a in arms};selection=json.loads((RAW/'selection.json').read_text())
def get(a,mode='sequential'):
 r=[x for x in rows if x['arm']==a and x['mode']==mode];assert len(r)==(10 if mode=='sequential' else 5);return r
def values(a,fn,mode='sequential'):return [100*st.mean(fn(x) for x in get(a,mode) if x['seed']==seed) for seed in range(531,536)]
def mean(a,fn,mode='sequential'):return st.mean(values(a,fn,mode))
def diag(x):return x['oracle_routes']['accuracy'] if x['arm'].startswith('branch') else x['oracle_pair_output_mask']['accuracy']
def ci(v):
 m=st.mean(v);h=2.776445105*st.stdev(v)/math.sqrt(5);return f'{m:+.2f} pp (95% t interval {m-h:+.2f} to {m+h:+.2f})'
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for i,a in enumerate(arms):
 axes[0].bar(i,mean(a,lambda x:x['accuracy'],'joint'),color='#507f59');axes[1].bar(i-.18,mean(a,lambda x:x['accuracy']),.35,color='#a4b5c4');b=axes[1].bar(i+.18,mean(a,diag),.35,color='#236e9e')
 if a.startswith('mlp'):b[0].set_hatch('//')
 for val in values(a,diag):axes[1].scatter(i+.18,val,s=9,color='#152634',zorder=3)
for ax in axes:
 ax.set_xticks(range(6),['MLP\n1 layer','MLP\n2 layers','Exact\n1 layer','Exact\n2 layers','Estimate\n1 layer','Estimate\n2 layers']);ax.set_ylim(0,105);ax.set_ylabel('Test accuracy (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_title('Joint training reference');axes[1].set_title('Sequential: automatic (grey), task-assisted (blue)')
fig.savefig(OUT/'depth_accuracy.png',dpi=180);fig.savefig(OUT/'depth_accuracy.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');labels=['Latest both','Historical first','Historical second','Historical both'];keys=['all_latest','first_historical','second_historical','all_historical']
for j,rule in enumerate(['exact','proxy']):
 a='branch2_'+rule
 vv=[mean(a,lambda x,k=k:x['layer_restoration'][k]['accuracy']) for k in keys];xx=[i+(j-.5)*.36 for i in range(4)];bars=ax.bar(xx,vv,.34,label='Exact scoring' if rule=='exact' else 'Gradient estimate')
 for b,v in zip(bars,vv):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=9)
ax.set_xticks(range(4),labels);ax.set_ylim(0,105);ax.set_ylabel('All-ten-class accuracy (%)');ax.set_title('Two-layer route restoration, final weights unchanged');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.savefig(OUT/'layer_restoration.png',dpi=180);fig.savefig(OUT/'layer_restoration.svg');plt.close(fig)
lines=['# Controlled depth comparison and layer-specific recovery','',
'Completed on 20 October 2026. Across five fresh seeds and both class orders, oracle recovery was 91.19% → 88.27% for exact scoring and 93.67% → 83.74% for gradient estimates when moving from one to two hidden layers at nearly equal parameter budget. The exact-score depth difference is uncertain; the estimated-score decline is clearer under this protocol. Restoring both layer configurations outperforms either alone. All automatic sequential accuracies remain near 20%; automatic retrieval is not included.','',
'## Accuracy','', '| Model | Parameters | Joint | Sequential automatic | Current-pair acquisition | Oracle / task-assisted | Oracle seed SD |', '|---|---:|---:|---:|---:|---:|---:|']
for a in arms:
 r=get(a)[0];lines.append(f"| {names[a]} | {r['parameters']:,} | {mean(a,lambda x:x['accuracy'],'joint'):.2f}% | {mean(a,lambda x:x['accuracy']):.2f}% | {mean(a,lambda x:x['acquisition']):.2f}% | {mean(a,diag):.2f}% | {st.stdev(values(a,diag)):.2f} pp |")
lines+=['','Branch oracle: the true digit pair selects saved hidden configurations, with final weights/head unchanged and all ten output classes eligible. MLP assistance: the true digit pair masks logits to two outputs. These are different interventions and cannot establish a pure architecture advantage. All automatic predictions allow ten classes and use no test labels or task ID.','', '![Depth accuracy](depth_accuracy.png)','',
'## Does depth reduce recoverable accuracy?','', 'Paired differences below are **two layers minus one layer**, averaging both orders within each seed. Intervals use five independent seed differences and a t distribution (four degrees of freedom); they describe this protocol, not all datasets or depths.','']
for rule in ['exact','proxy']:
 v1=values('branch1_'+rule,diag);v2=values('branch2_'+rule,diag);lines.append(f'- {rule}: {ci([b-a for a,b in zip(v1,v2)])}.')
lines+=['','At fixed parameter budget, depth trades off against layer width: one branch layer has 25 neurons × 4 branches; two layers each have 24 × 4. Thus this tests practical depth under a budget, not an isolated added layer with identical width and initialization. Learning rates are independently validation-selected. Five seeds do not establish a universal depth effect.','',
'The old 98.02% result used one 24×4 layer, an auxiliary alternative loss and different seeds/settings. These new same-loss results do not directly replicate that earlier condition or prove that depth alone explains the historical difference.','',
'## Which layer configurations must be restored?','', '| Scoring rule | Neither (latest routes) | First historical only | Second historical only | Both historical |', '|---|---:|---:|---:|---:|']
for rule in ['exact','proxy']:
 a='branch2_'+rule;v=[mean(a,lambda x,k=k:x['layer_restoration'][k]['accuracy']) for k in keys];lines.append('| '+rule+' | '+' | '.join(f'{x:.2f}%' for x in v)+' |')
lines+=['','![Layer restoration](layer_restoration.png)','',
'For each image, the correct pair selects that pair’s snapshot in the indicated layer(s). Other layers retain their final global route. The all-historical condition exactly matches the ordinary route oracle; all-latest matches automatic accuracy. Restoring a single layer can disrupt compatibility with the other layer. These interventions identify configuration dependencies, not the anatomical location of all stored knowledge.','',
'## Cost','', '| Model | CPU ms / training step | Versus same-depth MLP | Persistent model state | Saved forward storage |', '|---|---:|---:|---:|---:|']
for a in arms:
 p=profiles[a];d='1' if ('1' in a) else '2';baseline=profiles['mlp'+d]['training_step_median_ms'];r=get(a)[0];lines.append(f"| {names[a]} | {p['training_step_median_ms']:.3f} | {p['training_step_median_ms']/baseline:.2f}× | {r['inference_model_bytes']:,} bytes | {p['autograd_retained_unique_nonparameter_storage_bytes']/1e6:.3f} MB |")
lines+=['','Fresh process per arm, two CPU threads, batch 128, ten warmup updates and five blocks of twenty timed training steps; optimizer and routing updates included. Saved storage is deduplicated nonparameter storage retained for backward, including input, not total peak RAM. Exact alternatives run without gradients but still allocate transient workspace. No GPU or large-depth scaling inference follows. Detailed batch-1/batch-128 inference and process RSS are available in profile JSON files.','',
'## Protocol and verification','',
'- Real MNIST, full 60,000 training / 10,000 official test images in confirmation. Pixels divided by 255, tanh hidden responses, shared ten-output linear classifier.','- Five pairs (0/1 through 8/9), forward and reverse orders. Two epochs per current pair, batch 128, Adam, gradient clipping 5. No replay, pretraining, input-dependent retrieval, usage protection or boundary reset.','- All arms train factual cross-entropy only. Exact single-neuron alternative errors are computed without differentiation; approximate errors use factual backpropagated gradients. Error EMA 0.95 selects a global branch per neuron.','- Development seed 530 uses a stratified 50k fit / 10k validation split. For each arm and sequential/joint setting, choose among learning rates 0.0003, 0.001, 0.003 using final validation accuracy, with acquisition as a sequential tiebreak. Official test outcomes never choose settings.','- Fresh seeds 531–535: ten sequential and five independently trained joint runs per arm. Both orders are paired within seed. Total 36 development and 90 confirmation runs.','- Joint runs mix all classes with the same 120,000 example presentations as sequential training. No historical task snapshots affect ordinary training or predictions.','- Parameter count differences are below 1%. Branch state adds 400 bytes (one layer) or 768 bytes (two layers). A packed five-route oracle bank adds 125 or 240 bytes and is diagnostic-only.','- Exact alternatives match directly altered-route forwards in both depths; selected-row inference matches ordinary forwards; gradient derivatives match finite differences. Evaluation asserts unchanged model state. Both and neither restoration agree with independently computed oracle/automatic metrics.','',
'## Selected learning rates','', '| Arm | Sequential | Joint |','|---|---:|---:|']
for a in arms:lines.append(f"| {a} | {selection['sequential'][a]['chosen']['lr']} | {selection['joint'][a]['chosen']['lr']} |")
lines+=['','## Reproducibility','',
'- [Driver](../../bdh_cl/scripts/controlled_branch_depth.py), [archived driver](../../results/controlled_branch_depth_20261020/source.py), [protocol](../../results/controlled_branch_depth_20261020/protocol.json).','- [All validation candidates](../../results/controlled_branch_depth_20261020/selection.json), [90 confirmation records including per-pair restoration](../../results/controlled_branch_depth_20261020/results.json), [checks](../../results/controlled_branch_depth_20261020/verification.json), [checkpoints](../../results/controlled_branch_depth_20261020/models).','- Driver reuses the training/evaluation and profiling harness from [the preceding multilayer experiment](../../bdh_cl/scripts/multilayer_branch_comparison.py), adapting its MLP arm detection and replacing its architecture/update rule. No auxiliary objective is used.','',
'## Next decision','',
'Use the matched depth and restoration results to choose a backbone before adding online retrieval. A high oracle is a useful target but remains task-assisted. An autonomous follow-up must learn current input/activity associations during training, retrieve without labels/IDs, count all memory, and retain matched shared-memory and direct-input classifier controls.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:18]));print('\n'.join(lines[lines.index('## Which layer configurations must be restored?'):lines.index('## Which layer configurations must be restored?')+8]))
