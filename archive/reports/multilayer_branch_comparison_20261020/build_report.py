import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;base=ROOT/'results/multilayer_branch_comparison_20261020'
r=json.loads((base/'results.json').read_text());c=json.loads((base/'exact_ce_control/results.json').read_text());assert len(r)==27 and len(c)==9 and json.loads((base/'completion.json').read_text())['complete'] and json.loads((base/'exact_ce_control/completion.json').read_text())['complete'];r+=c
arms=['mlp','branch_exact','branch_exact_ce','branch_proxy'];names={'mlp':'Ordinary MLP','branch_exact':'Exact routing + auxiliary loss','branch_exact_ce':'Exact routing, CE only','branch_proxy':'Gradient-estimated routing, CE only'}
profiles={arm:json.loads((base/('exact_ce_control/profile.json' if arm=='branch_exact_ce' else f'profile_{arm}.json')).read_text()) for arm in arms}
def get(arm,mode='sequential'):
 a=[x for x in r if x['arm']==arm and x['mode']==mode];assert len(a)==(6 if mode=='sequential' else 3);return a
def seedvalues(a,func):return [st.mean(func(x) for x in a if x['seed']==s) for s in sorted(set(x['seed'] for x in a))]
def avg(a,func):return 100*st.mean(seedvalues(a,func))
def diagnostic(x):return x['oracle_pair_output_mask']['accuracy'] if x['arm']=='mlp' else x['oracle_routes']['accuracy']
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for i,arm in enumerate(arms):
 a=get(arm);j=get(arm,'joint');axes[0].bar(i,avg(j,lambda x:x['accuracy']),.65,color='#477d4e');axes[0].scatter([i]*3,[v*100 for v in seedvalues(j,lambda x:x['accuracy'])],color='#152634',s=12)
 axes[1].bar(i-.18,avg(a,lambda x:x['accuracy']),.34,color='#a4b5c4',label='Automatic, all ten outputs' if i==0 else None);bar=axes[1].bar(i+.18,avg(a,diagnostic),.34,color='#236e9e',label='Task-assisted diagnostic' if i==0 else None)
 if arm=='mlp':bar[0].set_hatch('//')
for ax in axes:ax.set_xticks(range(4),['MLP','Exact +\nauxiliary','Exact\nCE only','Gradient\nestimate']);ax.set_ylim(0,105);ax.set_ylabel('Test accuracy (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_title('Joint training: all classes mixed');axes[1].set_title('Sequential learning: final weights');axes[1].legend(fontsize=7,frameon=False)
fig.savefig(OUT/'accuracy.png',dpi=180);fig.savefig(OUT/'accuracy.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');colors=['#a4b5c4','#b15745','#c18a5e','#236e9e'];baseline=profiles['mlp']['training_step_median_ms']
for i,arm in enumerate(arms):
 ratio=profiles[arm]['training_step_median_ms']/baseline;axes[0].bar(i,ratio,color=colors[i]);axes[0].text(i,ratio+.12,f'{ratio:.2f}×',ha='center');mb=profiles[arm]['autograd_retained_unique_nonparameter_storage_bytes']/1e6;axes[1].bar(i,mb,color=colors[i]);axes[1].text(i,mb+.12,f'{mb:.2f}',ha='center')
for ax in axes:ax.set_xticks(range(4),['MLP','Exact +\nauxiliary','Exact\nCE only','Gradient\nestimate']);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Training step time relative to MLP');axes[0].set_title('Fresh-process CPU profiles, batch 128');axes[1].set_ylabel('Unique autograd-retained storage (MB)');axes[1].set_title('Saved forward storage; not total peak RAM')
fig.savefig(OUT/'compute.png',dpi=180);fig.savefig(OUT/'compute.svg');plt.close(fig)
lines=['# Two-hidden-layer branches versus a parameter-matched MLP','',
'Completed on 20 October 2026. **The gradient-estimated branch version retains high oracle-accessible accuracy at about 20% CPU training-step overhead, while exhaustive deep counterfactual training is about ten times slower. Neither network autonomously solves Split MNIST: all variants finish near 20% without an input-dependent retrieval mechanism.**','',
'## Architectures and parameter budgets','',
'| Architecture | Hidden computation | Trainable parameters | Inference numeric model state |','|---|---|---:|---:|']
for arm in ['mlp','branch_proxy']:
 x=get(arm)[0];lines.append(f'| {"Ordinary MLP" if arm=="mlp" else "Two-layer branch network"} | {"784 → 88 tanh → 88 tanh → 10" if arm=="mlp" else "784 → 24×4 tanh branches → 24 pooled → 24×4 tanh branches → 24 pooled → 10"} | {x["parameters"]:,} | {x["inference_model_bytes"]:,} bytes |')
lines+=['',
'Trainable parameter counts differ by 208, or approximately 0.27%. The branch model adds 192 FP32 usefulness estimates (768 bytes), one for each branch in both layers. Output widths differ: 24 pooled units per branch layer versus 88 ordinary units. Matching parameters does not match every capacity constraint or learning rule. Both architectures have two nonlinear hidden layers and one shared linear ten-class head.','',
'## Learning rules','',
'- **MLP:** ordinary cross-entropy and backpropagation.','- **Exact branches + auxiliary:** restore the original-style rule at both hidden layers. For each of 192 alternatives, replace one neuron’s selected response with one branch, holding all other route choices fixed. First-layer alternatives recompute the nonlinear second layer; second-layer alternatives use the exact linear-head shortcut. Train factual cross-entropy plus 0.5 mean alternative cross-entropy. An EMA of exact alternative errors selects each layer’s branches.','- **Gradient-estimated branches:** train factual cross-entropy only. After one backward pass, estimate each alternative loss as L + Σ_batch[(∂L/∂h_n)(b_(n,k)−h_n)]. The sum accounts for the 1/batch-size already included in mean-loss gradients. Apply the same EMA 0.95 and minimum-score branch selection. No class-logit tensor per alternative or auxiliary objective is computed.','- **Exact branches, CE only:** a follow-up control calculates exact alternative errors without differentiating them and trains factual cross-entropy only. It separates the routing approximation from removal of the auxiliary loss. It has its own validation-selected learning rates.','',
'The gradient estimate is a first-order approximation, not an exact alternative error. All layers still receive ordinary backpropagated gradients; this is not a biological local-learning claim. Hard routes are global across images at each training moment, selected from recent usefulness. No usage-based protection or online associative retrieval sidecar is included here.','',
'## Accuracy','',
'| Learning variant | Joint training | Sequential automatic | Current-task acquisition | Task-assisted diagnostic |','|---|---:|---:|---:|---:|']
for arm in arms:
 a=get(arm);j=get(arm,'joint');kind='pair output mask' if arm=='mlp' else 'both old layer routes';lines.append(f'| {names[arm]} | {avg(j,lambda x:x["accuracy"]):.2f}% | {avg(a,lambda x:x["accuracy"]):.2f}% | {avg(a,lambda x:x["acquisition"]):.2f}% | {avg(a,diagnostic):.2f}% ({kind}) |')
lines+=['',
'Joint training mixes all classes and is a capacity/acquisition reference, not continual learning. Sequential automatic predictions always consider all ten digits. The network’s latest global configuration remains fixed at test time, so it mostly recognizes the newest pair.','',
'For branch oracle recovery, the correct digit pair selects the two-layer configuration saved when that pair was learned. The final weights and classifier remain unchanged and all ten logits remain eligible. For the MLP diagnostic, the true pair explicitly masks the final classifier to two outputs. These are different task-assisted interventions; their numbers do not constitute a pure, equal-intervention architecture comparison or task-ID-free performance. Neither diagnostic means a plain MLP has lost all old information.','',
'![Accuracy comparisons](accuracy.png)','',
'The hatched MLP task-assisted bar uses pair masking; branch bars use historical hidden routes without output masking. No test data or task ID enters automatic inference.','',
'## Paired seed results and uncertainty','',
'| Variant | Joint accuracy seed SD | Automatic accuracy seed SD | Task-assisted accuracy seed SD |','|---|---:|---:|---:|']
for arm in arms:
 a=get(arm);j=get(arm,'joint');lines.append(f'| {names[arm]} | {100*st.stdev(seedvalues(j,lambda x:x["accuracy"])):.2f} pp | {100*st.stdev(seedvalues(a,lambda x:x["accuracy"])):.2f} pp | {100*st.stdev(seedvalues(a,diagnostic)):.2f} pp |')
lines+=['',
'The two class orders are averaged within seed before computing sequential SD; they are not six independent seeds. There are only three fresh seeds. The gradient estimate’s higher mean oracle recovery is descriptive, not evidence of statistical superiority or universal approximation quality.','',
'## Computational efficiency','',
'Each timing profile runs in a fresh process after accuracy experiments finish, on CPU with two Torch threads and synthetic batches of 128. Training uses ten warmup updates followed by five blocks of twenty updates. Values are median milliseconds per full training step, including optimizer, routing and usefulness updates.','',
'| Variant | Training step | Relative to MLP | Mean full sequential training time |','|---|---:|---:|---:|']
for arm in arms:
 p=profiles[arm];a=get(arm);lines.append(f'| {names[arm]} | {p["training_step_median_ms"]:.3f} ms | {p["training_step_median_ms"]/baseline:.2f}× | {st.mean(x["training_seconds"] for x in a):.3f} s |')
lines+=['',
'Full sequential training time covers 120,000 current-example presentations, excluding dataset loading, initialization and evaluations. It is recorded during the accuracy experiments and is less controlled than the dedicated profiles. Joint runs have the same number of training presentations; minibatch rounding differs slightly across task partitions. These measurements do not establish GPU performance or large-depth scaling.','',
'![Training cost and saved activation storage](compute.png)','',
'## Memory measurements','',
'| Variant | Model + Adam persistent numeric state | Autograd-retained unique nonparameter storage | Fresh-process peak RSS after training |','|---|---:|---:|---:|']
for arm in arms:
 a=get(arm)[0];p=profiles[arm];lines.append(f'| {names[arm]} | {a["inference_model_bytes"]+a["optimizer_bytes"]:,} bytes | {p["autograd_retained_unique_nonparameter_storage_bytes"]/1e6:.3f} MB | {p["process_peak_rss_after_training_bytes"]/1e6:.1f} MB |')
lines+=['',
'Autograd storage counts deduplicated tensor backing storage saved for backward, excluding model parameters and including the input. It is not total peak tensor memory: exact CE-only alternatives allocate temporary forward workspace even though their graphs are not retained. Process RSS includes Python, Torch, allocator caches, optimizer state and workspaces; it must not be interpreted as model storage. The profile also records the earlier process high-water mark, not a clean current-RSS subtraction.','',
'Both branch variants store nearly the same model state as the MLP. The exhaustive auxiliary graph, rather than routing state, accounts for its much larger retained activation storage. A bank of five two-layer configurations would add 240 packed uint8 indices/bytes; it is used only in the oracle audit and excluded from ordinary model inference state.','',
'## Inference latency','',
'Inference profiles use twenty warmups and ten blocks of 100 calls. Branch timings below use the gradient-estimated model as a representative of the shared architecture.','',
'| Architecture/path | Batch 1 | Batch 128 |','|---|---:|---:|']
for arm,path,label in [('mlp','standard','Ordinary MLP'),('branch_proxy','standard','Branches, compute all responses'),('branch_proxy','selected_weights','Branches, gather only selected weight rows')]:
 d={x['batch']:x['median_ms'] for x in profiles[arm]['inference'] if x['path']==path};lines.append(f'| {label} | {d[1]:.4f} ms | {d[128]:.4f} ms |')
lines+=['',
'The selected-row path is mathematically equivalent for a supplied fixed configuration and was verified against the full forward. It gathers rows on every call; no packed-weight cache is assumed. Gathering overhead hurts single-image latency, while reduced matrix computation helps at batch 128. There is no input-dependent router cost in these numbers. Different per-image configurations would need a separate benchmark.','',
'## Training protocol and controls','',
'All models use cached real MNIST images scaled to [0,1], two passes over each current class pair, batch 128, Adam and clipping 5. No pretraining, replay, task-ID input, task output mask in primary predictions, or boundary reset is used. Configuration snapshots are external diagnostic artifacts, not an autonomous archive.','',
'Development seed 520 uses 50,000 fitting and 10,000 held-out training images, stratified at 1,000 per digit. Each variant independently selects learning rate 0.0003/0.001/0.003. Sequential selection uses final all-class validation accuracy, with acquisition as a tiebreak; joint selection is independent. Official test labels never select settings.','',
'Confirmation uses fresh seeds 521/522/523, both forward and reverse pair order, all 60,000 training images and the full 10,000-image test set. Branch variants share initial weights and minibatch order for a seed/order; the MLP shares data/order but necessarily has different-shaped initial weights. Joint controls are trained independently from random initialization.','',
'The exact CE-only control was added after recognizing that the primary exact/proxy comparison changed both routing estimates and the auxiliary objective. Its settings were chosen on the same held-out development data before its own confirmation. It reuses the paired cohort and is not independent replication.','',
'## Interpretation and next step','',
'This small two-hidden-layer experiment supports a practical lead: the gradient-based usefulness estimate can be trained near ordinary MLP cost while preserving substantial accuracy accessible through old layer configurations. It avoids the expensive exact downstream counterfactual computation. It does not preserve the original algorithm unchanged, guarantee the same behavior at greater depth, or establish autonomous continual learning.','',
'The MLP remains better in joint learning under the selected budgets. The branch model’s advantage is currently in the configuration-conditioned diagnostic, with different forms of assistance across architectures. The next stage is online input-driven retrieval of configurations at both layers, compared with ordinary memory-based controls and with all extra routing state counted. Earlier results already showed that a direct fixed-key RLS classifier is a strong control; it must not be omitted.','',
'## Reproducibility','',
'- [Main driver](../../bdh_cl/scripts/multilayer_branch_comparison.py), [protocol](../../results/multilayer_branch_comparison_20261020/protocol.json), [selection and all development candidates](../../results/multilayer_branch_comparison_20261020/selection.json), [27 main confirmation runs](../../results/multilayer_branch_comparison_20261020/results.json), [sequential checkpoints](../../results/multilayer_branch_comparison_20261020/models).','- [Same-loss control driver](../../bdh_cl/scripts/multilayer_exact_ce_control.py), [control protocol](../../results/multilayer_branch_comparison_20261020/exact_ce_control/protocol.json), [control selection](../../results/multilayer_branch_comparison_20261020/exact_ce_control/selection.json), [nine control confirmation runs](../../results/multilayer_branch_comparison_20261020/exact_ce_control/results.json).','- CPU profiles: [MLP](../../results/multilayer_branch_comparison_20261020/profile_mlp.json), [exact + auxiliary](../../results/multilayer_branch_comparison_20261020/profile_branch_exact.json), [gradient estimate](../../results/multilayer_branch_comparison_20261020/profile_branch_proxy.json), [exact CE-only](../../results/multilayer_branch_comparison_20261020/exact_ce_control/profile.json).','- Verified full downstream exact alternatives against directly changed routes, selected-row inference against the ordinary branch forward, first-order gradients against finite differences, parameter counts, model-state invariance during evaluations, and completion of 24 development plus 36 confirmation runs.','']
(OUT/'REPORT.md').write_text('\n'.join(lines));print('All36confirmationruns,24developmentrunsandfourisolatedprofilesverified;reportgenerated.')
