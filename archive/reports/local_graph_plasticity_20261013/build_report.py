import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2];out=Path(__file__).parent
primary=Path('results/local_graph_plasticity_20261013');matched=Path('results/local_graph_matched_slowing_20261013');transfer=Path('results/local_graph_recruitment_transfer_20261013')
p=json.loads((root/primary/'results.json').read_text());q=json.loads((root/matched/'results.json').read_text());t=json.loads((root/transfer/'results.json').read_text());selection=json.loads((root/primary/'selection.json').read_text());slowsel=json.loads((root/matched/'selection.json').read_text())
full=lambda rows:[r for r in rows if r['metric_scope']=='full10000test']
models={}
for ds in ['mnist','fashion_mnist']:
 models[ds]={
 'Original branches':[r for r in full(p) if r['dataset']==ds and r['name']=='original'],
 'Novelty recruitment':full(t) if ds=='mnist' else [r for r in full(p) if r['dataset']==ds and r['name']=='selected_local'],
 'Constant MLP':[r for r in full(p) if r['dataset']==ds and r['name']=='mlp_constant'],
 'Primary slowing MLP':[r for r in full(p) if r['dataset']==ds and r['name']=='mlp_slows'],
 'Stronger slowing MLP':[r for r in full(q) if r['dataset']==ds and r['name']=='acquisition_match_selected']}
 for rs in models[ds].values():assert len(rs)==3
colors=['#3977b8','#167f79','#d67b27','#bd4553','#8b58a2']
fig,axs=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
for ax,ds in zip(axs,['mnist','fashion_mnist']):
 for (label,rs),color in zip(models[ds].items(),colors):
  x=[100*r['first_four'] for r in rs];y=[100*r['newest'] for r in rs]
  ax.errorbar(st.mean(x),st.mean(y),xerr=st.stdev(x),yerr=st.stdev(y),color=color,marker='^' if 'MLP' not in label else 'o',capsize=3,label=label,linestyle='none')
 ax.set_xlim(15,85);ax.set_ylim(15,100);ax.set_xlabel('Accuracy on the first four domains (%)');ax.set_ylabel('Accuracy on the newest domain (%)');ax.set_title('MNIST' if ds=='mnist' else 'Fashion MNIST');ax.grid(alpha=.2)
axs[0].legend(fontsize=9,loc='lower left')
fig.suptitle('After 32 domains: local recruitment improves Fashion MNIST, but slower SGD is a strong control\nSame fixed stored-size budget; three paired seeds; bars show one sample standard deviation',fontsize=12)
fig.savefig(out/'retention_acquisition.png',dpi=180);fig.savefig(out/'retention_acquisition.svg');plt.close(fig)
lines=['# Local branch plasticity: what blocks acquisition, and can recruitment fix it?','','The local recruitment rule substantially improves the branch model on Fashion MNIST without adding stored state. It does not establish a general advantage over a comparably sized ordinary MLP with a presentation-based learning-rate schedule. The fixed recruitment rule fails to improve overall MNIST accuracy on the paired transfer check. These positive and negative findings are both retained.','',
'## Controlled changes','',
'All local arms use the same 64 neurons × 4 branches, 256 input-response units, fixed class-code table, input weights, previous-minibatch error state, usage state, neighbor coactivity tables and inference decoder. Persistent numeric storage remains 812,480 bytes. The SGD MLP is 784→256→10 with 814,120 weight/bias bytes, no momentum or replay. A scalar training-presentation counter drives its schedule; it is not an image/configuration memory bank. Temporary working storage and Python overhead are excluded.','',
'1. **Protection on/off:** original rate is η/(1+successful usage/τ); the off arm uses constant η. Successful-usage state is still recorded for a controlled comparison.','2. **Local MSE/BCE gradients:** original update is input × (desired−response) × (1−response²). BCE uses input × (desired−response), the exact gradient for probability (1+tanh(a))/2 and target (1+code)/2. Nominal η, codes, branch-error selection and reliable coactivity stay fixed. This tests the saturation factor, not removal of coded teaching. Gradient magnitudes differ, and this is not a broad independently tuned comparison of all teaching schemes.','3. **Novelty-triggered recruitment:** if a neuron’s best branch has previous-minibatch mean squared error greater than 0.4, select its least-used branch instead. Otherwise select its lowest-error branch as before. The threshold is predefined, not selected on test data. It uses existing local error/usage state and does not receive a task boundary or ID. A high local error is not proof of a task change.','4. **Relative recycling:** additionally measure damping relative to the least-used branch in that neuron, keeping one candidate more plastic. This changes allocation and protection together; interpret it as a package.','5. **Slowing MLP:** η(t)=η₀/(1+t/s), with t the number of current training presentations. No domain resets or boundary information. The primary schedules use s=100,000/400,000; a follow-up broadens to 1,000/3,000/10,000/30,000. Validation also chooses schedules with recent-domain accuracy close to the branch model.','',
'## Protocol and selection','',
'MNIST and Fashion MNIST, all ten classes in every domain, identity followed by random pixel permutations. No task IDs, replay, input prototypes or saved configuration archive enters the learner. Each current domain receives two passes; original source images reappear transformed in later domains as in standard permuted-MNIST.','',
'Development uses fresh seeds 360/370, a stratified 50,000/10,000 training split and a fixed 2,000-example validation subset, excluded from every training domain. Six local variants and the MLP schedules are compared at 16 domains. Local selection maximizes the harmonic mean of all-domain and recent-four accuracy. A separate success gate requires newest accuracy +5 points, first-four accuracy within −5 points, and overall accuracy within −2 points of the original branch validation result.','',
'Fashion novelty recruitment passes that gate. The selected MNIST relative-recycling variant fails it and is reported as a diagnostic, not a solution. Confirmation uses three fresh paired seeds per dataset (361–363 / 371–373), 32 domains, all 60,000 training samples, and all 10,000 test samples per domain at the final checkpoint. Intermediate curves use a fixed stratified 2,000-example test subset. Hyperparameters stay fixed. No test labels are supplied to prediction or used to select settings.','',
'The stronger slowing-MLP and MNIST recruitment-transfer checks reuse the same confirmation seeds, so they are paired follow-ups, not extra independent replications. The recruitment transfer is explicitly adaptive: it tests the already defined rule on MNIST after observing its Fashion confirmation. No new threshold or update rule is tuned. Slowing schedules match recent-domain validation accuracy at 16 domains; they are not assumed to remain exactly acquisition-matched at 32. Actual gaps are shown below.','',
'## Development ablations','',
'| Dataset | Variant | All-domain accuracy | First four | Newest |','|---|---|---:|---:|---:|']
for ds,sel in selection.items():
 for c in sel['candidates']:lines.append(f'| {ds} | {c["name"]} | {100*c["accuracy"]:.2f}% | {100*c["first_four"]:.2f}% | {100*c["newest"]:.2f}% |')
lines+=['','Removing protection raises newest-domain accuracy but substantially reduces early retention. Changing the local gradient alone has little effect under absolute protection. Its acquisition benefit is larger without protection, especially on Fashion MNIST. Novelty recruitment is the clearest favorable change on Fashion validation; relative recycling does not preserve old accuracy on MNIST.','',
'## Full-test confirmation at 32 domains','',
'| Dataset | Model | All-domain accuracy | First-four accuracy | Newest accuracy |','|---|---|---:|---:|---:|']
for ds,model in models.items():
 for name,rs in model.items():
  m=lambda k:100*st.mean(r[k] for r in rs)
  lines.append(f'| {ds} | {name} | {m("accuracy"):.2f}% | {m("first_four"):.2f}% | {m("newest"):.2f}% |')
lines+=['',f'![Retention and acquisition]({out}/retention_acquisition.png)','',
'Primary slowing MLP uses the independently selected 400,000 schedule on both datasets. Stronger slowing MLP is selected to approximate the selected local variant’s recent-four validation accuracy: 10,000 on MNIST and 100,000 on Fashion. It was not selected by these test scores. The MNIST validation target was relative recycling, not the later novelty-transfer arm. See the complete control table below rather than interpreting this as exact matching of all final methods.','',
'## Every additional confirmation, including unsuccessful changes','',
'| Dataset | Follow-up / diagnostic | All-domain accuracy | First four | Recent four | Newest |','|---|---|---:|---:|---:|---:|']
for ds in ['mnist','fashion_mnist']:
 rs=[r for r in full(p) if r['dataset']==ds and r['name']=='selected_local'];m=lambda k:100*st.mean(r[k] for r in rs)
 lines.append(f'| {ds} | selected local ({selection[ds]["local"]["name"]}) | {m("accuracy"):.2f}% | {m("first_four"):.2f}% | {m("recent_four"):.2f}% | {m("newest"):.2f}% |')
 for name in sorted(set(r['name'] for r in full(q) if r['dataset']==ds)):
  rs=[r for r in full(q) if r['dataset']==ds and r['name']==name];m=lambda k:100*st.mean(r[k] for r in rs)
  lines.append(f'| {ds} | {name} | {m("accuracy"):.2f}% | {m("first_four"):.2f}% | {m("recent_four"):.2f}% | {m("newest"):.2f}% |')
lines+=['','## What is supported','',
'On Fashion MNIST, the original local allocation was a major bottleneck. Recruiting a less-used branch when existing branches recently fit poorly improves current learning without sacrificing the mean early-domain retention in these three paired runs. That is a real within-model improvement with the same stored state, training data and feedback. It is not a universal improvement: the exact fixed rule improves current MNIST accuracy at the cost of old accuracy and slightly lower overall accuracy.','',
'The ordinary slowing MLP removes a substantial part of the apparent special retention advantage. On Fashion, its stronger schedule has higher means than novelty recruitment for overall, first-four and newest accuracy. On MNIST, the stronger slowing schedule also outperforms novelty transfer on all three means. The broad claim that dendrites inherently retain better than ordinary networks is not supported by this comparison. Three seeds are insufficient for a broad significance claim.','',
'## Remaining uncertainty and next discriminating direction','',
'Usage-based slowing helps preserve old performance but restricts later adaptation. Local recruitment can improve that balance. The current experiment does not establish that branch structure, rather than plasticity allocation and fixed coded teaching, supplies a unique advantage. Teaching labels still provide a fixed desired response at every neuron; a biological source of that supervision is not demonstrated. Learned coactivity counts are explicit distributed memory, although examples and task-indexed configurations are not stored.','',
'The next useful comparison is a local flat learner with the same novelty/protection rule and teaching feedback, plus acquisition-focused alternatives to the fixed random class codes. A further claim should survive the ordinary slowing controls and the known inference-cost disadvantage. No conference-readiness or biological-realism claim is made.','',
'## Verification and reproduction','',
'Original variant exactly reproduces the prior branch training tensors on a small sequence. MSE and BCE local updates match their selected-unit autograd gradients. Slowing SGD follows the declared presentation-based schedule. All candidate choices and failed gates are archived.','',
'- `bdh_cl/scripts/local_graph_plasticity.py`, `results/local_graph_plasticity_20261013/`.','- `bdh_cl/scripts/local_graph_matched_slowing.py`, `results/local_graph_matched_slowing_20261013/`.','- `bdh_cl/scripts/local_graph_recruitment_transfer.py`, `results/local_graph_recruitment_transfer_20261013/`.','',
'Runtime: `/Users/harikrishna/Documents/lossfunk/cl-tokenformer/.venv/bin/python`, two CPU threads, cached real datasets. No external email or submission was sent.','']
(out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[lines.index('## Full-test confirmation at 32 domains'):lines.index('## Every additional confirmation, including unsuccessful changes')]))
