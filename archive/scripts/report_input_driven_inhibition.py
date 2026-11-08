"""Report all locked inhibition arms, including acquisition failures."""
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bdh_cl.scripts import input_driven_branch_inhibition as e

RAW=e.ROOT;OUT=Path('reports/input_driven_inhibition_20261026')
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def entropy(p):
 p=p.clamp(1e-7,1-1e-7);return -(p*p.log2()+(1-p)*(1-p).log2())
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True,parents=True)
 rows=json.loads((RAW/'confirmation.json').read_text())
 assert json.loads((RAW/'completion.json').read_text())['complete'] and len(rows)==120
 settings=json.loads((RAW/'selection_lock.json').read_text())
 assert hashlib.sha256((RAW/'selection.json').read_bytes()).hexdigest()==settings['sha256']
 (_, _),(tx,ty)=e.dataset()
 ids=torch.cat([torch.nonzero(ty==c).flatten()[:100] for c in range(10)])
 sx,sy=tx[ids],ty[ids];audits=[];diagnostics=[]
 for row in rows:
  path=RAW/'models'/f"{row['seed']}_{row['order']}_{row['arm']}.pt"
  model=e.Model(row['seed'],row['arm']);model.load_state_dict(torch.load(path,weights_only=True))
  before={k:v.clone() for k,v in model.state_dict().items()}
  result=e.evaluate(model,tx,ty)
  assert result['accuracy']==row['accuracy']
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
  audits.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'accuracy_reproduced':True})
  if row['joint']:continue
  with torch.no_grad():
   _,activity,inhibition,competition=model(sx,True)
   active=(activity>0).float()
   p=torch.stack([active[sy//2==task].mean(0) for task in range(5)])
   mi=entropy(p.mean(0))-entropy(p).mean(0)
   recruited=p>.1
   diagnostic={'seed':row['seed'],'order':row['order'],'arm':row['arm'],
    'mean_binary_activity_task_MI_bits':float(mi.mean()),
    'fraction_units_recruited_all5_pairs':float(recruited.all(0).float().mean()),
    'fraction_units_recruited_exactly1_pair':float((recruited.sum(0)==1).float().mean()),
    'fraction_never_active_on_probe':float((active.sum(0)==0).float().mean()),
    'mean_inhibition_input_std':float(inhibition.std(0).mean()),
    'latest_pair_below75percent':row['new_accuracy']<.75}
   if row['arm'].startswith('branch'):
    pre=(activity.reshape(-1,25,4)*model.gain).sum(-1)+model.soma_bias
    h=pre if row['arm']=='branch_linear_soma' else pre.tanh()
    latest=sy//2==(4 if row['order']=='forward' else 0)
    diagnostic.update({'mean_soma_feature_std_all':float(h.std(0).mean()),
     'mean_soma_feature_std_latest':float(h[latest].std(0).mean()),
     'soma_abs_above98percent':None if row['arm']=='branch_linear_soma' else float((h.abs()>.98).float().mean()),
     'fraction_examples_neurons_multiple_active_branches':float((active.reshape(-1,25,4).sum(-1)>=2).float().mean())})
   diagnostics.append(diagnostic)
 save(RAW/'audit.json',{'passed':True,'all120_checkpoints_reproduced':True,'evaluation_immutable':True,'checks':e.verify(),'models':audits})
 save(RAW/'activity_diagnostics.json',{'scope':'Post-hoc descriptive probe, not a tuned success criterion.100testimages/class; tasklabels used only for analysis. Recruited means active on>10%observations of a pair; threshold arbitrary, no search. Low mean single-unit MI does not rule out distributed task coding.','rows':diagnostics})
 seq={a:[r for r in rows if r['arm']==a and not r['joint']] for a in e.ARMS}
 joint={a:[r for r in rows if r['arm']==a and r['joint']] for a in e.ARMS}
 for a in e.ARMS:
  assert len(seq[a])==10 and len(joint[a])==5
  assert {(r['seed'],r['order']) for r in seq[a]}=={(s,o) for s in range(852,857) for o in ['forward','reverse']}
  assert all(r['presentations']==120000 for r in seq[a]+joint[a])
 def avg(a,field):return st.mean(r[field] for r in seq[a])
 def paired(a,b,field='accuracy'):
  ds=[100*(st.mean(r[field] for r in seq[a] if r['seed']==seed)-st.mean(r[field] for r in seq[b] if r['seed']==seed)) for seed in range(852,857)]
  mu=st.mean(ds);err=2.776445105*st.stdev(ds)/math.sqrt(5)
  return {'mean_pp':mu,'lower':mu-err,'upper':mu+err,'seed_differences_pp':ds}
 contrasts={f'full_minus_{a}':paired('branch_full',a) for a in ['mlp','flat_group_gated','flat_global_gated','branch_no_context','branch_no_competition','branch_linear_soma']}
 save(RAW/'contrasts.json',contrasts)
 summary={a:{'sequential_accuracy':avg(a,'accuracy'),'acquisition':avg(a,'acquisition'),
             'old_accuracy':avg(a,'old_accuracy'),'new_accuracy':avg(a,'new_accuracy'),
             'joint_accuracy':st.mean(r['accuracy'] for r in joint[a]),
             'latest_pair_failures':sum(r['new_accuracy']<.75 for r in seq[a]),
             'active_fraction':avg(a,'active_fraction'),'parameters':seq[a][0]['parameters']} for a in e.ARMS}
 save(RAW/'summary.json',summary)

 profile=[]
 for arm in ['branch_full','flat_group_gated','mlp','dense2']:
  r=next(r for r in seq[arm] if r['seed']==852 and r['order']=='forward')
  state=torch.load(RAW/'models'/f'852_forward_{arm}.pt',weights_only=True)
  m=e.Model(852,arm);m.load_state_dict(state)
  infer={}
  with torch.no_grad():
   for batch in [1,128]:
    values=[]
    for repeat in range(6):
     start=time.perf_counter()
     for _ in range(200):m(tx[:batch])
     if repeat:values.append((time.perf_counter()-start)*1000/200)
    infer[batch]=st.median(values)
  train=[]
  for repeat in range(5):
   m=e.Model(852,arm);m.load_state_dict(state);opt=torch.optim.Adam(m.parameters(),lr=r['lr'],foreach=False)
   start=time.perf_counter()
   for _ in range(200):
    opt.zero_grad(set_to_none=True);log,a,_,_=m(tx[:128],True)
    (F.cross_entropy(log,ty[:128])+r['sparsity']*a.mean()).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step()
   train.append((time.perf_counter()-start)*1000/200)
  profile.append({'arm':arm,'inference_ms':infer,'training_step_ms_batch128':st.median(train),
                  'inference_bytes':r['inference_bytes'],'training_persistent_bytes':r['training_persistent_bytes']})
 save(RAW/'profile.json',{'threads':2,'inference':'Five measured repetitions of200calls afterwarmup, CPU, fullimage-to-logits, includespooling/inhibition.','training':'Five freshcheckpoint-clone repetitions of200Adamsteps, batch128, includesforward/backward/clipping/optimizer. Timingsareimplementation-dependent,noGPU/energyclaim.','rows':profile})
 save(RAW/'source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'source.py',RAW/'protocol.json',RAW/'selection_lock.json',RAW/'confirmation.json',Path(__file__)]})

 fig,ax=plt.subplots(figsize=(12,5))
 labels=['Full\nbranches','No input\ninhibition','No lateral\ncompetition','Linear\nsoma','Flat grouped\ngated','Flat global\ngated','MLP','Dense\n2-layer'];xx=np.arange(8)
 ax.bar(xx-.25,[100*summary[a]['sequential_accuracy'] for a in e.ARMS],width=.25,label='Final sequential accuracy',color='#32838a')
 ax.bar(xx,[100*summary[a]['acquisition'] for a in e.ARMS],width=.25,label='New-pair acquisition',color='#a7c3bd')
 ax.bar(xx+.25,[100*summary[a]['joint_accuracy'] for a in e.ARMS],width=.25,label='Joint training reference',color='#ac845e')
 ax.set_xticks(xx,labels,fontsize=8);ax.set_ylim(0,106);ax.set_ylabel('Accuracy (%)');ax.set_title('Input-driven branch inhibition with global Adam: five seeds, both orders');ax.legend(fontsize=8,loc='lower right')
 fig.tight_layout();fig.savefig(OUT/'inhibition_results.png',dpi=180);fig.savefig(OUT/'inhibition_results.svg');plt.close(fig)
 lines=['# Input-driven branches with global Adam — 26 October 2026','',
 '**Global Adam can fit the branch architecture, but this implementation does not solve continual retention.** The full model reaches 96.30% under joint training, yet only 16.26% after sequential training. Unlike the ordinary controls, it also has substantial current-task acquisition failures in some fresh runs. Thus this is not purely a routing/forgetting result.','',
 '## Architecture and learning','',
 r'For each branch, $z_b=W_bx-\theta_b$, $I_b(x)=\operatorname{softplus}(V_bq(x)+c_b)$, and $r_b=\operatorname{ReLU}(z_b-I_b(x)-0.5\operatorname{mean}_{c\ne b}\operatorname{ReLU}(z_c))$. The mean is over the other three branches of the same neuron. The soma computes $h_n=\tanh(\sum_b a_{nb}r_{nb}+d_n)$, then a shared ten-class classifier reads the 25 soma outputs. The fixed input description $q$ is a normalized 4×4 pooled image.','',
 'Branches contribute directly according to their responses. Multiple branches can be active; there is no winner chosen by a usefulness state, no route bank, cue memory, separate retriever, recurrence or configuration search. Thresholds, gains, inhibition connections, soma biases and classifier weights all learn together via ordinary global Adam/backpropagation. There is no special plasticity-protection mechanism and no short-term memory state. This is a trainable architectural test, not a biologically local learning-rule claim.','',
 '## Matched controls','',
 '- **No context:** replace learned input-dependent inhibition with its fixed initial value, retaining lateral competition; 1,700 fewer weights than full branches.',
 '- **No competition:** remove lateral competition while retaining learned inhibition; same parameters.',
 '- **Linear soma:** remove the soma tanh; same parameters. This can be flattened into a gated ReLU layer with a constrained classifier, and that equivalence is verified.',
 '- **Flat grouped gated:** use the same 100 input-driven units and four-unit competition groups, but give every unit its own classifier connection instead of nonlinear soma integration.',
 '- **Flat global gated:** the flat control with competition over all 100 units.',
 '- **MLP / dense two-layer:** ordinary 101-ReLU-unit MLP or 91-ReLU→91-tanh model, within 0.4% of the full branch parameter budget.',
 '',
 'All arms received the same candidate sparsity penalty on mean first-layer activity. The main flat controls are within 1% of full-branch parameters. This is near-matched parameter/storage access, not exact equality of functions, FLOPs or CPU time.','',
 '## Validation and confirmation','',
 'Seeds 850/851 used a stratified 50,000-image fit/10,000-image training-validation split, forward task order. Learning rate .001/.003 and activity penalty 0/.01 were selected independently per architecture and separately for sequential versus joint training. All 128 development fits are retained. Sequential selection maximized the harmonic mean of acquisition and final accuracy; joint selection maximized accuracy. No official-test metrics selected settings.','',
 'Fresh seeds 852–856 used full 60,000/10,000 train/test splits. Each selected sequential arm ran both task orders (80 fits); each joint reference ran once per seed (40 fits). All used two passes over each current pair, batch 128, gradient clip 5, ordinary Adam, no resets, no old-task replay, no task IDs or task-boundary inputs, no output masking. Current training labels supervise CE; evaluation labels only produce metrics. The scheduler presents task blocks; task boundaries are not model inputs or triggers for state/optimizer changes.','',
 '## Results','',
 '| Model | Parameters | Final all-ten accuracy | New-pair acquisition | Earlier four final accuracy | Latest pair final accuracy | Joint reference |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in e.ARMS:
  s=summary[arm];lines.append(f"| {arm.replace('_',' ')} | {s['parameters']:,} | {100*s['sequential_accuracy']:.2f}% | {100*s['acquisition']:.2f}% | {100*s['old_accuracy']:.2f}% | {100*s['new_accuracy']:.2f}% | {100*s['joint_accuracy']:.2f}% |")
 lines+=['','Final all-ten accuracy is micro-averaged over the test set. Acquisition averages per-digit balanced current-pair accuracy immediately after its training; earlier/latest accuracies are also balanced within pairs. All ten logits remain eligible for these measurements. Joint references mix all ten digits with the same total presentations, so they are not continual-learning results.','',
 '![Results](inhibition_results.png)','',
 'The full model’s average acquisition is more than two points below the ordinary MLP, crossing the predeclared interpretability threshold. Its lower final accuracy therefore combines acquisition failure and failure to retain earlier performance. Even flat controls with approximately 99% acquisition retain essentially no earlier-pair accuracy. There is no branch-specific retention advantage.','',
 '## Paired final-accuracy differences','',
 'Descriptive unadjusted 95% t intervals over five seed-level differences, averaging both orders within seed. Small seed count and multiple comparisons limit inference; acquisition differences prevent a pure forgetting interpretation.','',
 '| Full branch model minus control | Difference | 95% interval |','|---|---:|---:|']
 for name,d in contrasts.items():lines.append(f"| {name.replace('_',' ')} | {d['mean_pp']:+.2f} pp | [{d['lower']:+.2f}, {d['upper']:+.2f}] |")
 lines+=['','## Activity and optimization diagnostics','',
 'The following are post-hoc descriptions on 100 test images per digit, not tuned success criteria. A unit is counted as recruited by a pair if active on more than 10% of its probe images; that threshold is arbitrary and was not searched. Single-unit activity/task mutual information is descriptive and can miss information in distributed population codes.','',
 '| Model | Active response fraction | Units recruited by all five pairs | Units recruited by exactly one pair | Latest-pair failures below 75% |','|---|---:|---:|---:|---:|']
 for arm in e.ARMS:
  dd=[d for d in diagnostics if d['arm']==arm]
  lines.append(f"| {arm.replace('_',' ')} | {100*summary[arm]['active_fraction']:.2f}% | {100*st.mean(d['fraction_units_recruited_all5_pairs'] for d in dd):.2f}% | {100*st.mean(d['fraction_units_recruited_exactly1_pair'] for d in dd):.2f}% | {summary[arm]['latest_pair_failures']}/10 |")
 lines+=['','Input-driven activity is not the same thing as stable task separation. Global gradients still change shared input/inhibition/classifier parameters; Adam moments can also continue parameter motion after an activity change. The soma/activation diagnostics are saved to help investigate instability, but saturation, dead units and classifier drift have not been causally isolated by this experiment. Removing the soma nonlinearity improves acquisition reliability in this sample, while not yielding retention. A normalized or otherwise stabilized soma would require a new validation/confirmation study, not retrospective tuning of these test runs.','',
 '## Memory and measured compute','',
 'No auxiliary inference memory is stored beyond learned weights/biases. Full branches use 322,340 bytes of FP32 parameters versus 321,220 for the MLP; the extra inhibition parameters are already included in that budget. Training-state counts include weights, Adam moments and optimizer scalar steps, excluding gradients, activations, allocator/Python overhead and data.','',
 '| Model | Inference state | Training persistent state | Inference batch 1 | Inference batch 128 | Training step batch 128 |','|---|---:|---:|---:|---:|---:|']
 for p in profile:lines.append(f"| {p['arm'].replace('_',' ')} | {p['inference_bytes']:,} B | {p['training_persistent_bytes']:,} B | {p['inference_ms'][1]:.4f} ms | {p['inference_ms'][128]:.4f} ms | {p['training_step_ms_batch128']:.4f} ms |")
 lines+=['','Profiles use two-thread CPU, repeated fresh checkpoint clones, full image-to-logits inference and forward/backward/clipping/Adam training. These are unoptimized small-model timings, not GPU/energy estimates. Equal parameter count does not imply equal latency, and dense tensor execution still computes inactive units; biological sparsity does not automatically save arithmetic here.','',
 '## Answer and limits','',
 '**Yes:** a global differentiable optimizer can train input-driven inhibition and branch integration without an explicit routing controller. Gradient checks and joint accuracy establish that this implementation is trainable.','',
 '**No demonstrated continual-learning benefit:** this choice of architecture/loss is not sufficient to preserve old skills. Some sequential runs also fail acquisition. Biological compartmentalization was not fully modeled, no local plasticity mechanism or fading state was included, and a specific failure of this unnormalized soma does not refute all dendritic architectures or all global optimizers.','',
 'The next scientific question should distinguish optimization stability from preservation: first stabilize new-task learning on validation data, then test whether any local write/protection mechanism improves retention beyond an ordinary equally sparse/gated control. Adding another external retriever would not answer this architectural question. No oracle recovery is reported here because no configuration bank exists in this model.','',
 '## Verification and evidence','',
 'All 120 saved checkpoints reproduce recorded accuracy exactly; inference leaves state unchanged. Checks verify nonzero finite gradients through inhibition/input/gains, the soma formula, independence from inference batch composition and the linear-soma equivalence. Source and selection hashes are preserved.','',
 '- [Experiment code](../../bdh_cl/scripts/input_driven_branch_inhibition.py), [audit/report/profile code](../../bdh_cl/scripts/report_input_driven_inhibition.py).',
 '- [Protocol](../../results/input_driven_inhibition_20261026/protocol.json), [all validation fits](../../results/input_driven_inhibition_20261026/development.json), [locked settings](../../results/input_driven_inhibition_20261026/selection_lock.json), [all confirmation trajectories](../../results/input_driven_inhibition_20261026/confirmation.json).',
 '- [Summary](../../results/input_driven_inhibition_20261026/summary.json), [contrasts](../../results/input_driven_inhibition_20261026/contrasts.json), [activity diagnostics](../../results/input_driven_inhibition_20261026/activity_diagnostics.json), [profiles](../../results/input_driven_inhibition_20261026/profile.json), [audit](../../results/input_driven_inhibition_20261026/audit.json).','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({'summary':summary,'contrasts':contrasts,'profile':profile,'audit_passed':True},indent=2),flush=True)

if __name__=='__main__':main()
