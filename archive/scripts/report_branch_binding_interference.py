"""Audit, profile and report the locked synthetic binding study."""
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
from bdh_cl.scripts.branch_binding_interference import Model, data, evaluate, verify

RAW=Path('results/branch_binding_interference_20261025')
REPORT=Path('reports/branch_binding_interference_20261025')
def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False))
def key(r): return r['architecture'],r['regulation'],r['variant'],r['joint']
def main():
 torch.set_num_threads(2)
 REPORT.mkdir(exist_ok=True,parents=True)
 rows=json.loads((RAW/'confirmation.json').read_text())
 assert json.loads((RAW/'completion.json').read_text())['complete']
 assert len(rows)==150
 groups={k:[r for r in rows if key(r)==k] for k in sorted({key(r) for r in rows})}
 for k,rr in groups.items():
  assert len(rr)==10 and {(r['seed'],r['order']) for r in rr}=={(s,o) for s in range(702,707) for o in ['forward','reverse']}
  assert all(r['steps']==384 for r in rr)
 audits=[]
 for r in rows:
  if r['joint']: continue
  path=RAW/'models'/f"{r['seed']}_{r['order']}_{r['architecture']}_{r['regulation']}_{r['variant']}.pt"
  m=Model(r['seed'],r['architecture'],r['regulation']!='none')
  m.load_state_dict(torch.load(path,weights_only=True))
  before={k:v.clone() for k,v in m.state_dict().items()}
  phases=data(r['seed'])
  if r['order']=='reverse': phases=phases[::-1]
  result=evaluate(m,[p[1] for p in phases])
  assert abs(st.mean(x['accuracy'] for x in result)-r['accuracy'])<1e-10
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
  audits.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'accuracy_matches':True,'evaluation_state_unchanged':True})
 audit={'passed':True,'confirmation_runs':len(rows),'saved_model_reproductions':len(audits),'checks':verify(),'checkpoints':audits}
 save(RAW/'audit.json',audit)

 # Timings use fresh cloned models, never alter reported checkpoints/results.
 profile=[]
 for arch,reg in [('branches','none'),('branches','local'),('branches','shared'),('mlp','none'),('mlp','local')]:
  r=next(r for r in rows if r['seed']==702 and r['order']=='forward' and r['architecture']==arch and r['regulation']==reg and r['variant']=='selected' and not r['joint'])
  path=RAW/'models'/f"702_forward_{arch}_{reg}_selected.pt"
  inference={}
  for n in [1,128]:
   m=Model(702,arch,reg!='none');m.load_state_dict(torch.load(path,weights_only=True))
   x=data(702)[0][0][0][:n]
   measurements=[]
   with torch.no_grad():
    for i in range(6):
     start=time.perf_counter()
     for _ in range(300): m(x)
     if i: measurements.append((time.perf_counter()-start)*1000/300)
   inference[n]=st.median(measurements)
  train_ms=[]
  for repeat in range(5):
   m=Model(702,arch,reg!='none');m.load_state_dict(torch.load(path,weights_only=True))
   opt=torch.optim.Adam(m.parameters(),lr=r['lr'],foreach=False)
   x,y=data(702)[0][0];x,y=x[:128],y[:128]
   start=time.perf_counter()
   for _ in range(300):
    opt.zero_grad(set_to_none=True);logits,a=m(x,True);F.cross_entropy(logits,y).backward()
    if reg!='none':
     scale=1/(1+r['strength']*m.exposure)
     if reg=='shared':scale=scale.mean().expand_as(scale)
     w=m.input.weight.detach().clone();b=m.input.bias.detach().clone()
    opt.step()
    with torch.no_grad():
     if reg!='none':
      m.input.weight.copy_(w+scale[:,None]*(m.input.weight-w));m.input.bias.copy_(b+scale*(m.input.bias-b));m.exposure.add_(a.detach().square().mean(0))
   train_ms.append((time.perf_counter()-start)*1000/300)
  profile.append({'architecture':arch,'regulation':reg,'inference_ms':inference,'train_step_ms':st.median(train_ms),'parameters':r['parameters'],'training_persistent_bytes':r['training_persistent_bytes'],'auxiliary_bytes':r['auxiliary_bytes'],'inference_bytes':r['inference_bytes']})
 save(RAW/'profile.json',{'threads':2,'batch':128,'train_repetitions':5,'train_steps_per_repetition':300,'inference_repetitions':5,'inference_calls_per_repetition':300,'notes':'Fresh checkpoint clones, CPU, training includes forward/backward/Adam/local regulation; no data generation or held-out evaluation timed. No latency claim outside this tiny task.','rows':profile})

 def mean(k,field):return st.mean(r[field] for r in groups[k])
 def paired(a,b,field='accuracy'):
  ds=[100*(st.mean(r[field] for r in groups[a] if r['seed']==s)-st.mean(r[field] for r in groups[b] if r['seed']==s)) for s in range(702,707)]
  mu=st.mean(ds);e=2.776445105*st.stdev(ds)/math.sqrt(5)
  return {'mean_pp':mu,'lower':mu-e,'upper':mu+e,'seed_differences_pp':ds}
 k=lambda a,g,v='selected':(a,g,v,False)
 contrasts={
  'primary_branch_local_minus_shared':paired(k('branches','local'),k('branches','shared')),
  'branch_local_minus_shuffled':paired(k('branches','local'),k('branches','shuffled')),
  'branch_local_minus_unregulated_equal_LR':paired(k('branches','local'),k('branches','none','local_setting_control')),
  'branch_local_minus_MLP_local':paired(k('branches','local'),k('mlp','local')),
  'MLP_local_minus_shared_equal_settings':paired(k('mlp','local'),k('mlp','shared','local_setting_control')),
 }
 save(RAW/'contrasts.json',contrasts)
 names={k('branches','local'):'Branches + local regulation',k('branches','none'):'Branches + Adam',k('branches','none','local_setting_control'):'Branches + Adam, local-matched LR',k('branches','shared'):'Branches + shared regulation',k('branches','shuffled'):'Branches + shuffled regulation',k('mlp','none'):'MLP + Adam',k('mlp','local'):'MLP + local regulation',k('mlp','shared'):'MLP + shared, selected settings',k('mlp','shared','local_setting_control'):'MLP + shared, local-matched settings',k('mlp','shuffled'):'MLP + shuffled, selected settings',k('mlp','shuffled','local_setting_control'):'MLP + shuffled, local-matched settings',k('linear_branches','none'):'Linear branches + Adam'}
 show=[k('branches','none'),k('branches','local'),k('branches','shared'),k('branches','shuffled'),k('mlp','none'),k('mlp','local')]
 fig,ax=plt.subplots(1,2,figsize=(12,4.8))
 for panel,field,title in [(0,'accuracy','Final accuracy on all combinations'),(1,'acquisition','Accuracy immediately after learning each phase')]:
  values=[100*mean(v,field) for v in show]
  ax[panel].bar(range(6),values,color=['#7897ad','#277e83','#82b0aa','#b8c9c2','#b38d68','#806a9c'])
  for i,v in enumerate(values):ax[panel].text(i,v+.6,f'{v:.2f}',ha='center',fontsize=9)
  ax[panel].set_xticks(range(6),['Branch\nAdam','Branch\nlocal','Branch\nshared','Branch\nshuffled','MLP\nAdam','MLP\nlocal'],fontsize=8)
  ax[panel].set_ylim(0,106);ax[panel].set_ylabel('Accuracy (%)');ax[panel].set_title(title,fontsize=11)
 fig.suptitle('Synthetic sequential feature binding: five seeds, both orders')
 fig.tight_layout();fig.savefig(REPORT/'binding_results.png',dpi=180);fig.savefig(REPORT/'binding_results.svg');plt.close(fig)
 lines=['# Does branch-local learning reduce interference? — 25 October 2026','',
 '**Result: this test did not establish a branch-specific advantage.** Activity-dependent regulation improved retention versus unrestricted Adam, but shared and shuffled regulation performed similarly. An equally sized ordinary MLP with local regulation was at least competitive. These are synthetic feature-binding results, not Split MNIST class-incremental accuracy.','',
 '## What was tested','',
 'Each example contains one of eight shapes and one of eight colours, encoded as two noisy one-hot vectors (16 inputs). Four successive phases introduce disjoint sets of 16 shape–colour conjunctions. Each phase has identical shape and colour marginals, balanced binary labels, and a single fixed label function throughout training. The target depends on the conjunction. There are 1,024 training observations and 1,024 independent noisy evaluation observations per phase. All 64 conjunctions are eventually learned. This tests retention of familiar combinations under observation noise, not generalization to unseen combinations.','',
 'The network receives only the current observation. There are no task IDs, task-boundary inputs, replay, stored input cues, external configurations, or per-phase optimizer/state resets. Evaluation labels are used only for metrics. The experiment scheduler presents four blocks; this is not a boundary-free real-world data stream.','',
 'The branched network has 16 groups of four ReLU subunits; each group sums its branch outputs before a shared binary classifier. The ordinary MLP has 59 ReLU units and a binary classifier. Parameter counts are 1,122 versus 1,123. A linear-branch control removes ReLU; its chance-level acquisition makes it a representational diagnostic, not a fair retention comparator.','',
 '## Learning rule','',
 r'For each branch (or MLP hidden unit), accumulate past mean-squared activity: $u_j\leftarrow u_j+\operatorname{mean}_{batch}a_j^2$. Before that write, scale its incoming Adam parameter update by $s_j=1/(1+\lambda u_j)$. The shared control applies the mean multiplier to every unit; the shuffled control randomly permutes the same current multiplier vector. The classifier is not regulated. No oracle or counterfactual branch errors are computed.','',
 'This is an engineered plasticity regulator applied to backpropagation/Adam. It is not a fully local biological learning rule. Its state is cumulative, not fading short-term memory. It is applied after Adam because scaling gradients before Adam can be cancelled by normalization. Shuffled vectors are regenerated each update, so their effect can resemble uniform slowing.','',
 '## Locked validation and confirmation','',
 'Development used seeds 700/701 with forward order, three learning rates and two regulation strengths. Every candidate was retained (90 runs). Each architecture/rule independently selected the harmonic mean of acquisition and final retention; no official confirmation measurements selected settings. Fresh seeds 702–706, both orders, produced 150 confirmation runs, including equal-local-setting controls and joint references. The equal-setting controls were added and locked before confirmation to address generic learning-rate slowing. Each sequential run uses 384 updates, 12 passes per current phase, batch 128. Joint references use the same total observations/updates but mix all phases and use the sequentially selected unregulated LR; they were not independently optimized.','',
 '| Model | Final all-phase accuracy | New-phase acquisition | Earlier-three final accuracy | Forgetting |','|---|---:|---:|---:|---:|']
 for kk,name in names.items():
  lines.append(f"| {name} | {100*mean(kk,'accuracy'):.2f}% | {100*mean(kk,'acquisition'):.2f}% | {100*mean(kk,'old_accuracy'):.2f}% | {100*mean(kk,'forgetting'):.2f} pp |")
 lines+=['','Acquisition is accuracy on a phase immediately after its training. Forgetting averages the drop from the maximum post-acquisition accuracy to final accuracy over the earlier three phases. Both nonlinear architectures reach 100% joint-reference accuracy; linear joint accuracy is approximately chance.','',
 '![Results](binding_results.png)','',
 '## Paired evidence','',
 'Intervals below are descriptive, unadjusted 95% t intervals over five seed-level differences, averaging the two orders within each seed. Primary endpoint is local versus shared regulation within the branched model. Other contrasts are supporting; multiple comparisons and a small seed count limit claims.','',
 '| Contrast in final accuracy | Difference | 95% interval |','|---|---:|---:|']
 for title,d in contrasts.items():lines.append(f"| {title.replace('_',' ')} | {d['mean_pp']:+.2f} pp | [{d['lower']:+.2f}, {d['upper']:+.2f}] |")
 lines+=['',
 'New-phase acquisition differs by less than two percentage points for all nonlinear controls. The primary comparison has a near-zero point estimate; its interval includes zero. This does not prove exact equivalence, but gives no support for a meaningful benefit of aligning this regulator with individual branch histories. The same generic regulation can help an ordinary MLP. The positive comparison with unregulated Adam therefore does not establish a dendritic mechanism.','',
 '## Mechanism and implementation checks','',
 'The branch computation is exactly representable as an ordinary ReLU layer whose readout columns are tied within groups. Outputs and parameter gradients agree numerically. This equivalence is expected: useful architectural inductive biases can be represented as ordinary computation graphs. Here it prevents treating the word “branch” as an additional mechanism. The unconstrained MLP comparison additionally changes hidden width and readout tying, so it is a practical near-budget comparison rather than a perfect single-factor ablation.','',
 'All 120 saved sequential checkpoints reproduce recorded final accuracy exactly under reevaluation; evaluation leaves model/activity state unchanged. Phase marginals and labels are balanced; the saved-state audit and source hashes are archived. First-update held-out old-loss rise and new-loss reduction were recorded for every transition in the raw results, without using them as feedback or hyperparameter selection. A single Adam update is not a full adaptation-cost curve, so these probes are secondary diagnostics.','',
 '## Resources and timing','',
 'Activity state is training-only: 256 bytes for branches or 236 for the ordinary MLP. Deployment uses only weights/biases: 4,488 versus 4,492 bytes. Training-persistent counts below include FP32 weights, actual Adam moments/scalar steps, and activity state; they exclude transient gradients/activations, allocator/Python overhead and data. Thus inference storage is matched closely, but regulated arms are not exactly matched for training state.','',
 '| Model | Training persistent state | Inference state | Train step, batch 128 | Inference, batch 128 |','|---|---:|---:|---:|---:|']
 for p in profile:lines.append(f"| {p['architecture']} / {p['regulation']} | {p['training_persistent_bytes']:,} B | {p['inference_bytes']:,} B | {p['train_step_ms']:.4f} ms | {p['inference_ms'][128]:.4f} ms |")
 lines+=['','Fresh CPU profiles use two threads and repeated checkpoint clones. No large retrieval bank or iterative communication is present. Times are for a tiny model; they do not establish deep-network efficiency, energy use or GPU performance. All compared runs use equal presentations/update counts, not exactly equal elapsed compute. The local rule adds training work; no branch advantage was established even with that allowance.','',
 '## What this means and the next decision','',
 '**Supported:** nonlinear hidden computation solves the binding task; modest activity-dependent plasticity regulation can improve retention without blocking new learning in this toy setting.','',
 '**Not supported:** branch-specific plasticity is the cause of that improvement, branches beat a matched ordinary network, the grant’s short-term-memory mechanism is validated, or the method solves general task-ID-free class-incremental learning.','',
 'This negative mechanism result should narrow the hypothesis, not trigger a search for a favourable seed. Do not expand this regulator merely because it beats unrestricted Adam. A subsequent project would need a genuinely different branch mechanism (for example, input-specific local competition/plasticity), explicit predictions that distinguish it from shared slowing and ordinary sparse gating, and a predeclared harder transfer test. A larger benchmark alone will not fix missing mechanism attribution.','',
 '## Reproduce and inspect','',
 '- [Experiment driver](../../bdh_cl/scripts/branch_binding_interference.py), [report/audit/profile driver](../../bdh_cl/scripts/report_branch_binding_interference.py).',
 '- [Protocol](../../results/branch_binding_interference_20261025/protocol.json), [pre-confirmation addendum](../../results/branch_binding_interference_20261025/confirmation_addendum.json), [locked selection](../../results/branch_binding_interference_20261025/selection_lock.json).',
 '- [All validation runs](../../results/branch_binding_interference_20261025/development.json), [all confirmation runs and trajectories](../../results/branch_binding_interference_20261025/confirmation.json), [paired contrasts](../../results/branch_binding_interference_20261025/contrasts.json).',
 '- [Checkpoint audit](../../results/branch_binding_interference_20261025/audit.json), [CPU profile](../../results/branch_binding_interference_20261025/profile.json), [source hashes](../../results/branch_binding_interference_20261025/source_hashes.json).',
 '', 'Commands (from repository root, using the project Python environment):','',
 '```sh','python -m bdh_cl.scripts.branch_binding_interference development','python -m bdh_cl.scripts.branch_binding_interference confirmation','python -m bdh_cl.scripts.report_branch_binding_interference','```','',
 'Training drivers refuse to overwrite existing results. Archive/change the output directory for a deliberate reproduction; do not delete the recorded evidence.']
 (REPORT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'contrasts':contrasts,'audit_passed':True,'profile':profile},indent=2),flush=True)

if __name__=='__main__': main()
