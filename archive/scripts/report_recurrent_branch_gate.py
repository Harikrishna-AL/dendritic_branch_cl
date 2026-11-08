"""Audit and report the prospective recurrent gate experiment, including failures."""
import hashlib
import json
import math
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bdh_cl.scripts import recurrent_branch_gate_test as e

OUT=Path('reports/recurrent_branch_gate_20261026')
RAW=e.ROOT
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True,parents=True)
 rows=json.loads((RAW/'confirmation.json').read_text())
 assert json.loads((RAW/'completion.json').read_text())['complete'] and len(rows)==10
 assert {(r['seed'],r['order']) for r in rows}=={(s,o) for s in range(812,817) for o in ['forward','reverse']}
 settings=json.loads((RAW/'selection_lock.json').read_text())['settings']
 raw=e.load_raw('data_cache/mnist');x,y=e.tensors(raw,'test')
 audit=[]
 for record in rows:
  assert record['audit']['state_immutable'] and record['audit']['oracle_bank_used_only_after_automatic_predictions']
  assert record['timing']['backbone_presentations']==120000
  path=RAW/f"{record['order']}_{record['seed']}.pt"
  checkpoint=torch.load(path,weights_only=True)
  m=e.b.make(record['seed'],'branch1_proxy');m.load_state_dict(checkpoint['backbone'])
  for i,s in enumerate(settings):
   router=e.Updater(record['seed']+10000);router.load_state_dict(checkpoint['routers'][i]);before=e.digest(m,[router])
   pred=e.predictions(m,router,x,s,s['steps'],'final')
   acc=float((pred==y).float().mean())
   expected=next(r for r in record['methods'] if r['family']==s['family'] and r['initial']=='final' and r['inference_steps']==s['steps'] and not r['hard'])
   assert acc==expected['accuracy'] and e.digest(m,[router])==before
  audit.append({'seed':record['seed'],'order':record['order'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'all_five_primary_predictions_reproduced':True})

 # Independently train one backbone without any sidecars or sidecar RNG calls.
 train_x,train_y=e.tensors(raw,'train');seed=810
 generator=torch.Generator().manual_seed(seed)
 valid=torch.cat([torch.nonzero(train_y==c).flatten()[torch.randperm(int((train_y==c).sum()),generator=generator)[:1000]] for c in range(10)])
 mask=torch.ones(len(train_y),dtype=torch.bool);mask[valid]=False
 bx,by=train_x[mask],train_y[mask]
 m=e.b.make(seed,'branch1_proxy');opt=torch.optim.Adam(m.parameters(),lr=.003)
 for stage,pair in enumerate(e.SPLITS):
  ids=torch.nonzero(torch.isin(by,torch.tensor(pair))).flatten()
  for epoch in range(2):
   g=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=g)].split(128):e.b.step(m,opt,bx[ix],by[ix],'branch1_proxy')
 expected=torch.load(RAW/'dev_810.pt',weights_only=True)['backbone']
 assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items())
 save(RAW/'audit.json',{'passed':True,'runs':audit,'primary_saved_readouts':50,'no_sidecar_backbone_bitwise_matches_development810':True,'inference_uses_no_target_or_task_or_bank_argument':True})

 def selected(record,family,initial='final',steps=None):
  s=next(s for s in settings if s['family']==family)
  steps=s['steps'] if steps is None else steps
  return next(r for r in record['methods'] if r['family']==family and r['initial']==initial and r['inference_steps']==steps and not r['hard'])
 def average(family,field='accuracy',initial='final',steps=None):return st.mean(selected(r,family,initial,steps)[field] for r in rows)
 def paired(a,b,initial='final',asteps=None,bsteps=None):
  diffs=[100*st.mean(selected(r,a,initial,asteps)['accuracy']-selected(r,b,initial,bsteps)['accuracy'] for r in rows if r['seed']==s) for s in range(812,817)]
  mean=st.mean(diffs);err=2.776445105*st.stdev(diffs)/math.sqrt(5)
  return {'mean_pp':mean,'lower':mean-err,'upper':mean+err,'seed_differences_pp':diffs}
 contrasts={
  'recurrent3_minus_one_pass':paired('recurrent_full','one_pass'),
  'recurrent3_minus_one_pass_extra':paired('recurrent_full','one_pass_extra'),
  'recurrent3_minus_no_neighbor':paired('recurrent_full','recurrent_no_neighbor'),
  'same_recurrent_weights_3_minus_1':paired('recurrent_full','recurrent_full',asteps=3,bsteps=1),
  'same_recurrent_weights_5_minus_3':paired('recurrent_full','recurrent_full',asteps=5,bsteps=3)}
 save(RAW/'contrasts.json',contrasts)
 curves={}
 for family in e.FAMILIES:
  for initial in ['final','uniform']:
   acquisition=[];old=[]
   for record in rows:
    cc=record['timing']['diagnostic_probe_curves']
    acq=[next(m for m in c['methods'] if m['family']==family and m['initial']==initial)['pair_accuracy'][i] for i,c in enumerate(cc)]
    last=next(m for m in cc[-1]['methods'] if m['family']==family and m['initial']==initial)['pair_accuracy']
    acquisition.append(st.mean(acq));old.append(st.mean(last[:4]))
   curves[f'{family}_{initial}']={'mean_current_pair_acquisition':st.mean(acquisition),'final_old_four_pairs_probe':st.mean(old)}
 save(RAW/'probe_summary.json',curves)

 # End-to-end inference, including image feature extraction and all branches.
 cp=torch.load(RAW/'forward_812.pt',weights_only=True)
 m=e.b.make(812,'branch1_proxy');m.load_state_dict(cp['backbone'])
 profile=[]
 with torch.no_grad():
  for family in ['one_pass','one_pass_extra','recurrent_full','recurrent_local']:
   index=next(i for i,s in enumerate(settings) if s['family']==family);s=settings[index]
   r=e.Updater(10812);r.load_state_dict(cp['routers'][index])
   for steps in ([1] if s['steps']==1 else [1,3,5]):
    ms={}
    for batch in [1,128]:
     measurements=[]
     for repeat in range(6):
      start=time.perf_counter()
      for _ in range(30):e.predictions(m,r,x[:batch],s,steps,'final')
      if repeat:measurements.append((time.perf_counter()-start)*1000/30)
     ms[batch]=st.median(measurements)
    profile.append({'family':family,'steps':steps,'inference_ms':ms})
 save(RAW/'profile.json',{'threads':2,'profile':'Two-thread CPU, end-to-end prediction, including pooling/key, all100branch responses, recurrentupdates and head; five measured repetitions of30calls afterwarmup. No GPU/energy claim.','rows':profile})
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'source.py',RAW/'confirmation_source.py',RAW/'selection_lock.json',RAW/'protocol.json',RAW/'confirmation_addendum.json',Path(e.b.__file__),Path(e.old.__file__),Path(__file__)]}
 save(RAW/'source_hashes.json',hashes)
 summary={f:{'final':average(f),'uniform':average(f,initial='uniform'),'old':average(f,'old_accuracy'),'new':average(f,'new_accuracy')} for f in e.FAMILIES}
 save(RAW/'summary.json',{'families':summary,'contrasts':contrasts,'baselines':{key:st.mean(r['baselines'][key]['accuracy'] for r in rows) for key in rows[0]['baselines']}})

 fig,ax=plt.subplots(1,2,figsize=(12,4.8))
 labels=['One pass','One pass\nextra training','Recurrent\nfull','Recurrent\nno neighbor','Recurrent\nlocal']
 indices=np.arange(5)
 ax[0].bar(indices-.18,[100*average(f) for f in e.FAMILIES],width=.36,label='Final-route initialization',color='#2f8086')
 ax[0].bar(indices+.18,[100*average(f,initial='uniform') for f in e.FAMILIES],width=.36,label='Uniform initialization',color='#a2c0ba')
 ax[0].set_xticks(indices,labels,fontsize=8);ax[0].set_ylim(0,100);ax[0].set_ylabel('All-ten-class accuracy (%)');ax[0].set_title('Automatic retrieval after sequential training');ax[0].legend(fontsize=8)
 for initial,color in [('final','#2f8086'),('uniform','#ad835a')]:
  vals=[100*average('recurrent_full',initial=initial,steps=t) for t in [1,3,5]]
  ax[1].plot([1,3,5],vals,'o-',color=color,label=initial+' initialization')
 ax[1].set_ylim(0,100);ax[1].set_xticks([1,3,5]);ax[1].set_xlabel('Gate update rounds, same recurrent weights');ax[1].set_ylabel('All-ten-class accuracy (%)');ax[1].set_title('Does more recurrence recover older skills?');ax[1].legend(fontsize=8)
 fig.suptitle('Prospective recurrent branch routing: five fresh seeds, both orders')
 fig.tight_layout();fig.savefig(OUT/'recurrent_gates.png',dpi=180);fig.savefig(OUT/'recurrent_gates.svg');plt.close(fig)

 lines=['# Recurrent branch-gate retrieval — 26 October 2026','',
 '**The tested recurrent updater did not recover earlier skills.** This is a negative result for this small, prospectively trained updater, not proof that recurrent retrieval is impossible. Labels supervised current-example training only; inference used input/network signals and no task IDs, classification-error feedback or route banks.','',
 '## Experiment','',
 'Split MNIST introduces five digit pairs sequentially, two passes per current pair. The existing one-layer 25-neuron, four-branch tanh backbone learns by factual cross-entropy, Adam and gradient-estimated branch usefulness. Updaters learn alongside it from the same current stream, using the first 32 examples of each backbone minibatch of up to 128. Their branch responses and classifier parameters are detached before updater training; they never affect backbone gradients or routing. Each updater sees approximately one quarter as many current-example presentations as the backbone, before extra optimizer repetitions; all updater controls have identical access. No old-task examples are revisited after moving on.','',
 'A shared 856-parameter rule receives each neuron’s four branch responses, four current gate probabilities, a 16-dimensional fixed image description, a learned four-dimensional neuron identity, optional neighboring gates and optional global class probabilities. It makes bounded residual updates to four gate logits. Neighbor communication uses a fixed ring. The recurrent rule is trained through three rounds; the one-pass rule has identical parameters. A one-pass extra-training control performs three optimizer updates on the current batch, matching recurrent-cell call counts approximately rather than exact FLOPs.','',
 'Training initializes each example with a random choice between current-route and uniform logits, plus small noise. Inference tests final-route and uniform initialization without noise. All weights are frozen at inference; only per-image gate activations change. The default classifier uses a soft branch mixture, with hard projection retained as a secondary readout. The original trained backbone uses hard gates, so deployment with mixtures is a functional change and a possible limitation.','',
 'Two training-validation seeds (810/811) selected per-family learning rates from .001/.003 at the declared trained horizon and final-route initialization. Five fresh seeds (812–816), both orders, confirmed locked settings on the official test split. All predefined 1/3/5-round, initialization and hard-projection conditions are retained. Learning curves on a fixed test subset were added before confirmation as diagnostics only; they did not affect weights or settings.','',
 '## Automatic results','',
 '| Updater | Final-route initialization | Uniform initialization | Earlier four pairs, final init | Latest pair, final init |','|---|---:|---:|---:|---:|']
 for f in e.FAMILIES:
  s=summary[f];lines.append(f"| {f.replace('_',' ')} | {100*s['final']:.2f}% | {100*s['uniform']:.2f}% | {100*s['old']:.2f}% | {100*s['new']:.2f}% |")
 lines+=['','The table uses each updater’s trained horizon (one or three rounds). Both initialization conditions use the same trained weights and learning rate; uniform was not independently optimized. All ten output classes remain eligible.','',
 '| Matched-backbone reference | All-ten-class accuracy |','|---|---:|']
 for name in rows[0]['baselines']:
  value=100*st.mean(r['baselines'][name]['accuracy'] for r in rows)
  label=name.replace('_',' ')+(' — label-assisted diagnostic' if name=='historical_oracle' else '')
  lines.append(f'| {label} | {value:.2f}% |')
 lines+=['','The oracle selects historical hard configurations using the true digit pair. It is computed only after automatic predictions and is not autonomous continual learning. Local-average routing has 26,000 bytes of input-association memory and sees all current backbone examples. Direct ridge classifiers use current labels and no neural backbone at inference; these are stronger memory-based references, not memory-matched recurrent controls.','',
 '![Recurrent gate results](recurrent_gates.png)','',
 '## Paired comparisons','',
 'Unadjusted 95% t intervals over five seed differences, first averaging both orders within seed; descriptive supporting contrasts are not multiplicity-corrected. Independently validation-selected learning rates differ across some families. Same-weights iteration curves hold that factor fixed.','',
 '| Final-initialization contrast | Difference | 95% interval |','|---|---:|---:|']
 for name,d in contrasts.items():lines.append(f"| {name.replace('_',' ')} | {d['mean_pp']:+.2f} pp | [{d['lower']:+.2f}, {d['upper']:+.2f}] |")
 lines+=['','## Acquisition and retention diagnostics','',
 '| Updater, initialization | Mean current-pair accuracy immediately after its training | Final earlier-four-pair probe accuracy |','|---|---:|---:|']
 for name,c in curves.items():lines.append(f"| {name.replace('_',' ')} | {100*c['mean_current_pair_acquisition']:.2f}% | {100*c['final_old_four_pairs_probe']:.2f}% |")
 lines+=['','These curves use 100 fixed test observations per class, all ten output classes eligible. They distinguish current acquisition from earlier-skill access, but cannot isolate forgetting of updater weights from drift in backbone representations/head. The entire trained system changes during acquisition.','',
 '## Resource cost','',
 'Each updater has 856 learned parameters: 3,424 bytes of FP32 persistent inference weights. Its Adam weights/moments/scalar steps require approximately 10,292 bytes during training. The backbone adds 315,440 bytes including usefulness state; its parameters are 78,760. Gate logits alone require 400 bytes per image temporarily, in addition to probabilities, branch responses and recurrent workspace; no per-image gates or input examples persist across predictions. No prototype or configuration bank is needed by the updater.','',
 'All 100 branch responses are computed once per image and reused across rounds; this single-layer implementation does not require a full backbone pass per round. Multilayer routing could invalidate that simplification. The dynamics are finite learned updates, with no proof of convergence or energy descent.','',
 '| Updater | Rounds | End-to-end batch 1 | End-to-end batch 128 |','|---|---:|---:|---:|']
 for p in profile:lines.append(f"| {p['family']} | {p['steps']} | {p['inference_ms'][1]:.4f} ms | {p['inference_ms'][128]:.4f} ms |")
 lines+=['','Two-thread CPU timings include key extraction, branch computation, updater and classifier. They exclude loading/model construction. These are small-model measurements, not GPU/energy claims.','',
 '| Training sidecar | Mean router-only training time | Optimizer updates per backbone step | Recurrent-cell calls per backbone step |','|---|---:|---:|---:|']
 for i,s in enumerate(settings):
  seconds=st.mean(r['timing']['router_seconds'][i] for r in rows)
  lines.append(f"| {s['family']} | {seconds:.3f} s | {s['updates']} | {s['updates']*s['steps']} |")
 lines+=['','Router-only times exclude shared feature extraction, backbone learning, association controls and evaluation. Combined all-candidate runtime must not be interpreted as a deployable method’s cost. Equal call count is only an approximate compute control; measured sidecar times can differ.','',
 '## Interpretation','',
 'This updater did not learn a durable label-free retrieval procedure through sequential supervised training. More inference iterations did not supply the missing evidence needed to select earlier routes. Initializing near the final route can retain a latest-task bias; uniform initialization is a relevant control, not a guaranteed remedy. Historical oracle recovery continues to indicate accessible skills under assisted configurations, but does not guarantee that this small updater can infer them.','',
 'The conclusion is limited to this architecture, training schedule, learning-rate range and soft-gate objective. The experiment does not rule out generative/predictive feedback, a better preserved updater, joint backbone–router training, different capacity, or recurrent associative memory. Those would be new hypotheses, not explanations established here. No post-test tuning or favourable-seed search was performed.','',
 'A sharper next mechanism would give gates a label-free observation-prediction objective and test whether decreasing that objective actually improves classification, using frozen feature controls and matched one-pass inference. Without such a signal, recurrence can settle into a confidently wrong or recent-task solution. Do not claim attractor retrieval merely because the gates iterate.','',
 '## Verification and evidence','',
 'All 50 primary saved-updater readouts reproduce recorded accuracy exactly under frozen reevaluation. Backbone training without any sidecar reproduces development seed 810 bitwise, verifying that sidecars did not change acquisition. Probability normalization, mixture readout and finite training gradients are checked. Historical routes remain analysis-only.','',
 '- [Experiment driver](../../bdh_cl/scripts/recurrent_branch_gate_test.py), [report/audit/profile driver](../../bdh_cl/scripts/report_recurrent_branch_gate.py).',
 '- [Protocol](../../results/recurrent_branch_gate_20261026/protocol.json), [locked selection](../../results/recurrent_branch_gate_20261026/selection_lock.json), [pre-confirmation diagnostic addendum](../../results/recurrent_branch_gate_20261026/confirmation_addendum.json).',
 '- [All validation candidates](../../results/recurrent_branch_gate_20261026/development.json), [all confirmation conditions and curves](../../results/recurrent_branch_gate_20261026/confirmation.json), [summary](../../results/recurrent_branch_gate_20261026/summary.json), [contrasts](../../results/recurrent_branch_gate_20261026/contrasts.json).',
 '- [Audit](../../results/recurrent_branch_gate_20261026/audit.json), [profiles](../../results/recurrent_branch_gate_20261026/profile.json), [source hashes](../../results/recurrent_branch_gate_20261026/source_hashes.json).','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({'summary':summary,'contrasts':contrasts,'audit_passed':True},indent=2),flush=True)

if __name__=='__main__':main()
