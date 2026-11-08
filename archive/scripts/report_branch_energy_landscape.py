"""Saved-state verification and full report for the bounded energy pilot."""
import hashlib
import json
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bdh_cl.scripts import branch_energy_landscape as e

RAW=e.ROOT;OUT=Path('reports/branch_energy_landscape_20261026')
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True,parents=True)
 records=json.loads((RAW/'ranking.json').read_text())
 assert len(records)==2 and {r['training_seed'] for r in records}=={830,831}
 qualification=json.loads((RAW/'qualification.json').read_text())
 raw=e.load_raw('data_cache/mnist');x,y=e.tensors(raw)
 checks=[]
 for record in records:
  seed=record['training_seed'];state=torch.load(RAW/f'{seed}.pt',weights_only=True)
  _,valid=e.validation_ids(y,seed);ids=valid[:,:100].flatten()
  before=hashlib.sha256((RAW/f'{seed}.pt').read_bytes()).hexdigest()
  result=e.ranking(state,x[ids],y[ids],seed+30000)
  for a,b in zip(result['rows'],record['rows']):
   assert a['method']==b['method'] and a['selected_accuracy']==b['selected_accuracy']
   assert abs(a['correctness_auc']-b['correctness_auc'])<1e-12
  assert hashlib.sha256((RAW/f'{seed}.pt').read_bytes()).hexdigest()==before
  statistics=np.load(RAW/f'{seed}_statistics.npz')
  solved=np.stack([np.linalg.solve(statistics['cov'][v].astype(np.float64),statistics['cross'][v].T.astype(np.float64)).T for v in range(4)]).astype(np.float32)
  assert np.array_equal(solved,state['ridge_weight'].numpy())
  q=e.compress(np.concatenate([e.old.keys(z) for z in x[ids].split(256)]),16)
  direct_weight=np.linalg.solve(statistics['direct_cov'].astype(np.float64),statistics['direct_cross'].T.astype(np.float64)).T
  direct_pred=(q@direct_weight.T).argmax(-1)
  record['baseline']['direct_ridge16_supervised_reference']=float((direct_pred==y[ids].numpy()).mean())
  checks.append({'seed':seed,'ranking_reproduced':True,'ridge_solve_matches_saved':True,'checkpoint_sha256':before})

 # Branch outputs cannot see their own target quadrant.
 views=e.MASK.reshape(4,1,28,28)
 pooled=F.adaptive_avg_pool2d(views,4).flatten(1)
 assert all(bool((pooled[v,e.TARGET[v]]==0).all()) for v in range(4))
 torch.manual_seed(1);v=torch.randn(7,25,4);p=e.project_simplex(v)
 assert bool((p>=0).all()) and torch.allclose(p.sum(-1),torch.ones(7,25),atol=1e-6)
 state=torch.load(RAW/'830.pt',weights_only=True);_,valid=e.validation_ids(y,830)
 ctx=e.context(state,x[valid[0,:4]])
 with torch.no_grad():rr,_=e.raw_energies(ctx,e.candidates(ctx,30001));_,reference=e.standardized(rr)
 variable=torch.randn(4,25,4).softmax(-1).detach().requires_grad_(True)
 direction=torch.randn_like(variable);direction-=direction.mean(-1,keepdim=True);direction*=.01
 def objective(g):
  rr,_=e.raw_energies(ctx,g[:,None]);return e.standardized(rr,reference)[0]['ridge_cue_pmi'].sum()
 analytic=(torch.autograd.grad(objective(variable),variable)[0]*direction).sum()
 with torch.no_grad():finite=(objective(variable+.001*direction)-objective(variable-.001*direction))/.002
 assert torch.allclose(analytic,finite,atol=.02,rtol=.03),(analytic,finite)

 # Sidecars have not changed the original supervised acquisition.
 mask,_=e.validation_ids(y,830);bx,by=x[mask],y[mask]
 m=e.backbone.make(830,'branch1_proxy');opt=torch.optim.Adam(m.parameters(),lr=.003)
 for stage,pair in enumerate(e.SPLITS):
  ids=torch.nonzero(torch.isin(by,torch.tensor(pair))).flatten()
  for epoch in range(2):
   rng=torch.Generator().manual_seed(830*100000+stage*100+epoch)
   for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):e.backbone.step(m,opt,bx[ix],by[ix],'branch1_proxy')
 assert all(torch.equal(v,state['backbone'][k]) for k,v in m.state_dict().items())
 audit={'passed':True,'ranking_runs':checks,'target_pixels_absent_from_masked_inputs':True,
        'simplex_projection_valid':True,'composite_energy_directional_gradient_checked':True,
        'no_sidecar_backbone_bitwise_matches830':True,'classification_descent_experiment_run':False}
 save(RAW/'audit.json',audit)
 save(RAW/'direct_reference.json',[{'seed':r['training_seed'],'accuracy':r['baseline']['direct_ridge16_supervised_reference']} for r in records])

 # Explicitly a union-of-candidates screening profile, not one energy deployment.
 with torch.no_grad():
  samples=[]
  for repeat in range(6):
   start=time.perf_counter();cc=e.context(state,x[valid[0,:100]])
   pool=e.candidates(cc,30002);rr,_=e.raw_energies(cc,pool);e.standardized(rr)
   if repeat:samples.append(time.perf_counter()-start)
 save(RAW/'screening_profile.json',{'threads':2,'images':100,'candidate_configurations_per_image':64,'median_seconds':st.median(samples),
      'scope':'All six energies computed together, including construction/loading model and four masked plus clean views; not isolated deployable latency.'})
 summaries={method:{field:st.mean(next(r for r in record['rows'] if r['method']==method)[field] for record in records)
                     for field in ['correctness_auc','mean_spearman_energy_CE','selected_accuracy']} for method in e.METHODS}
 save(RAW/'summary.json',{'energies':summaries,'qualification':qualification})
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'source.py',RAW/'protocol.json',RAW/'qualification.json',RAW/'ranking.json',Path(e.backbone.__file__),Path(e.old.__file__),Path(__file__)]}
 save(RAW/'source_hashes.json',hashes)

 fig,ax=plt.subplots(1,2,figsize=(12,4.8))
 labels=['Confidence','Masked\nAdam','Masked\nridge','Input\ncue','Ridge\n+ cue','Ridge + cue\n+ neighbors']
 for i,record in enumerate(records):
  values=[next(r for r in record['rows'] if r['method']==method) for method in e.METHODS]
  ax[0].bar(np.arange(6)+(i-.5)*.36,[100*r['selected_accuracy'] for r in values],width=.36,label=f"Seed {record['training_seed']}",color=['#3b858b','#b28a60'][i])
  ax[1].plot(np.arange(6),[r['correctness_auc'] for r in values],'o-',color=['#3b858b','#b28a60'][i],label=f"Seed {record['training_seed']}")
 for a in ax:a.set_xticks(range(6),labels,fontsize=8)
 ax[0].set_ylim(0,100);ax[0].set_ylabel('Selected configuration accuracy (%)');ax[0].set_title('Lowest energy among 64 label-free proposals');ax[0].legend(fontsize=8)
 ax[1].axhline(.5,ls='--',color='gray',label='Random ordering');ax[1].set_ylim(0,1);ax[1].set_ylabel('Within-image correctness AUC');ax[1].set_title('Does lower energy rank correct predictions higher?');ax[1].legend(fontsize=8)
 fig.suptitle('Energy ranking pilot: training-validation only, two seeds')
 fig.tight_layout();fig.savefig(OUT/'energy_ranking.png',dpi=180);fig.savefig(OUT/'energy_ranking.svg');plt.close(fig)
 lines=['# Input-conditioned branch energy: validation pilot — 26 October 2026','',
 '**No new energy passed the predeclared screen for recurrent descent.** Masked prediction gave a weak ranking signal, but adding it to a simple input-cue energy reduced selected accuracy in both validation runs. No official-test benchmark or classification descent experiment was run. This result concerns the tested energies, not whether an appropriate energy landscape can exist.','',
 '## Question and scope','',
 'Could a label-free energy assign lower values to branch configurations that classify a particular input correctly? This is a necessary practical diagnostic for the proposed retrieval mechanism, but high average ranking AUC is not a mathematical prerequisite for a useful minimum. We tested energy ranking before spending on recurrent inference.','',
 'Two fresh backbone seeds (830/831) each learned sequential Split MNIST using 50,000 training images, two passes per current digit pair. A stratified 10,000-image subset of the original training split was held out; 100 images per class (1,000 total) formed each ranking set. A different 100 per class was reserved for conditional dynamics, but was not used because the screen failed. Both orders, independent seeds and official-test accuracy have not been confirmed in this pilot. The standard loader reads both raw splits; no official-test images or labels enter these calculations.','',
 'All ten output classes remain eligible. The supervised backbone and global usefulness traces are unchanged from the earlier 25-neuron, four-branch protocol. Current labels train the backbone and a reference direct classifier only. Energy learners receive current images, masked responses and pre-update branch activity; no class labels, task IDs or boundary inputs. Optimizers and state are not reset between phases. Historical configurations are kept only for explicitly assisted diagnostics.','',
 '## Energy construction','',
 '- **Confidence:** low classification entropy, a baseline that can prefer confidently incorrect predictions.',
 '- **Masked Adam / masked ridge:** hide each of four 14×14 quadrants, and predict the four coarse image cells belonging to the hidden region from the selected branch responses to the remaining image. Four linear decoders, 416 weights total, learn from current pixel targets. Adam updates can forget; ridge retains cumulative covariance/cross statistics. Their linear hypothesis spaces are identical.',
 '- **Input cue:** compatibility with a 16-dimensional mean input description stored separately for each branch. This is explicit auxiliary input memory, closely related to our earlier cue method.',
 '- **Ridge + cue / + neighbors:** combine standardized prediction and cue energies, optionally adding 0.25 times a learned local PMI compatibility energy over a 40-edge grid.',
 '',
 'Component energies are centered/scaled per image using the same label-free candidate pool. The scales would stay fixed during descent. All energy variants add a fixed 0.05 mean branch-gate entropy penalty. Coefficients, predictor learning rate (.003), ridge regularization (1.0), and cue decay (.995) were fixed before results; no rescue sweep was performed. Low reconstruction error is a hypothesis for useful configurations, not an assumed proxy for classification correctness.','',
 '## Autonomous proposals versus assisted diagnostics','',
 'The main pool has 64 configurations per image: final hard gates, uniform gates, cue-selected hard gates, softened final gates, 24 random hard configurations, 24 perturbations of the final configuration and 12 random soft configurations. These are generated without labels and contain no historical route bank. The SAME pool is scored by every energy.','',
 '**The proposal pool itself uses the 6,800-byte cue memory**, even for confidence and prediction-only energies. Their selected accuracy therefore is not evidence of memory-free retrieval. The bank-assisted diagnostic separately tries all five saved historical configurations; it is not the autonomous result.','',
 '## Ranking and selection results','',
 '| Energy | Seed 830 selected accuracy | Seed 831 selected accuracy | Correctness AUC, 830 / 831 | Mean energy–classification-loss rank correlation |','|---|---:|---:|---:|---:|']
 for method in e.METHODS:
  rr=[next(r for r in record['rows'] if r['method']==method) for record in records]
  lines.append(f"| {method.replace('_',' ')} | {100*rr[0]['selected_accuracy']:.2f}% | {100*rr[1]['selected_accuracy']:.2f}% | {rr[0]['correctness_auc']:.3f} / {rr[1]['correctness_auc']:.3f} | {summaries[method]['mean_spearman_energy_CE']:+.3f} |")
 lines+=['','Correctness AUC compares correct versus incorrect candidate predictions within the SAME image, with ties scored one half. Only images with both outcomes contribute. Positive Spearman correlation means higher energy tends to accompany higher classification loss. Labels are used for these analyses only after energies and predictions are computed.','',
 '| Reference on the same ranking sets | Seed 830 | Seed 831 |','|---|---:|---:|']
 for name in records[0]['baseline']:
  lines.append(f"| {name.replace('_',' ')} | {100*records[0]['baseline'][name]:.2f}% | {100*records[1]['baseline'][name]:.2f}% |")
 lines+=['','The direct ridge reference learns from current training class labels and has no neural backbone at inference; its 16-dimensional input statistics require 1,664 bytes plus any derived weight cache. It is a supervised baseline, not an energy learner. Candidate coverage is a label-assisted upper bound: the fraction of images with ANY correct candidate prediction. It is not achievable accuracy without a selector. Historical oracle uses the true digit pair to select a saved configuration. Neither is automatic class-incremental performance. These validation-only values must not be substituted for earlier official-test cohort results.','',
 '![Energy ranking pilot](energy_ranking.png)','',
 'The masked ridge energy selects 35.30% accuracy on average, versus 34.30% for fixed uniform gates; it is worse than uniform on seed 830 and better on seed 831. The cue baseline averages 48.15%. Adding ridge prediction lowers cue selection to 40.35%, and adding neighbors further lowers it to 35.30%. Confidence is systematically poorly aligned with classification loss. These two-seed observations do not establish a general performance advantage or a confidence interval.','',
 '## What happened to downhill inference?','',
 'Before results, we required a novel prediction/composite energy to achieve correctness AUC at least 0.60 and selected accuracy at least five points above final routing in BOTH seeds. Composites also had to beat the direct cue baseline by one point in BOTH seeds. None qualified, so no label-free classification descent or official-test confirmation was launched.','',
 'This screen is conservative, not a theorem: the cue baseline itself has low average AUC while still selecting useful minima. Nevertheless, the new prediction terms failed the stronger practical comparison against the existing cue in both seeds. We therefore have no evidence that adding these terms makes the landscape more useful. We cannot claim an observed failure of gradient descent, because classification descent was not run.','',
 '## Separate historical-bank-assisted diagnosis','',
 '| Energy selecting among five saved routes | Seed 830 | Seed 831 |','|---|---:|---:|']
 for method in e.METHODS:lines.append(f"| {method.replace('_',' ')} | {100*records[0]['bank_assisted_energy_selection_diagnostic'][method]:.2f}% | {100*records[1]['bank_assisted_energy_selection_diagnostic'][method]:.2f}% |")
 lines+=['','The bank contains old useful configurations, but the energy does not reliably identify them. This helps separate absence of candidate solutions from poor scoring, without converting the diagnostic into a no-bank claim.','',
 '## Costs','',
 'The backbone has 78,760 learned parameters plus its 100-value usefulness state. The cue proposal memory adds 6,800 bytes. Masked decoders have 416 FP32 weights (1,664 bytes). Ridge training retains 12,480 bytes of sufficient statistics; caching its derived decoder weights adds 1,664 bytes. The optional neighbor table adds 2,560 bytes. A ridge+cue+neighbor system retaining its training statistics and derived cache therefore adds **23,504 bytes**, excluding transient workspaces. If training statistics are discarded after learning, its inference-only extra state is 11,024 bytes, but future continual updates would then lose those statistics. Adam decoder weights/moments/scalar step add 4,996 bytes during training, in addition to the common cue state.','',
 'Masked prediction needs four occluded branch-response passes plus a clean classification pass. Evaluating 64 proposed configurations adds substantial work. The saved screening profile measures all six energies together, including model construction and all views; it is not isolated deployment latency. There is no demonstrated compute or memory advantage over an ordinary MLP or our simpler cue retrieval.','',
 '## Verification and conclusion','',
 'Saved-checkpoint ranking metrics reproduce exactly; ridge weights match the stored sufficient-statistic solve; target regions are absent from masked inputs; simplex projection and the composite energy directional derivative are checked. An independently trained backbone without energy learners matches seed 830 bitwise. Decoder learning therefore did not alter the stored skills being compared.','',
 '**Supported:** masked prediction error contains a weak, inconsistent relationship to useful routing, stronger than confidence in this pilot.','',
 '**Not established:** a reliably correct energy landscape, improved retrieval beyond cue memory, useful recurrent descent, unique dendritic benefit or a biological learning mechanism. The decoder is a shared learned module and can itself suffer drift/forgetting; covariance retention carries explicit memory costs.','',
 'The hypothesis remains viable only with a better specified energy. A subsequent study should identify a signal tied to task-relevant features, test ranking against the cue baseline on independent validation data, and show benefits beyond simply relabeling the existing cue method as energy minimization. More iterations alone are not justified by this pilot.','',
 '## Evidence','',
 '- [Experiment driver](../../bdh_cl/scripts/branch_energy_landscape.py), [audit/report driver](../../bdh_cl/scripts/report_branch_energy_landscape.py).',
 '- [Protocol](../../results/branch_energy_landscape_20261026/protocol.json), [all ranking results](../../results/branch_energy_landscape_20261026/ranking.json), [qualification decision](../../results/branch_energy_landscape_20261026/qualification.json), [summary](../../results/branch_energy_landscape_20261026/summary.json).',
 '- [Audit](../../results/branch_energy_landscape_20261026/audit.json), [screening cost](../../results/branch_energy_landscape_20261026/screening_profile.json), [source hashes](../../results/branch_energy_landscape_20261026/source_hashes.json).','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({'summary':summaries,'audit_passed':True,'qualification':qualification},indent=2),flush=True)

if __name__=='__main__':main()
