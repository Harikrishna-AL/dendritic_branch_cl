import json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;base=ROOT/'results/cifar100_oracle_routes_20261014';r=json.loads((base/'results.json').read_text());assert len(r)==18 and json.loads((base/'completion.json').read_text())['complete']
def get(mode,arm,e):
 a=[x for x in r if (x['mode'],x['arm'],x['epochs'])==(mode,arm,e)];assert len(a)==3;return a
def val(a,key):return 100*st.mean(x[key]['accuracy'] for x in a)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,e in zip(axes,[2,8]):
 a=get('sequential','local',e);acq=np.mean([x['acquisition'] for x in a],axis=0)*100;oracle=np.mean([x['oracle']['task_accuracy'] for x in a],axis=0)*100;final=np.mean([x['final']['task_accuracy'] for x in a],axis=0)*100
 ax.plot(range(1,11),acq,marker='o',label='When task was learned',color='#477d4e');ax.plot(range(1,11),oracle,marker='o',label='Final weights + old route oracle',color='#236e9e');ax.plot(range(1,11),final,marker='o',label='Final automatic configuration',color='#b15745');ax.axhline(10,linestyle='--',color='#777',label='Correct-group random guess')
 ax.set_title(f'{e} passes per task');ax.set_xlabel('Ten-class task index');ax.set_xticks(range(1,11));ax.set_ylim(0,45);ax.grid(alpha=.2)
axes[0].set_ylabel('Task accuracy, all 100 outputs eligible (%)');axes[1].legend(fontsize=7,frameon=False)
fig.savefig(OUT/'acquisition_recovery.png',dpi=180);fig.savefig(OUT/'acquisition_recovery.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,e in zip(axes,[2,8]):
 matrix=np.mean([x['route_by_task_accuracy'] for x in get('sequential','local',e)],axis=0)*100;im=ax.imshow(matrix,vmin=0,vmax=35,cmap='Blues');ax.set_title(f'{e} passes per task');ax.set_xlabel('True class-group index (analysis only)');ax.set_ylabel('Historical route index');ax.set_xticks(range(10),range(1,11));ax.set_yticks(range(10),range(1,11))
fig.colorbar(im,ax=axes,label='Accuracy (%)');fig.savefig(OUT/'route_matrix.png',dpi=180);plt.close(fig)
lines=['# Split CIFAR-100: historical branch-route oracle test','',
'Completed on 14 October 2026. **Historical routes improve final accuracy, but the strong 98% MNIST recovery does not reproduce in this small raw-pixel CIFAR-100 model. Acquisition and joint capacity are also weak, so this is not a decisive test of the broader branch-memory hypothesis.**','',
'## Results','',
'| Sequential model | Passes per task | Mean accuracy when each task was learned | Final automatic accuracy | Final correct-route oracle |','|---|---:|---:|---:|---:|']
for arm,e,name in [('local',2,'Original hard branch rule'),('uniform',2,'Uniform branch averaging'),('mlp',2,'Matched ordinary MLP'),('local',8,'Original hard branch rule, longer training')]:
 a=get('sequential',arm,e);oracle=f'{val(a,"oracle"):.2f}%' if 'oracle' in a[0] else 'Not available';lines.append(f'| {name} | {e} | {100*st.mean(x["mean_acquisition"] for x in a):.2f}% | {val(a,"final"):.2f}% | {oracle} |')
lines+=['',
'All values average three seeds (491–493), with one fixed random class order. Every prediction has all 100 logits eligible. Oracle route selection uses the true ten-class group to select an externally saved configuration, so it is explicitly task-assisted analysis. There is no output mask in the branch oracle. Uniform averaging ignores configurations.','',
'| Joint exposure capacity control | Full 100-class accuracy |','|---|---:|']
for e in [2,8]:lines.append(f'| Same branch architecture, {e} epochs with all classes mixed | {val(get("joint","local",e),"final"):.2f}% |')
lines+=['',
'Joint exposure uses the same total number of gradient-training presentations as its sequential counterpart. These controls show limited acquisition of fine-grained CIFAR-100 distinctions in this architecture; they are not continual-learning results.','',
'![Acquisition versus final recovery](acquisition_recovery.png)','',
'## Dataset and architecture','',
'The official CIFAR-100 archive was found in the parent project, copied into the workspace and verified against its published archive checksum by the existing loader. Data comprise 45,000 fitting images, 5,000 held-out training images (50/class) and the full official 10,000 test images (100/class). The loader fixes a random class order with seed 2026; ten disjoint tasks each contain ten classes. That original-ID class order is recorded in the protocol.','',
'The original MNIST architecture is adapted only for 3,072 RGB pixel inputs and 100 class outputs: 24 neurons × four tanh branches, a pooled 24-dimensional hidden vector and one shared linear classifier. Inputs are raw pixels scaled to [0,1]. Training uses current ten-class cross-entropy plus 0.5 times the mean counterfactual branch cross-entropy, branch-error EMA 0.95, hard minimum-error gates, Adam 0.001, clipping 5, batch size 128. There is no pretraining, convolution, augmentation, replay or optimizer/memory reset between tasks. This is a direct small-model transfer, not a competitive CIFAR-100 vision system.','',
'The uniform arm has identical architecture, initialization and auxiliary objective but constant 1/4 branch weights. The ordinary baseline has 93 tanh hidden units and standard cross-entropy, selected to give a similar parameter count. It does not receive the counterfactual auxiliary loss.','',
'| Model | Trainable parameters | Persistent numeric model bytes |','|---|---:|---:|']
for arm in ['local','uniform','mlp']:
 x=get('sequential',arm,2)[0];lines.append(f'| {arm} | {x["parameters"]:,} | {x["model_bytes"]:,} |')
lines+=['',
'The branch model is 297,508 parameters and the MLP is 295,189, approximately 0.8% fewer. Each branch checkpoint stores a 96-value error buffer. The external ten-route bank adds 240 int64 indices, or 1,920 bytes. Optimizer state and temporary counterfactual tensors are excluded from the persistent inference figures.','',
'## Oracle audit','',
'After each task we externally saved the 24 winning branch indices. At the end, the same final network weights and classifier were evaluated under every saved configuration on every test image. Each configuration produced 100 logits. Only afterward did true class-group information select the corresponding route for the oracle metric. No correct class within the group was supplied.','',
'Independent original forward calls, with only the gate buffer restored to each old route, reproduced all cached route predictions exactly. Final model state was restored and checked unchanged. A cyclic wrong-route control assigns each true group the next historical route and achieves 0.00% in both branch budgets. This is a diagnostic manipulation, not an inference algorithm.','',
'![Historical route by true class-group accuracy](route_matrix.png)','',
'## Retention of what was acquired','',
'| Budget | First nine tasks at acquisition | First nine tasks under final oracle routes | Last task under final route |','|---|---:|---:|---:|']
for e in [2,8]:
 a=get('sequential','local',e);acq=100*st.mean(st.mean(x['acquisition'][:9]) for x in a);old=100*st.mean(st.mean(x['oracle']['task_accuracy'][:9]) for x in a);new=100*st.mean(x['oracle']['task_accuracy'][-1] for x in a);lines.append(f'| {e} passes/task | {acq:.2f}% | {old:.2f}% | {new:.2f}% |')
lines+=['',
'The historical routes do not recover most of the acquired accuracy on older tasks. This differs from the nearly intact MNIST recovery. But initial task learning was limited too, so a low final score combines acquisition limitations and loss of accessible earlier skills.','',
'## Interpreting task assistance','',
'Unrestricted random 100-class guessing has expected accuracy 1%. An oracle identifying the correct ten-class group permits a strategy that guesses within that group to achieve 10%. The branch oracle does **not** explicitly mask outputs, but its provided group information makes 10% a useful contextual reference. Its 9.02% two-pass score is below that reference; 13.66% after eight passes is only modestly above it. Comparing solely with 1% would exaggerate the strength of the oracle result.','',
'The final ordinary MLP reaches '+f'{val(get("sequential","mlp",2),"final"):.2f}%'+ ' without assistance. A separate diagnostic explicitly masks its outputs to the true ten-class task and reaches '+f'{val(get("sequential","mlp",2),"oracle_task_output_mask"):.2f}%'+'. This uses a different oracle intervention from branch-route restoration, so it is not a pure architecture comparison. It demonstrates that supplying task information can help an ordinary classifier too.','',
'## Conclusion and next step','',
'There is a measurable route-dependent effect at unchanged final weights, but no strong CIFAR-100 retained-skill result in this pilot. Increasing training improves acquisition and some recovery, without closing the gap. Results cannot establish a branch-specific advantage, and the original fixed hyperparameters were not independently tuned for CIFAR-100.','',
'The next useful experiment should use a stronger visual representation shared by branch and ordinary controls, first demonstrate good acquisition, and then repeat this unrestricted 100-class oracle audit. A frozen pretrained encoder would test the downstream routing mechanism; a jointly learned encoder would test a broader learning system. Those settings must be reported separately from this from-scratch raw-pixel pilot.','',
'## Reproducibility and checks','',
'- [Experiment driver](../../bdh_cl/scripts/cifar100_oracle_routes.py), [protocol and class order](../../results/cifar100_oracle_routes_20261014/protocol.json), [18 run records](../../results/cifar100_oracle_routes_20261014/results.json), [final checkpoints and routes](../../results/cifar100_oracle_routes_20261014/models).','- Verified dataset checksum, counterfactual logits against direct neuron replacement, 100-output shape, all ten historical-route predictions against independent original forwards, and complete final-state invariance during oracle analysis.','- Three seeds, one class order, small raw-pixel architecture, fixed transferred optimizer/settings, and two predetermined training budgets. Test outcomes did not select hyperparameters. No novelty, broad MLP superiority or conference-readiness claim follows from these numbers.','']
(OUT/'REPORT.md').write_text('\n'.join(lines));print('All18runs verified; generated report and scientific figures.')
