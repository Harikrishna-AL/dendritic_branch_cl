"""Verify statistics-free deployment and summarize the retrieval experiments."""
import json
import statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.slow_route_consolidation import cue,recruit


@torch.no_grad()
def predict(x,state,gamma):
    # This signature accepts inputs only: no labels/IDs/statistical archive.
    z=torch.tanh(cue(x)@state['projection'])
    z=torch.cat([z,torch.ones(len(z),1)],1)
    j=recruit((z@state['routing_weights']).reshape(-1,24,4),state['neighbor_links'],gamma)
    b=torch.tanh(F.linear(x,state['branch_weights'],state['branch_bias']).reshape(-1,24,4))
    h=b.gather(2,j[:,:,None]).squeeze(-1)
    return F.linear(h,state['classifier_weights'],state['classifier_bias']).argmax(1)


def main():
    torch.set_num_threads(2)
    root=Path('results/branch_route_solution_20261013');root.mkdir(exist_ok=False)
    studies={name:json.loads(Path(f'results/{folder}/results.json').read_text()) for name,folder in [
        ('first','slow_route_consolidation_20261013'),('replication','image_specific_route_teacher_20261013'),('compact','block_route_consolidation_20261013')]}
    selected=json.loads(Path('results/image_specific_route_teacher_20261013/selection.json').read_text())['selected']
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    audits=[]
    for order in ['forward','reverse']:
        for seed in [111,112,113]:
            source=torch.load(f'results/image_specific_route_teacher_20261013/{order}_{seed}.pt',weights_only=True)
            penalty=torch.eye(513,dtype=torch.float64)*selected['ridge'];penalty[-1,-1]=0
            route_only=torch.linalg.solve(source['gram']+penalty,source['cross'])
            assert torch.allclose(route_only,source['head'],atol=1e-10,rtol=1e-10)
            b=source['backbone']
            state={'projection':source['projection'],'routing_weights':source['head'].float(),
                   'neighbor_links':source['links'].float(),'branch_weights':b['first.weight'],'branch_bias':b['first.bias'],
                   'classifier_weights':b['head.weight'],'classifier_bias':b['head.bias']}
            before={k:v.clone() for k,v in state.items()}
            pred=torch.cat([predict(bx,state,selected['gamma']) for bx in x.split(512)])
            accuracy=float((pred==y).double().mean())
            reference=next(r['accuracy'] for r in studies['replication'] if r['seed']==seed and r['order']==order and r['arm']=='selected_route')
            assert abs(accuracy-reference)<1e-12
            assert all(torch.equal(v,state[k]) for k,v in before.items())
            torch.manual_seed(seed);initial=make('local','hard');initial.requires_grad_(False)
            first_change=float((b['first.weight']-initial.first.weight).norm());head_change=float((b['head.weight']-initial.head.weight).norm())
            assert first_change>0 and head_change>0
            torch.save({'state':state,'gamma':selected['gamma']},root/f'deploy_{order}_{seed}.pt')
            audits.append({'seed':seed,'order':order,'accuracy':accuracy,'inference_numeric_bytes':sum(v.numel()*v.element_size() for v in state.values()),
                           'route_only_solve_matches':True,'statistics_free_deployment_matches':True,'inference_state_unchanged':True,
                           'branch_weight_change_from_initial':first_change,'classifier_change_from_initial':head_change,
                           'excluded':'Gram, class targets/direct classifier, cross-products, saved historical routes, SGD control, final usefulness buffer. Continued training still requires statistics.'})
    (root/'audit.json').write_text(json.dumps(audits,indent=2));(root/'audit_source.py').write_text(Path(__file__).read_text())
    def mean(study,arm,key='accuracy'):
        return 100*st.mean(r[key] for r in studies[study] if r['arm']==arm)
    lines=['# Retrieving retained branch skills automatically','','## Main finding','',
           'A useful input-to-branch router can be learned during the original sequential training stream, without storing example/configuration pairs. The original branch weights and shared classifier continue learning. At inference, the image itself selects branches through learned routing connections; all ten digit outputs remain available.', '',
           '**This is a promising retrieval result, not evidence that branches outperform strong ordinary continual-learning models.** Covariance statistics protect the router. The strongest direct classifier in these tests performs better without using the learned branch backbone.', '',
           '| Experiment | Ordinary final gates | Protected input-to-branch routing | Direct analytic classifier |','|---|---:|---:|---:|',
           f'| First development/confirmation study | {mean("first","ordinary_gate"):.2f}% | {mean("first","protected_route"):.2f}% | {mean("first","direct_classifier"):.2f}% |',
           f'| Fresh teaching-signal study | {mean("replication","ordinary_gate"):.2f}% | {mean("replication","selected_route"):.2f}% | {mean("replication","direct_classifier"):.2f}% |', '',
           'Each row averages three fresh random seeds tested in both forward and reverse digit-pair orders. Task orders sharing a seed are paired, not six independent model seeds. Development uses separate seeds and a 50,000/10,000 stratified training split. All choices are made before official test scoring. No model sees future-task inputs for fitting before their turn.', '',
           '![Retrieval controls and compact-memory comparison](retrieval_results.png)', '',
           '## How it works, in plain language','',
           '1. Short-term usefulness scores decide which branch each neuron uses during training. These scores change with recent training feedback.',
           '2. The current image and the actually selected branches teach an additional, slower input-to-branch router. The router does not receive task names or phase-end saved configurations.',
           '3. The router accumulates statistics of its fixed input cues and branch choices. These keep earlier associations involved when new associations are learned. Random nonlinear cues give it more expressive input features.',
           '4. At test time, an image activates the learned router. Each neuron selects one of its four branches. Optional neighboring-branch coactivity nudges the choices through three communication loops.',
           '5. The selected branch responses pass through the unchanged final classifier. No labels, task IDs, output masks, or saved configurations are consulted.', '',
           'The router controls inference only in these pilots; the fast usefulness teacher still controls backbone training. This is an online distillation mechanism, not yet one self-consistent recurrent circuit for acquisition and retrieval.', '',
           '## Equations','',
           r'Let $b_{nk}(x)=\tanh(w_{nk}^Tx+a_{nk})$ be branch responses, and let fast usefulness errors select $j_n=\arg\min_k E_{nk}$.', '',
           r'The fixed cue is $\phi(x)=[\tanh(R^Tq(x));1]$, where $q$ is normalized2×2 pooled pixels and $R$ is a random196×512 projection, fixed before learning.', '',
           r'For each CURRENT training sample, the routing target is $t_{nk}=\mathbf1[k=j_n]$. Accumulate $G\leftarrow G+\phi\phi^T$ and $C\leftarrow C+\phi t^T$. Solve $U=(G+\lambda D)^{-1}C$, with unpenalized intercept.', '',
           r'At inference, $s_{nk}(x)=(\phi(x)^TU)_{nk}$, $\hat j_n=\arg\max_k s_{nk}(x)$, and $z(x)=W_{\rm head}[b_{n,\hat j_n}(x)]_n+a_{\rm head}$. Neighbor terms can be added before choosing $\hat j_n$.', '',
           'The96 routing targets are branch choices, not digit labels. Training labels supervise the original branch model. A separately stored10-output direct-classifier control uses labels; its cross-products are mathematically independent of the routing solve. Deployment audits recompute routing weights using ONLY the96 branch-target cross-products.', '',
           '## What the controls show','',
           f'- Ordinary online SGD routing: {mean("replication","sgd_route"):.2f}%. Protecting input-to-route connections matters in this tested setting; a single fixed SGD rate is not an exhaustive optimizer comparison.',
           f'- Correct input cue: {mean("replication","selected_route"):.2f}%; mismatched image cue: {mean("replication","wrong_cue"):.2f}%. The router uses information in the image to expose the retained skills.',
           f'- Without neighbor communication: {mean("replication","no_communication"):.2f}%; with it: {mean("replication","selected_route"):.2f}%. The effect is small and needs larger confirmation; most of the recovery comes from the protected router.',
           '- Replacing the actual active-route teacher with per-image alternative-error teachers made development accuracy worse (actual88.88%, hard alternative81.58%, soft alternative64.39%). All candidates are recorded; these variants were not promoted to confirmation.',
           f'- Smaller cue groups: {mean("compact","compact"):.2f}% using1.73MB of router numeric state, versus {mean("compact","full"):.2f}% using3.30MB for full covariance on the SAME additional fresh seeds. This cuts router state by about48% at a3.77percentage-point average accuracy cost.',
           '- A separately validation-selected direct analytic classifier reaches about94%. It uses the same fixed cue family and sample exposure, but bypasses the learned branches. These models are not total-storage or parameter matched; the branch model is larger. This rules out claiming a demonstrated dendrite-specific performance advantage.', '',
           '## Memory and deployment','',
           'Full continuing-learning router state is3,297,800 numeric bytes, plus302,824bytes for the branch backbone including its usefulness buffer. This excludes dataset buffers, SGD/direct experimental controls, optimizer state and temporary solve allocations. No individual examples or whole configurations are archived for the learner.', '',
           f'Statistics-free inference checkpoints contain {audits[0]["inference_numeric_bytes"]:,} numeric bytes each (about0.90MB): branch weights, shared classifier, fixed cue projection, routing synapses, and local neighbor links. They exclude covariance, cross-products, oracle histories, labels and the original usefulness buffer. These checkpoints reproduce all six confirmation accuracies exactly. Continuing learning needs the statistical state again; removing it for inference does not make its training cost disappear.', '',
           '## What is still unproven','',
           'This closes part of the historical-route gap: useful routes can be acquired contemporaneously and retrieved from inputs without an external archive. It does not establish that the network preserves every old skill perfectly, that short-term branch memory alone provides retention, that a local biological learning rule works, or that branched representations beat matched ordinary networks. The fixed router cues, global covariance protection and counterfactual backbone objective are material parts of the result.', '',
           'The most informative next direction is to replace covariance protection with smaller synaptic/local mechanisms while keeping this verified retrieval result as a reference, and to measure any branch advantage against equally protected ordinary networks. Simply attaching this analytic router to branches is not by itself a novel anti-forgetting mechanism.', '',
           '## Files','',
           '- `../slow_route_consolidation_20261013/`: first study, full protocol, validation candidates, per-seed results and checkpoints.',
           '- `../image_specific_route_teacher_20261013/`: fresh teaching-signal development and confirmation.',
           '- `../block_route_consolidation_20261013/`: compact-statistics comparison.',
           '- `audit.json`: inference-only reproduction and route-target independence checks.',
           '- `deploy_*.pt`: inference checkpoints containing no covariance or historical configuration data.',
           '- Scripts: `bdh_cl/scripts/{slow_route_consolidation,image_specific_route_teacher,block_route_consolidation,audit_route_consolidation}.py`.', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels=['Final fast\ngates','SGD\nrouter','Protected\nrouter','Wrong\ninput cue','Direct\nclassifier']
    arms=['ordinary_gate','sgd_route','selected_route','wrong_cue','direct_classifier']
    vals=[mean('replication',a) for a in arms]
    axes[0].bar(range(5),vals,color=['#999999','#999999','#258b7a','#999999','#456bb1'])
    axes[0].set_xticks(range(5),labels);axes[0].set_ylim(0,103);axes[0].set_ylabel('Final all-digit accuracy (%)');axes[0].set_title('Fresh confirmation: three seeds, two orders')
    for i,v in enumerate(vals):axes[0].text(i,v+1.2,f'{v:.1f}',ha='center')
    for arm,color,label in [('compact','#dd9952','Small cue groups'),('full','#258b7a','Full covariance')]:
        rs=[r for r in studies['compact'] if r['arm']==arm]
        axes[1].scatter(rs[0]['numeric_bytes']/1e6,mean('compact',arm),s=100,color=color,label=label)
        axes[1].annotate(f'{mean("compact",arm):.1f}%',(rs[0]['numeric_bytes']/1e6,mean('compact',arm)),xytext=(5,5),textcoords='offset points')
    axes[1].set_xlim(1.3,3.8);axes[1].set_ylim(78,92);axes[1].set_xlabel('Router numeric memory for continued learning (MB)');axes[1].set_ylabel('Final all-digit accuracy (%)');axes[1].set_title('Additional paired compact-memory experiment');axes[1].legend(loc='lower right',frameon=False)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(root/'retrieval_results.png',dpi=180);fig.savefig(root/'retrieval_results.svg');plt.close(fig)
    print(json.dumps({'deployment_audits':len(audits),'numeric_inference_bytes':audits[0]['inference_numeric_bytes'],'report':str(root/'REPORT.md')},indent=2))


if __name__=='__main__':main()
