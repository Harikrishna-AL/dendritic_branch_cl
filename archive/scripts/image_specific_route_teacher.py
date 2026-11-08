"""Follow-up: distill individual alternative-branch preferences online."""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.slow_route_consolidation import run,evaluate,diagnostics,ordinary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'development_seed':110,'confirmation_seeds':[111,112,113],
              'teachers':['actual','counterfactual','soft_counterfactual'],'ridges':[.1,1.,10.],'gammas':[0.,.1,.5],
              'features':'Fixed512random tanh pooled normalized pixel cues, inherited from preceding pilot.',
              'development':'Fresh50k fit/10k stratified training validation; width inherited, select teacher/filter/ridge/communication only on validation. Direct classifier selected separately.',
              'confirmation':'Fresh full60k training and official10k testing; both forward/reverse. Original learned24x4branch backbone two epochs per pair. No pretraining, old-task replay, task ID or boundary messages to memories.',
              'teacher':'Actual batch-global active route vs per-image argmin existing counterfactual CE vs softmax(-CE/.2). All candidates/errors already computed in original training; still explicitly label-derived alternative-branch credit and NOT a biological learning rule.',
              'memory':'Same analytic input-to-branch covariance-protected synapses; links updated from contemporaneous per-image teacher choices. No image/prototype/configuration archive. Extra global covariance persists. Random feature transform fixed, branch backbone and shared classifier learned throughout.',
              'controls':'Repeat actual teacher at its own validation-selected settings on same fresh seeds/backbones; direct analytic10class classifier with own selected settings; no communication, SGD routing, wrong input cue, ordinary gates. No parameter/storage match claimed.',
              'scope':'Test better teaching signals, not biological locality or dendrite-specific superiority.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'slow_route_consolidation.py').write_text(Path('bdh_cl/scripts/slow_route_consolidation.py').read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=torch.Generator().manual_seed(110)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
    fit=torch.ones(len(y),dtype=torch.bool);fit[val]=False
    candidates=[]
    for teacher in protocol['teachers']:
        model,memories,history,_=run(110,x,y,fit,list(SPLITS),[512],teacher)
        for filtered in [False,True]:
            memory=memories[512,filtered]
            for ridge in protocol['ridges']:
                head=memory.solve(ridge)
                for gamma in protocol['gammas']:
                    result=evaluate(model,memory,x[val],y[val],head,gamma);result.pop('routes')
                    candidates.append({'teacher':teacher,'filtered':filtered,'ridge':ridge,'gamma':gamma,**result})
        print('dev',teacher,max(round(r['accuracy']*100,2) for r in candidates if r['teacher']==teacher),flush=True)
    selected=max(candidates,key=lambda r:r['accuracy'])
    actual=max([r for r in candidates if r['teacher']=='actual'],key=lambda r:r['accuracy'])
    direct=max(candidates,key=lambda r:r['direct_accuracy'])
    (out/'selection.json').write_text(json.dumps({'candidates':candidates,'selected':selected,'actual':actual,'direct':direct},indent=2))
    print('selected',selected,flush=True)
    rows=[]
    for order in ['forward','reverse']:
        pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
        for seed in protocol['confirmation_seeds']:
            backbone=None
            for teacher in sorted(set([selected['teacher'],'actual',direct['teacher']])):
                model,memories,history,seconds=run(seed,x,y,torch.ones(len(y),dtype=torch.bool),pairs,[512],teacher)
                if backbone is None:backbone={k:v.clone() for k,v in model.state_dict().items()}
                else:assert all(torch.equal(v,backbone[k]) for k,v in model.state_dict().items())
                for name,settings in [('selected_route',selected),('actual_route',actual)]:
                    if teacher!=settings['teacher']:continue
                    memory=memories[512,settings['filtered']];head=memory.solve(settings['ridge'])
                    arms=[(name,settings['gamma'],False,False)]
                    if name=='selected_route':arms += [('no_communication',0.,False,False),('sgd_route',0.,True,False),('wrong_cue',settings['gamma'],False,True)]
                    for arm,gamma,sgd,wrong in arms:
                        result=evaluate(model,memory,tx,ty,head,gamma,sgd,wrong)
                        row={'seed':seed,'order':order,'arm':arm,'teacher':teacher,**diagnostics(model,tx,ty,result['routes'],history,pairs),'accepted':memory.accepted,'numeric_router_bytes':memory.bytes(),'seconds_all_controls':seconds}
                        rows.append(row);print(order,seed,arm,round(row['accuracy']*100,2),flush=True)
                    if name=='selected_route':
                        torch.save({'backbone':backbone,'projection':memory.projection,'gram':memory.gram,'cross':memory.cross[:,:96].clone(),'head':head[:,:96].clone(),'links':memory.links,'history_analysis_only':history},out/f'{order}_{seed}.pt')
                        rows.append({'seed':seed,'order':order,'arm':'ordinary_gate','accuracy':ordinary(model,tx,ty)})
                if teacher==direct['teacher']:
                    memory=memories[512,direct['filtered']];head=memory.solve(direct['ridge'])
                    result=evaluate(model,memory,tx,ty,head)
                    rows.append({'seed':seed,'order':order,'arm':'direct_classifier','accuracy':result['direct_accuracy']})
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Image-specific route teaching','','All routing associations are acquired from current training samples while the original branch weights/classifier learn. No task IDs, saved examples or configuration archive. Full covariance statistics are explicit extra central memory.','', '| Method | Mean accuracy ± SD (six runs) |','|---|---:|']
    for arm in ['ordinary_gate','actual_route','selected_route','no_communication','sgd_route','wrong_cue','direct_classifier']:
        values=[r['accuracy']*100 for r in rows if r['arm']==arm]
        lines.append(f'| {arm} | {st.mean(values):.2f} ± {st.stdev(values):.2f}% |')
    lines+=['',f'Validation-selected route settings: `{selected}`.', '',
            'The actual-route control has independently selected settings and identical final backbone weights. The direct classifier has independently selected regularization/filtering and learns labels from the same stream/cues. It does not use the learned branch backbone and is not parameter/storage matched.', '',
            'The selected teacher uses already-computed alternative-branch training errors if selection favors counterfactual variants. That is supervised global credit assignment, not actual-outcome-only local eligibility learning. Changes in its accuracy must not be presented as validation of a biological rule or a short-term-state-specific advantage.', '',
            'Historical-route oracle is only a reference, not an upper bound for individually taught configurations. The router never receives phase-end routes or oracle task information. Model/head remain unchanged during inference. Per-seed results, development candidates, full protocols and deployment/statistics checkpoints are saved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:14]),flush=True)


if __name__=='__main__':main()
