"""Class-node feedback for latent branch selection, no true inference labels.

This is a latent energy classifier, not evidence of biological feedback and
not a new claim about maxout/latent-variable inference being novel.
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.conference_gate_benchmark import BUILD,fit,BranchRouter,BUDGET
from bdh_cl.scripts.coherent_route_benchmark import PooledDirect,cache


@torch.no_grad()
def feedback_predict(model,features,branches,beta,temperature):
    unary=(features@model.head.float()).reshape(-1,24,4)
    w=model.model.head.weight
    # One branch contribution and input cue per (class, neuron, branch).
    value=branches[:,None]*w[None,:,:,None]+beta*unary[:,None]
    if temperature==0:scores=value.max(-1).values.sum(-1)
    else:scores=(temperature*torch.logsumexp(value/temperature,-1)).sum(-1)
    # Class-independent offset improves numerical behavior for large beta.
    scores=scores-beta*unary.max(-1).values.sum(-1)[:,None]+model.model.head.bias[None]
    return scores.argmax(-1)


@torch.no_grad()
def evaluate(model,features,branches,y,beta,temperature):
    pred=[];start=time.perf_counter()
    for i in range(0,len(y),512):pred.append(feedback_predict(model,features[i:i+512],branches[i:i+512],beta,temperature))
    pred=torch.cat(pred)
    return {'accuracy':float((pred==y).float().mean()),'cached_inference_seconds':time.perf_counter()-start,'prediction':pred}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);BUILD['pooled_direct']=PooledDirect
    protocol={'datasets':['mnist','fashion_mnist'],'development_seeds':[170,180],'confirmation_seeds':[[171,172,173],[181,182,183]],
              'betas':[0.,.1,.5,1.,2.,5.,10.,100.],'temperatures':[0.,.1,1.],'ridges':[.1,10.],
              'budget_bytes':BUDGET,'training':'Unchanged original learned branch model and online protected input-to-branch router. No task IDs, boundary writes, image/configuration archive, testlabel feedback or outputmask. Same current stream,120k presentations. Extra global covariance retained.',
              'energy':'score_c = bias_c + sum_n max_k[W_cn*b_nk(x) + beta*s_nk(x)], or logsumexp marginalization with fixed temperature. All10classes hypothesized from input, none supplied as truth. beta->infinity approximates original input-only routing. beta0 ignores routing cue and is a class-dependent maxout control.',
              'development':'Fresh50k/10k training validation chooses beta/temperature/ridge. Direct analytic baseline independently chooses raw or pooled cues, scale and ridge at same ongoing numeric storage. No official test selection.',
              'scope':'Inference mechanism and retrieval of retained skills. This energy construction/latent maximization is not asserted novel or biologically implemented. Any gain must be compared with strong ordinary models and attributed carefully.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    selections={};rows=[]
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['development_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False
        model,_=fit('branch',seed,{'ridge':1.,'gamma':0.},x,y,mask,list(SPLITS),x[val],y[val],False)
        features,branches=cache(model,x[val]);candidates=[]
        for ridge in protocol['ridges']:
            model.settings['ridge']=ridge;model.refresh()
            for beta in protocol['betas']:
                for temperature in protocol['temperatures']:
                    result=evaluate(model,features,branches,y[val],beta,temperature);result.pop('prediction')
                    candidates.append({'ridge':ridge,'beta':beta,'temperature':temperature,**result})
            print('dev',dataset,'ridge',ridge,'best',round(max(r['accuracy'] for r in candidates)*100,2),flush=True)
        selected=max(candidates,key=lambda r:r['accuracy'])
        # Input-only router regularization selected separately.
        independent=[]
        for ridge in protocol['ridges']:
            model.settings['ridge']=ridge;model.refresh()
            pred=[]
            for i in range(0,len(val),512):pred.append(model.predict(x[val[i:i+512]]))
            independent.append({'ridge':ridge,'accuracy':float((torch.cat(pred)==y[val]).float().mean())})
        independent=max(independent,key=lambda r:r['accuracy'])
        direct_candidates=[]
        for arm in ['direct','pooled_direct']:
            for scale in [.5,1.,2.]:
                for ridge in [.1,1.,10.]:
                    _,result=fit(arm,seed,{'scale':scale,'ridge':ridge},x,y,mask,list(SPLITS),x[val],y[val],False)
                    direct_candidates.append({'arm':arm,'scale':scale,'ridge':ridge,'accuracy':result['accuracy']})
        direct=max(direct_candidates,key=lambda r:r['accuracy'])
        selections[dataset]={'selected':selected,'independent':independent,'direct':direct,'candidates':candidates,'direct_candidates':direct_candidates}
        (out/'selection.json').write_text(json.dumps(selections,indent=2));print('selected',dataset,selected,'direct',direct,flush=True)
        for order in ['forward','reverse']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for seed in protocol['confirmation_seeds'][di]:
                model,result=fit('branch',seed,{'ridge':selected['ridge'],'gamma':0.},x,y,torch.ones(len(y),dtype=torch.bool),pairs,tx,ty)
                features,branches=cache(model,tx);before=[v.clone() for v in model.arrays()]
                for arm,beta,temp in [('feedback',selected['beta'],selected['temperature']),('no_cue',0.,0.)]:
                    evaluated=evaluate(model,features,branches,ty,beta,temp);pred=evaluated.pop('prediction')
                    old=(ty!=pairs[-1][0])&(ty!=pairs[-1][1])
                    rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':arm,**evaluated,'old_accuracy':float((pred[old]==ty[old]).float().mean()),'numeric_bytes':model.bytes()})
                    print(dataset,order,seed,arm,round(evaluated['accuracy']*100,2),flush=True)
                assert all(torch.equal(a,b) for a,b in zip(before,model.arrays()))
                # Learned synapses/first layer/shared classifier fixed across arms.
                torch.save({'backbone':model.model.state_dict(),'projection':model.projection,'head':model.head,'links':model.links},out/f'{dataset}_{order}_{seed}.pt')
                model.settings['ridge']=independent['ridge'];model.refresh()
                from bdh_cl.scripts.conference_gate_benchmark import score
                evaluated=score(model,tx,ty,pairs)
                rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':'independent','accuracy':evaluated['accuracy']})
                _,result=fit(direct['arm'],seed,{'scale':direct['scale'],'ridge':direct['ridge']},x,y,torch.ones(len(y),dtype=torch.bool),pairs,tx,ty)
                rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':'direct',**result})
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Classifier feedback for branch selection','','No true inference label: all ten class hypotheses are evaluated from input. Same learned branch network and protected routing memory. No saved route or example bank. Global covariance protection remains.','', '| Dataset | Method | Accuracy ± SD (six paired runs) |','|---|---|---:|']
    for dataset in protocol['datasets']:
        for arm in ['independent','feedback','no_cue','direct']:
            values=[100*r['accuracy'] for r in rows if r['dataset']==dataset and r['arm']==arm]
            lines.append(f'| {dataset} | {arm} | {st.mean(values):.2f} ± {st.stdev(values):.2f}% |')
    lines+=['','Latent class-dependent maxout or marginalization is an engineering feedback control, not a novelty claim. Extra covariance memory and original supervised counterfactual learning remain necessary components. Data, preprocessing, hyperparameters, budgets and fresh seeds are documented in protocol/selection files. Class hypotheses are predictions, never oracle test labels.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
