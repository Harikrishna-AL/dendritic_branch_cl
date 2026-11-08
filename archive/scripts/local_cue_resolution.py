"""Test sensory cue bottleneck and code width in the covariance-free learner."""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.local_coded_feedback import train,evidence,ClassAssociation


@torch.no_grad()
def cache(model,x):
    responses=[];cues=[]
    for bx in x.split(512):
        responses.append(torch.tanh(model.data_features(bx)@model.weight));cues.append(evidence(model,bx))
    return torch.cat(responses),torch.cat(cues)


@torch.no_grad()
def predict(model,response,cue,beta):
    codes=model.codes.repeat_interleave(model.k,dim=1)
    if beta=='independent':
        if model.k==4:active=torch.arange(model.n)[None]*4+cue.reshape(-1,model.n,4).argmax(-1)
        else:active=cue.topk(model.active_count,dim=1).indices
        b=response.gather(1,active)
        return (b[:,:,None]*model.codes.T[active//model.k]).sum(1).argmax(1)
    value=-(response[:,None]-codes[None]).square()+beta*cue[:,None]
    if model.k==4:score=value.reshape(-1,10,model.n,4).max(-1).values.sum(-1)
    else:score=value.topk(model.active_count,dim=-1).values.sum(-1)
    return score.argmax(1)


@torch.no_grad()
def accuracy(model,cached,y,beta):
    response,cue=cached
    p=torch.cat([predict(model,response[i:i+512],cue[i:i+512],beta) for i in range(0,len(y),512)])
    return float((p==y).float().mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'datasets':['mnist','fashion_mnist'],'dev_seeds':[230,240],'confirmation_seeds':[[231,232,233],[241,242,243]],
              'branch_grid':'neurons24/64, cue196pooled/784raw, previous-batch error memory rho0 inherited from preceding training validation, rate.1/1/10, tau1000/10000. beta0/.001/.01/.1/1/10/independent. ALL candidates reported, settings selected on training validation only.',
              'flat':'Same chosen physical number ofdata units and cue dimensions, same local learning/routing/feedback. Independently select rho0/.95, activeunits N/2N/4N, rates/tau/beta. Approx matched numericstate (<1% output-code difference). No deliberately narrow flat control.',
              'direct':'Raw/pooled Bernoulli and cosine class-associative synapses independently selected, far smaller memory. No global covariance, replay, prototype/configuration archive, task ID or testlabels for ANY learner. Class sums/coactivity are persistent local associative statistics.',
              'training':'Two current-pair epochs; fixed output codes; local selected-unit gradient from per-neuron supervised coded teaching. Plasticity damped by reliable usage, no boundary resets. This is an abstract rule related to existing target-projection learning, not a novel biological derivation.',
              'gate':'Fresh3seeds x2orders confirmation only if selectedbranch validation>=60%. Higher resolution/wider models increase memory; exact bytes reported. No conference-ready claim from these simpledatasets.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text());selections={};rows=[]
    for di,dataset in enumerate(protocol['datasets']):
        raw=load_raw(f'data_cache/{dataset}');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        seed=protocol['dev_seeds'][di];rng=torch.Generator().manual_seed(seed)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;candidates=[]
        betas=[0.,.001,.01,.1,1.,10.,'independent']
        for neurons in [24,64]:
            for cue_dim in [196,784]:
                for rate in [.1,1.,10.]:
                    for tau in [1000.,10000.]:
                        setting={'neurons':neurons,'cue_dim':cue_dim,'rate':rate,'tau':tau,'rho':0.}
                        model,_=train(seed,'branch',setting,x,y,mask,list(SPLITS));cached=cache(model,x[val])
                        for beta in betas:candidates.append({'arm':'branch',**setting,'beta':beta,'accuracy':accuracy(model,cached,y[val],beta),'numeric_bytes':model.bytes()})
                print('dev',dataset,'branch',neurons,cue_dim,'best',round(max(r['accuracy'] for r in candidates if r['neurons']==neurons and r['cue_dim']==cue_dim)*100,2),flush=True)
        branch=max(candidates,key=lambda r:r['accuracy'])
        for rho in [0.,.95]:
            for count in [branch['neurons'],2*branch['neurons'],4*branch['neurons']]:
                for rate in [.1,1.,10.]:
                    for tau in [1000.,10000.]:
                        setting={'neurons':branch['neurons'],'cue_dim':branch['cue_dim'],'active_count':count,'rate':rate,'tau':tau,'rho':rho}
                        model,_=train(seed,'flat',setting,x,y,mask,list(SPLITS));cached=cache(model,x[val])
                        for beta in betas:candidates.append({'arm':'flat',**setting,'beta':beta,'accuracy':accuracy(model,cached,y[val],beta),'numeric_bytes':model.bytes()})
                print('dev',dataset,'flat',rho,count,'best',round(max(r['accuracy'] for r in candidates if r['arm']=='flat' and r['rho']==rho and r['active_count']==count)*100,2),flush=True)
        flat=max([r for r in candidates if r['arm']=='flat'],key=lambda r:r['accuracy'])
        direct=[]
        for dim in [196,784]:
            for rule in ['cosine','bernoulli']:
                model=ClassAssociation(dim,rule)
                for _ in range(2):
                    for ids in torch.nonzero(mask,as_tuple=True)[0].split(512):model.learn(x[ids],y[ids])
                pred=torch.cat([model.predict(bx) for bx in x[val].split(512)])
                direct.append({'dim':dim,'rule':rule,'accuracy':float((pred==y[val]).float().mean())})
        selected={'branch':branch,'flat':flat,'direct':max(direct,key=lambda r:r['accuracy'])}
        selections[dataset]={'selected':selected,'candidates':candidates,'direct_candidates':direct};(out/'selection.json').write_text(json.dumps(selections,indent=2))
        print('selected',dataset,selected,flush=True)
        if branch['accuracy']<.6:continue
        for order in ['forward','reverse']:
            pairs=list(SPLITS) if order=='forward' else list(reversed(SPLITS))
            for seed in protocol['confirmation_seeds'][di]:
                for arm in ['branch','flat']:
                    setting=selected[arm];model,_=train(seed,arm,setting,x,y,torch.ones(len(y),dtype=torch.bool),pairs)
                    cached=cache(model,tx);a=accuracy(model,cached,ty,setting['beta'])
                    rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':arm,'accuracy':a,'independent_accuracy':accuracy(model,cached,ty,'independent'),'numeric_bytes':model.bytes()})
                    print(dataset,order,seed,arm,round(a*100,2),flush=True)
                    torch.save({'weight':model.weight,'codes':model.codes,'error':model.error,'usage':model.usage,'cue_sum':model.cue_sum,'cue_count':model.cue_count,'setting':setting},out/f'{dataset}_{order}_{seed}_{arm}.pt')
                setting=selected['direct'];model=ClassAssociation(setting['dim'],setting['rule'])
                for pair in pairs:
                    ids=torch.nonzero((y==pair[0])|(y==pair[1]),as_tuple=True)[0]
                    for _ in range(2):
                        for batch in ids.split(512):model.learn(x[batch],y[batch])
                pred=torch.cat([model.predict(bx) for bx in tx.split(512)])
                rows.append({'dataset':dataset,'seed':seed,'order':order,'arm':'direct','accuracy':float((pred==ty).float().mean()),'numeric_bytes':model.sums.numel()*4+model.counts.numel()*4})
                (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Local cue resolution and grouped competition','','Covariance-free local learning; no task IDs, replay, global gradients, configuration archive or test-label feedback. Fixed coded teaching and branch-local associative cue sums are explicit ingredients.','', '| Dataset | Architecture | Selected validation | Fresh test mean | Numeric state |','|---|---|---:|---:|---:|']
    for dataset,selection in selections.items():
        for arm in ['branch','flat','direct']:
            rs=[r for r in rows if r['dataset']==dataset and r['arm']==arm]
            value=f'{100*st.mean(r["accuracy"] for r in rs):.2f}%' if rs else 'Not run'
            mem=f'{rs[0]["numeric_bytes"]:,}bytes' if rs else '—'
            lines.append(f'| {dataset} | {arm} | {100*selection["selected"][arm]["accuracy"]:.2f}% | {value} | {mem} |')
    lines+=['','Flat units independently tune activity capacity and learning/history/inference settings. Direct associative classifier is substantially smaller and receives standard training labels. Numerical memory does not include transient minibatches/validation caching; no peak-RAM/compute-match claim. These are local mechanism pilots, not a major-conference result.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
