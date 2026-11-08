"""Direct exchange of partial sensory responses, distinct from routing compatibility."""
import statistics as st
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.branch_fields_retrieval import OUT,Model,train,evaluate,settings,family,save

ARMS=['full_branch1_self','full_branch1_neighbor',
      'quadrant_branch1_self','quadrant_branch1_neighbor',
      'quadrant_branch2_self','quadrant_branch2_neighbor']


class Exchange(Model):
    def __init__(self,seed,arm):
        base,mode=arm.rsplit('_',1)
        super().__init__(seed,base)
        self.arm=arm;self.mode=mode
        self.gain=nn.Parameter(torch.full((25,4),.1))

    def response(self,x):
        z=F.linear(x,self.hidden.weight*self.field.repeat(25,1),self.hidden.bias).reshape(-1,25,4)
        evidence=torch.tanh(z)
        # Four image quarters have a geometrical cycle: TL -> TR -> BR -> BL.
        ids=torch.tensor([0,1,3,2]);inverse=torch.argsort(ids)
        cycle=evidence[:,:,ids]
        incoming=((cycle.roll(1,-1)+cycle.roll(-1,-1))*.5)[:,:,inverse] if self.mode=='neighbor' else evidence
        return (.75*evidence+.25*torch.tanh(z+self.gain*incoming)).flatten(1)


def verify():
    m=Exchange(1,'quadrant_branch1_neighbor');x=torch.rand(3,784)
    with torch.no_grad():
        m.gain.zero_()
        assert torch.allclose(m.response(x),Model.response(m,x),atol=1e-7)
        m.gain.fill_(1.)
        changed=x.clone();changed[:,14:28]=changed[:,14:28]+.5
        # Top-left response can now receive top-right evidence despite its own field restriction.
        assert not torch.equal(m.response(x).reshape(-1,25,4)[:,:,0],m.response(changed).reshape(-1,25,4)[:,:,0])
    m=Exchange(1,'quadrant_branch2_neighbor');log=m(x)
    F.cross_entropy(log,torch.arange(3)).backward()
    assert m.gain.grad is not None and torch.isfinite(m.gain.grad).all()
    return {'zero_gain_recovers_no_exchange_response':True,'neighbor_field_can_influence_local_response':True,'exchange_gain_has_finite_gradients':True}


def main():
    torch.set_num_threads(2)
    save('feature_exchange_verification.json',verify())
    save('feature_exchange_protocol.json',{
        'question':'Does direct exchange of partial sensory responses help, beyond compatibility messages used only for routing?',
        'arms':ARMS,'update':'One exchange step amongfour branches within eachgroup. Local z sees ownfield. e=tanh(z);h=.75e+.25tanh(z+gain*mean_neighbor_e).100learnedgains initialized.1. Geometric adjacency TL-TR-BR-BL. Self-only uses e in place of neighbor mean, sameparameters.',
        'selection':'Seed560 stratified50k/10k,Adam.001/.003 by historicalvalidation then perreadout automaticvalidation. Threefreshseeds561-563 bothorders, samepairedcohort asprimary, not independent replication.',
        'interpretation':'Added as direct sensory exchange followup to primary routing compatibility test. Receivers indirectly see neighboring fields; gradients can reach unselected sending branches. Thus this can change protection aswell as featureintegration. One step and tiny gains search do not exhaust communication designs.',
        'storage':'100extra trainable scalars. Cues and associative methods unchanged; model sensoryexchange happens before masking, separately from optional routingcompatibility.'})
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    gen=torch.Generator().manual_seed(560)
    valid=torch.cat([torch.nonzero(y==c).flatten()[torch.randperm(int((y==c).sum()),generator=gen)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[valid]=False;fx,fy,vx,vy=x[mask],y[mask],x[valid],y[valid]
    selection={}
    for arm in ARMS:
        candidates=[]
        for lr in [.001,.003]:
            m,mem,bank,acq=train(560,arm,lr,fx,fy,vx,vy,SPLITS,Exchange)
            rr,diag=evaluate(m,mem,bank,vx,vy,SPLITS,settings(arm))
            candidates.append({'lr':lr,'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq)})
            print('DEV',arm,lr,round(100*diag['historical']['accuracy'],2),flush=True)
        chosen=max(candidates,key=lambda r:(r['diagnostics']['historical']['accuracy'],r['acquisition']))
        selected=[]
        for name in sorted(set(family(s) for s in chosen['methods'])):
            s=max([s for s in chosen['methods'] if family(s)==name],key=lambda s:s['accuracy'])
            selected.append({k:s[k] for k in ['method','scope','gamma','ridge'] if k in s})
        selection[arm]={'lr':chosen['lr'],'readouts':selected,'candidates':candidates}
        save('feature_exchange_selection.json',selection)
    rows=[]
    for seed in [561,562,563]:
        for name,order in [('forward',SPLITS),('reverse',SPLITS[::-1])]:
            for arm in ARMS:
                c=selection[arm];m,mem,bank,acq=train(seed,arm,c['lr'],x,y,tx,ty,order,Exchange)
                rr,diag=evaluate(m,mem,bank,tx,ty,order,c['readouts'])
                rows.append({'seed':seed,'order':name,'arm':arm,'lr':c['lr'],'methods':rr,'diagnostics':diag,'acquisition':st.mean(acq),
                             'gain_mean':float(m.gain.detach().mean()),'gain_abs_mean':float(m.gain.detach().abs().mean())})
                save('feature_exchange_results.json',rows)
                torch.save({'model':m.state_dict(),'bank_analysis_only':bank,'memory':{'cov':mem.cov,'cross':mem.cross,'direct':mem.direct,'sums':mem.sums,'links':mem.links}},OUT/f'{seed}_{name}_{arm}.pt')
                print('TEST',seed,name,arm,'historical',round(100*diag['historical']['accuracy'],2),[(family(s),round(100*s['accuracy'],2)) for s in rr],flush=True)
    save('feature_exchange_completion.json',{'complete':len(rows)==36,'confirmation_backbones':len(rows),'development_backbones':12})


if __name__=='__main__':main()
