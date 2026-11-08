"""Strong compact central assembly control and communication ablations."""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_domains import GraphLearner,permutations,train as branch_train,evaluate as branch_evaluate
from bdh_cl.scripts.conference_gate_benchmark import Direct
from bdh_cl.scripts.local_coded_feedback import ClassAssociation


class CentralAssembly(GraphLearner):
    def __init__(self,seed,setting):
        super().__init__(seed,'flat',setting)
        self.links=torch.empty(0)
        self.assemblies=torch.zeros(4,256);self.counts=torch.zeros(4)
        self.threshold=setting['threshold']

    @torch.no_grad()
    def observe(self,active,n):
        pattern=torch.zeros(256);pattern[active]=1
        used=self.counts>0
        if used.any():
            similarity=torch.nn.functional.normalize(self.assemblies,dim=1)@torch.nn.functional.normalize(pattern,dim=0)
            similarity[~used]=-1;best=int(similarity.argmax())
        else:best=-1
        if best<0 or similarity[best]<self.threshold:
            free=torch.nonzero(~used,as_tuple=True)[0]
            best=int(free[0]) if len(free) else int(self.counts.argmin())
            self.assemblies[best]=pattern;self.counts[best]=n
        else:
            self.counts[best]+=n
            self.assemblies[best]+=(pattern-self.assemblies[best])*(n/self.counts[best])

    @torch.no_grad()
    def learn(self,x,y):
        z=self.data_features(x);b=torch.tanh(z@self.weight)
        target=self.codes[y];active=self.select();self.observe(active,len(x))
        residual=target[:,active]-b[:,active]
        rate=self.rate/(1+self.usage[active]/self.tau)
        self.weight[:,active]+=(z.T@(residual*(1-b[:,active].square())))/len(x)*rate[None]
        self.usage[active]+=(residual.abs()<.2).float().sum(0)
        self.error.copy_((b-target).square().mean(0).reshape(self.n,self.k))

    @torch.no_grad()
    def predict(self,x):
        response=torch.tanh(self.data_features(x)@self.weight)
        value=-(response[:,None]-self.codes[None]).square()
        assembly=self.assemblies[self.counts>0]
        return (value@assembly.T).max(-1).values.argmax(-1)

    def bytes(self):return super().bytes()+self.assemblies.numel()*4+self.counts.numel()*4


def central_train(seed,setting,x,y,mask,perms):
    model=CentralAssembly(seed,setting);ids=torch.nonzero(mask,as_tuple=True)[0]
    for domain,p in enumerate(perms):
        for epoch in range(2):
            generator=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
            for batch in ids[torch.randperm(len(ids),generator=generator)].split(128):model.learn(x[batch][:,p],y[batch])
    return model


@torch.no_grad()
def evaluate(model,x,y,perms):
    result=[]
    for p in perms:
        pred=torch.cat([model.predict(bx[:,p]) for bx in x.split(256)])
        result.append(float((pred==y).float().mean()))
    return {'accuracy':st.mean(result),'domains':result}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=Path('results/local_graph_domains_20261013');selection=json.loads((source/'selection.json').read_text())
    protocol={'source_cohort':'local_graph_domains_20261013, reused for paired controls, not independent confirmation','development_seed':270,'test_seeds':[271,272,273],
              'central':'Four learned activity-assembly prototype vectors acquired continuously from actual selected neuronal activity. No input descriptions, labels, domain ID or boundary writes to assembly learner. These ARE central associative pattern memories in synaptic weights, explicitly different from branch-local communication. Replace least-supported prototype if novel and full; threshold.5/.8/.95 selected on training validation.',
              'matching':'Same256data units/64active units, local coded gradients, 480k presentations, rate1/tau10000 inherited from validation. Compact context has820,240numericbytes vs812,480branch, less than1% difference. Dense flat graph previous1,078,272bytes is not minimal ordinary memory.',
              'ablations':'Same saved final branch weights, gamma0; branch-index permutation of communication edges, no input/weights changed. All settings before test scoring. DIL same10labels all4domains; no CIL superiority claim.',
              'simple_controls':'Raw784Bernoulli class synapses (31,400bytes), frozen random tanh direct analytic head sized to branchnumericbudget, independently select scale/ridge on same validation. Analytic model uses covariance, explicitly engineering baseline. No replay/testlabels/masks.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist');x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    generator=torch.Generator().manual_seed(270)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=generator)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;mean=x[mask].mean(0);rows=[];selections={}
    # Same direct implementation, but reduce dimension to current budget.
    class BudgetDirect(Direct):
        def __init__(self,seed,setting):
            super().__init__(seed,setting)
            width=1
            while 784*(width+1)*4+8*(width+2)**2+160*(width+2)<=812480:width+=1
            self.width=width;g=torch.Generator().manual_seed(seed+10000)
            self.projection=torch.randn(784,width,generator=g)
            self.gram=torch.zeros(width+1,width+1,dtype=torch.float64);self.cross=torch.zeros(width+1,10,dtype=torch.float64);self.head=torch.zeros_like(self.cross)
    def direct_train(seed,setting,perms,mask):
        model=BudgetDirect(seed,setting);ids=torch.nonzero(mask,as_tuple=True)[0]
        for p in perms:
            for _ in range(2):
                for batch in ids.split(256):model.learn(x[batch][:,p],y[batch])
        model.refresh();return model
    for kind in ['global','mean_matched']:
        perms=permutations(270,kind,mean);base={'rate':1.,'tau':10000.,'gated':False};candidates=[]
        for threshold in [.5,.8,.95]:
            model=central_train(270,{**base,'threshold':threshold},x,y,mask,perms)
            result=evaluate(model,x[val],y[val],perms);candidates.append({'threshold':threshold,**result})
        central=max(candidates,key=lambda r:r['accuracy']);direct_candidates=[]
        for scale in [.5,1.,2.]:
            for ridge in [.1,1.,10.]:
                model=direct_train(270,{'scale':scale,'ridge':ridge},perms,mask)
                direct_candidates.append({'scale':scale,'ridge':ridge,**evaluate(model,x[val],y[val],perms)})
        direct=max(direct_candidates,key=lambda r:r['accuracy']);selections[kind]={'central':central,'direct':direct,'central_candidates':candidates,'direct_candidates':direct_candidates}
        (out/'selection.json').write_text(json.dumps(selections,indent=2));print('dev',kind,'central',central,'direct',direct,flush=True)
        for seed in [271,272,273]:
            perms=permutations(seed,kind,x.mean(0))
            model=central_train(seed,{**base,'threshold':central['threshold']},x,y,torch.ones(len(y),dtype=torch.bool),perms)
            result=evaluate(model,tx,ty,perms);rows.append({'kind':kind,'seed':seed,'arm':'central_assembly',**result,'numeric_bytes':model.bytes(),'assembly_counts':model.counts.tolist()})
            print(kind,seed,'central',round(result['accuracy']*100,2),flush=True)
            model=direct_train(seed,{'scale':direct['scale'],'ridge':direct['ridge']},perms,torch.ones(len(y),dtype=torch.bool))
            result=evaluate(model,tx,ty,perms);rows.append({'kind':kind,'seed':seed,'arm':'direct_analytic',**result,'numeric_bytes':model.bytes()})
            nb=ClassAssociation(784,'bernoulli')
            for p in perms:
                for _ in range(2):
                    for ids in torch.arange(len(y)).split(256):nb.learn(x[ids][:,p],y[ids])
            result=evaluate(nb,tx,ty,perms);rows.append({'kind':kind,'seed':seed,'arm':'direct_local',**result,'numeric_bytes':31400})
            checkpoint=torch.load(source/f'{kind}_{seed}_branch.pt',weights_only=True)
            setting=selection[kind]['selected']['branch'];model=GraphLearner(seed,'branch',setting)
            for field in ['weight','codes','links','error','usage']:setattr(model,field,checkpoint[field])
            for arm,gamma in [('branch',setting['gamma']),('no_communication',0.)]:
                result=branch_evaluate(model,tx,ty,perms,gamma);rows.append({'kind':kind,'seed':seed,'arm':arm,**result,'numeric_bytes':model.bytes()})
            links=model.links.clone();shuffle=torch.Generator().manual_seed(seed+80000)
            for i in range(len(model.links)):
                row=torch.randperm(4,generator=shuffle);col=torch.randperm(4,generator=shuffle)
                model.links[i]=model.links[i][row][:,col]
            result=branch_evaluate(model,tx,ty,perms,setting['gamma']);rows.append({'kind':kind,'seed':seed,'arm':'scrambled_communication',**result,'numeric_bytes':model.bytes()})
            model.links.copy_(links)
            (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Compact assembly and strong controls for response-derived context','','Paired follow-up to localgraph cohort. Compact central coactivity synapses are within1% of branchnumericstate; dense graph savings alone do not establish a unique branch advantage.','', '| Stream | Model | Mean accuracy | Numeric state |','|---|---|---:|---:|']
    for kind in ['global','mean_matched']:
        for arm in ['branch','no_communication','scrambled_communication','central_assembly','direct_analytic','direct_local']:
            rs=[r for r in rows if r['kind']==kind and r['arm']==arm]
            lines.append(f'| {kind} | {arm} | {100*st.mean(r["accuracy"] for r in rs):.2f}% | {rs[0]["numeric_bytes"]:,}bytes |')
    lines+=['','No testlabels, domain identity, outputmask or historical configuration archive enters prediction. Central assemblies are learned activity prototypes: do not describe them as no memory, or falsely claim that dendrites are uniquely capable of this retrieval. Sourcecohort reused, so controls are paired rather than fresh independent replications.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
