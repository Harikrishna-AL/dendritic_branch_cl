"""Same input-synapse budget, more branches per neuron: validation-only selection."""
import argparse,json,statistics as st
from pathlib import Path
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_resource_benchmark import BASE,permutations,train,acc,latency,dump,report,make as old_make
from bdh_cl.scripts import local_graph_resource_benchmark as resource
from bdh_cl.scripts.local_coded_branch_pilot import LocalCoded

class Grouped(LocalCoded):
    def __init__(self,seed,s):
        k=s['branches'];n=256//k
        super().__init__(seed,'branch',s['rate'],s['tau'],n,196)
        # Reinitialize for changed K with exactly256input response units.
        self.k=k;self.units=256;g=torch.Generator().manual_seed(seed)
        self.weight=torch.randn(785,256,generator=g)*.02
        self.codes=torch.empty(10,n)
        for i in range(n):
            order=torch.randperm(10,generator=g);self.codes[:,i]=-.8;self.codes[order[:5],i]=.8
        self.error=torch.full((n,k),.64);self.usage=torch.zeros(256)
        self.cue_sum=torch.empty(0);self.cue_count=torch.empty(0)
        self.links=torch.zeros(n-1,k,k);self.gamma=s['gamma']
    def select(self):return torch.arange(self.n)*self.k+self.error.argmin(-1)
    @torch.no_grad()
    def learn(self,x,y):
        z=self.data_features(x);response=torch.tanh(z@self.weight);target=self.codes[y].repeat_interleave(self.k,dim=1);active=self.select()
        residual=target[:,active]-response[:,active];reliable=(residual.abs()<.2).float();j=active%self.k
        self.links[torch.arange(self.n-1),j[:-1],j[1:]]+=(reliable[:,:-1]*reliable[:,1:]).sum(0)
        rate=self.rate/(1+self.usage[active]/self.tau)
        self.weight[:,active]+=(z.T@(residual*(1-response[:,active].square())))/len(x)*rate[None]
        self.usage[active]+=reliable.sum(0);self.error.copy_((response-target).square().mean(0).reshape(self.n,self.k))
    @torch.no_grad()
    def predict(self,x):
        response=torch.tanh(self.data_features(x)@self.weight).reshape(-1,self.n,self.k)
        value=-(response[:,None]-self.codes[None,:,:,None]).square()
        counts=self.links+1.;edge=(counts/counts.sum(-1,keepdim=True)).log();dp=value[:,:,0]
        for n in range(1,self.n):dp=(dp[:,:,:,None]+self.gamma*edge[n-1][None,None]).max(2).values+value[:,:,n]
        return dp.max(-1).values.argmax(-1)
    def bytes(self):return sum(t.numel()*t.element_size() for t in [self.weight,self.codes,self.error,self.usage,self.links])
    def deployed_bytes(self):return self.bytes()

resource.make=lambda seed,arm,s:Grouped(seed,s) if arm=='grouped' else old_make(seed,arm,s)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    source=json.loads(Path('results/local_graph_resource_benchmark_20261013/selection.json').read_text())
    dump(out/'protocol.json',{'question':'Does redistributing a fixed256input-response units into more branches per neuron support8domains better?','domains':8,'groupings':[{'neurons':64,'branches':4},{'neurons':32,'branches':8}],'selection':'Reuse training-only dev300/310 split, allholdout excluded,2000validationexamples. Per grouping rate.3/1,tau10000/40000,gamma0/.1/1. All choices locked before fresh321-323/331-333 officialtest. Strong SGD256 and compact central settings inherited from independently tuned resource benchmark; retrained on same freshseeds.', 'limits':'Changing grouping also changes fixed-code length64versus32, active branches64versus32, and communication topology. This is a package comparison, not isolation of grouping alone. No new test-label feedback or replay. Every failed candidate reported.'})
    (out/'source.py').write_text(Path(__file__).read_text());selection={};rows=[]
    for dataset,dev,testbase in [('mnist',300,320),('fashion_mnist',310,330)]:
        raw=load_raw('data_cache/'+dataset);x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        rng=torch.Generator().manual_seed(dev);val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;vid=val.reshape(10,1000)[:,:200].flatten();tid=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)])
        ps=permutations(dev,8);candidates=[]
        for k in [4,8]:
            for rate in [.3,1.]:
                for tau in [10000.,40000.]:
                    s={'branches':k,'rate':rate,'tau':tau,'gamma':.1};model,_,_,_=train(dev,'grouped',s,x,y,mask,ps)
                    for gamma in [0.,.1,1.]:
                        model.gamma=gamma;setting={**s,'gamma':gamma};a=acc(model,x[vid],y[vid],ps);candidates.append({'setting':setting,'accuracy':st.mean(a),'domains':a,'bytes':model.bytes()})
                    print('dev',dataset,k,rate,tau,'best',round(max(r['accuracy'] for r in candidates if r['setting']['branches']==k)*100,2),flush=True)
                    dump(out/'development_progress.json',{'dataset':dataset,'candidates':candidates})
        chosen=max(candidates,key=lambda r:r['accuracy']);selection[dataset]={'chosen':chosen,'candidates':candidates};dump(out/'selection.json',selection);print('selected',dataset,chosen,flush=True)
        settings={'grouped':chosen['setting'],'central':source[f'{dataset}_8']['central']['chosen']['setting'],'sgd_mlp':source[f'{dataset}_8']['sgd_mlp']['chosen']['setting']}
        for seed in range(testbase+1,testbase+4):
            ps=permutations(seed,8)
            for arm,s in settings.items():
                model,matrix,seconds,state=train(seed,arm,s,x,y,torch.ones(len(y),dtype=torch.bool),ps,(tx[tid],ty[tid]));final=acc(model,tx,ty,ps)
                row={'dataset':dataset,'domains':8,'seed':seed,'arm':arm,'setting':s,'accuracy':st.mean(final),'old_accuracy':st.mean(final[:-1]),'new_accuracy':final[-1],'domain_accuracy':final,'retention_matrix':matrix,'forgetting':st.mean(max(matrix[t][d] for t in range(d,8))-matrix[-1][d] for d in range(7)),'acquisition':st.mean(matrix[d][d] for d in range(8)),'train_seconds':seconds,'train_numeric_bytes':state,'deployed_numeric_bytes':model.deployed_bytes(),'latency':latency(model,tx)}
                if arm=='grouped':
                    gamma=model.gamma;model.gamma=0.;row['no_communication_accuracy']=st.mean(acc(model,tx,ty,ps));model.gamma=gamma
                rows.append(row);dump(out/'results.json',rows);print('test',dataset,seed,arm,round(row['accuracy']*100,2),flush=True)
    lines=['# Fixed input budget: branch grouping on eight domains','','| Dataset | Model | Accuracy | Acquisition | Forgetting (pp) | Numeric MB | Training s | Batch128 ms |','|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset in ['mnist','fashion_mnist']:
        for arm in ['grouped','central','sgd_mlp']:
            rs=[r for r in rows if r['dataset']==dataset and r['arm']==arm];m=lambda k:st.mean(r[k] for r in rs)
            lines.append(f'| {dataset} | {arm} | {100*m("accuracy"):.2f}% | {100*m("acquisition"):.2f}% | {100*m("forgetting"):.2f} | {m("train_numeric_bytes")/1e6:.3f} | {m("train_seconds"):.2f} | {st.mean(r["latency"]["128"]["median_ms"] for r in rs):.3f} |')
    lines+=['','Three fresh paired seeds. Grouping, fixed-code length, active-unit count and chain topology change together. This experiment does not isolate a uniquely dendritic mechanism. See selection.json for all failed candidates. Central/MLP controls inherit their independently validated settings and are retrained on these fresh streams.',''];(out/'REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
