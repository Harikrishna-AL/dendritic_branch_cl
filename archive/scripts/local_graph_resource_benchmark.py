"""Fresh paired domain scaling; explicit accuracy, retention, state and latency."""
import argparse, json, time, statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.local_graph_domains import GraphLearner, branch_scores
from bdh_cl.scripts.graph_domain_controls import CentralAssembly
from bdh_cl.scripts.local_coded_feedback import ClassAssociation

BASE={'rate':1.,'tau':10000.,'gated':True,'gamma':.1}

class Branch(GraphLearner):
    def __init__(self,seed,s): super().__init__(seed,'branch',s); self.gamma=s['gamma']
    @torch.no_grad()
    def predict(self,x): return branch_scores(self,torch.tanh(self.data_features(x)@self.weight),self.gamma).argmax(1)
    def deployed_bytes(self):return self.bytes()

class Central(CentralAssembly):
    def __init__(self,seed,s):
        super().__init__(seed,s);self.assemblies=torch.zeros(s['slots'],256);self.counts=torch.zeros(s['slots'])
    def deployed_bytes(self):return self.bytes()

class MLP:
    def __init__(self,seed,s):
        torch.manual_seed(seed); w=256 if s['opt']=='sgd' else 85
        self.net=torch.nn.Sequential(torch.nn.Linear(784,w),torch.nn.ReLU() if s['act']=='relu' else torch.nn.Tanh(),torch.nn.Linear(w,10))
        self.opt=torch.optim.SGD(self.net.parameters(),lr=s['lr']) if s['opt']=='sgd' else torch.optim.Adam(self.net.parameters(),lr=s['lr'])
    def learn(self,x,y):
        self.opt.zero_grad(set_to_none=True);F.cross_entropy(self.net(x),y).backward();self.opt.step()
    @torch.no_grad()
    def predict(self,x):return self.net(x).argmax(1)
    def deployed_bytes(self):return sum(t.numel()*t.element_size() for t in self.net.state_dict().values())
    def bytes(self):return self.deployed_bytes()+sum(t.numel()*t.element_size() for d in self.opt.state.values() for t in d.values() if isinstance(t,torch.Tensor))

class NB(ClassAssociation):
    def __init__(self,seed,s):super().__init__(784,'bernoulli')
    def bytes(self):return (self.sums.numel()+self.counts.numel())*4
    def deployed_bytes(self):return self.bytes()


def make(seed,arm,s):return {'branch':Branch,'central':Central,'sgd_mlp':MLP,'adam_mlp':MLP,'local_nb':NB}[arm](seed,s)
def permutations(seed,domains):
    rng=torch.Generator().manual_seed(seed+70000)
    return [torch.arange(784)]+[torch.randperm(784,generator=rng) for _ in range(domains-1)]

@torch.no_grad()
def acc(model,x,y,perms):
    return [float((torch.cat([model.predict(b[:,p]) for b in x.split(256)])==y).float().mean()) for p in perms]

def train(seed,arm,s,x,y,mask,perms,eval_data=None):
    model=make(seed,arm,s);ids=torch.nonzero(mask,as_tuple=True)[0];matrix=[];seconds=0.;state_peak=0
    for domain,p in enumerate(perms):
        t=time.perf_counter()
        for epoch in range(2):
            rng=torch.Generator().manual_seed(seed*100000+domain*100+epoch)
            for ix in ids[torch.randperm(len(ids),generator=rng)].split(128):model.learn(x[ix][:,p],y[ix])
        seconds+=time.perf_counter()-t;state_peak=max(state_peak,model.bytes())
        if eval_data is not None:matrix.append(acc(model,*eval_data,perms[:domain+1]))
    return model,matrix,seconds,state_peak

@torch.no_grad()
def latency(model,x):
    result={}
    for batch in [1,128]:
        bx=x[:batch]
        for _ in range(3):model.predict(bx)
        vals=[]
        for _ in range(15):
            t=time.perf_counter();model.predict(bx);vals.append(time.perf_counter()-t)
        result[str(batch)]={'median_ms':1000*st.median(vals),'samples_per_second':batch/st.median(vals)}
    return result

def dump(path,obj):path.write_text(json.dumps(obj,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    dump(out/'protocol.json',{'datasets':['mnist','fashion_mnist'],'domains':[4,8],'stream':'Identity then fixed random pixel permutations; all ten labels in each domain. No model gets task ID or boundary notification. Two passes per current domain. Same source images transformed in new domains, not replay of past-domain images.', 'development':'Dev300/310; stratified50k fit/10k holdout, all holdout excluded from every domain. Use fixed2000validation examples for hyperparameter selection. Each dataset/domain count tuned separately. Fresh301-303/311-313 confirmation; full60ktrain and10ktest. Test retention matrices use fixed stratified2000test subset; final accuracy full10k per domain.', 'controls':'SGD256 MLP approximately same deployed and ongoing numeric budget, Adam85 approximately same ongoing budget including moments. Central prototype capacity4 or8 validated; its numeric bytes disclosed. Branch settings inherited validated rule, gamma independently selected0/.1/1. Local Bernoulli control31,400bytes. No replay in any arm.', 'metrics':'Full final accuracy all/old/new domains; retention subset peak-to-final forgetting and acquisition accuracy; two-thread CPU learning time excludes evaluation; median warmed inference batch1/128. Bytes include all persistent tensors and optimizer state, exclude gradients, temporary buffers, Python overhead. Timings are measured, not equal-compute protocols.', 'safety':'No test tuning, no conference-ready promise, no claim this is a language-model benchmark or exact reproduction of published biological models.'})
    (out/'source.py').write_text(Path(__file__).read_text()); selections={};rows=[]
    for dataset,dev in [('mnist',300),('fashion_mnist',310)]:
        raw=load_raw('data_cache/'+dataset)
        x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
        tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
        rng=torch.Generator().manual_seed(dev)
        val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
        mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;vid=val.reshape(10,1000)[:,:200].flatten()
        tid=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:200] for c in range(10)])
        for domains in [4,8]:
            ps=permutations(dev,domains);key=f'{dataset}_{domains}'; selections[key]={}
            for arm in ['branch','central','sgd_mlp','adam_mlp','local_nb']:
                if arm=='branch':
                    model,_,_,_=train(dev,arm,BASE,x,y,mask,ps)
                    candidates=[]
                    for gamma in [0.,.1,1.]:
                        model.gamma=gamma;s={**BASE,'gamma':gamma};a=acc(model,x[vid],y[vid],ps);candidates.append({'setting':s,'accuracy':st.mean(a),'domains':a})
                else:
                    if arm=='central':ss=[{**BASE,'threshold':t,'slots':k} for k in ([4] if domains==4 else [4,8]) for t in [.5,.8]]
                    elif arm=='sgd_mlp':ss=[{'opt':'sgd','act':a,'lr':lr} for a in ['relu','tanh'] for lr in [.03,.3]]
                    elif arm=='adam_mlp':ss=[{'opt':'adam','act':'relu','lr':lr} for lr in [.001,.003]]
                    else:ss=[{}]
                    candidates=[]
                    for s in ss:
                        model,_,_,_=train(dev,arm,s,x,y,mask,ps);a=acc(model,x[vid],y[vid],ps);candidates.append({'setting':s,'accuracy':st.mean(a),'domains':a})
                        print('dev',key,arm,s,round(st.mean(a)*100,2),flush=True)
                chosen=max(candidates,key=lambda r:r['accuracy']);selections[key][arm]={'chosen':chosen,'candidates':candidates};dump(out/'selection.json',selections)
                print('selected',key,arm,chosen,flush=True)
            # All choices locked before any official test result for this cohort.
            for seed in range(dev+1,dev+4):
                ps=permutations(seed,domains)
                for arm,sel in selections[key].items():
                    s=sel['chosen']['setting'];model,matrix,seconds,state=train(seed,arm,s,x,y,torch.ones(len(y),dtype=torch.bool),ps,(tx[tid],ty[tid]))
                    final=acc(model,tx,ty,ps)
                    forgetting=[max(matrix[t][d] for t in range(d,domains))-matrix[-1][d] for d in range(domains-1)]
                    row={'dataset':dataset,'domains':domains,'seed':seed,'arm':arm,'setting':s,'accuracy':st.mean(final),'old_accuracy':st.mean(final[:-1]),'new_accuracy':final[-1],'domain_accuracy':final,'retention_matrix':matrix,'forgetting':st.mean(forgetting),'acquisition':st.mean(matrix[d][d] for d in range(domains)),'train_seconds':seconds,'train_numeric_bytes':state,'deployed_numeric_bytes':model.deployed_bytes(),'latency':latency(model,tx)}
                    if arm=='branch':
                        row['no_communication_accuracy']=st.mean(acc(BranchCopy(model,0.),tx,ty,ps))
                    rows.append(row);dump(out/'results.json',rows);print('test',dataset,domains,seed,arm,round(row['accuracy']*100,2),'forget',round(row['forgetting']*100,2),'seconds',round(seconds,2),flush=True)
    report(out,rows)

class BranchCopy:
    def __init__(self,model,gamma):self.model=model;self.gamma=gamma
    @torch.no_grad()
    def predict(self,x):return branch_scores(self.model,torch.tanh(self.model.data_features(x)@self.model.weight),self.gamma).argmax(1)

def report(out,rows):
    lines=['# Local branch memory: performance and resource benchmark','','Three fresh paired seeds per dataset and domain count. These are task-ID-free domain-incremental streams, not class-incremental learning. All models see the same current training examples. Validation selection is recorded separately.','','| Dataset | Domains | Model | Final accuracy | Old domains | Forgetting (pp) | Train seconds | Train MB | Deployed MB | Batch1 ms | Batch128 ms |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in ['mnist','fashion_mnist']:
        for domains in [4,8]:
            for arm in ['branch','central','sgd_mlp','adam_mlp','local_nb']:
                rs=[r for r in rows if (r['dataset'],r['domains'],r['arm'])==(dataset,domains,arm)]
                if not rs:continue
                mean=lambda k:st.mean(r[k] for r in rs)
                lines.append(f'| {dataset} | {domains} | {arm} | {100*mean("accuracy"):.2f}% | {100*mean("old_accuracy"):.2f}% | {100*mean("forgetting"):.2f} | {mean("train_seconds"):.2f} | {mean("train_numeric_bytes")/1e6:.3f} | {mean("deployed_numeric_bytes")/1e6:.3f} | {st.mean(r["latency"]["1"]["median_ms"] for r in rs):.3f} | {st.mean(r["latency"]["128"]["median_ms"] for r in rs):.3f} |')
    lines+=['','Forgetting is measured on the fixed stratified2000test subset as best observed accuracy minus final accuracy for each old domain. Final and old accuracy use all10ktest images per domain. Storage includes persistent numeric tensors, not gradients, activations, temporary allocations or Python overhead. CPU timing includes current-domain learning and data indexing; evaluation is excluded. No claim of equal FLOPs or biologically implemented inference. Central capacity is independently validated and its extra state disclosed.','']
    (out/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
