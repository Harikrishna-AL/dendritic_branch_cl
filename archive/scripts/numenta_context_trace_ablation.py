"""Same Numenta context-input architecture, alternative segment selectors."""
import argparse
import json
import statistics as st
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.scripts.numenta_split_mnist_comparison import DendriticMLP, ROOT, rezero_weights, state_hash, prototype_module
from bdh_cl.data.mnist import load_raw

ARMS=['native_max','trace_ce','trace_aux']
class ContextTrace(torch.nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.net=DendriticMLP(784,10,[64,64],5,784,True,kw_percent_on=.05,weight_sparsity=.5,dendrite_weight_sparsity=0.,context_percent_on=.1)
        self.register_buffer('error',torch.full((2,64,5),float(np.log(10))))
    def selected(self,d,i):
        if self.arm=='native_max':return d.abs().argmax(-1)
        return self.error[i].argmin(-1).expand(len(d),-1)
    def activate(self,z,i,record=True):
        act=self.net._activations[i]
        if record:return act(z)
        training=act.training;act.eval()
        result=act(z);act.train(training)
        return result
    def forward(self,x,c,record=True):
        for i,layer in enumerate(self.net._layers):
            y=layer.module(x);d=layer.segments(c)
            gate=d.gather(2,self.selected(d,i)[...,None]).squeeze(-1).sigmoid()
            x=self.activate(y*gate,i,record)
        return self.net._output_layers[0](x)
    def alternatives(self,x,c):
        # Exact one-neuron substitutions, including downstream k-Winners changes.
        errors=[];current=x
        for i,layer in enumerate(self.net._layers):
            y=layer.module(current);d=layer.segments(c)
            z=y*d.gather(2,self.selected(d,i)[...,None]).squeeze(-1).sigmoid()
            delta=y[:,:,None]*d.sigmoid()-z[:,:,None]
            alt=z[:,None,None,:]+delta[:,:,:,None]*torch.eye(64)[None,:,None,:]
            h=self.activate(alt.reshape(-1,64),i,False)
            for j in range(i+1,2):
                l=self.net._layers[j];yy=l.module(h);dd=l.segments(c)
                base_gate=dd.gather(2,self.selected(dd,j)[...,None]).squeeze(-1).sigmoid()
                gate=base_gate[:,None,None,:].expand(-1,64,5,-1).reshape(-1,64)
                h=self.activate(yy*gate,j,False)
            logits=self.net._output_layers[0](h).reshape(len(x),64,5,10)
            errors.append(logits)
            current=self.activate(z,i,False)
        return torch.stack(errors,dim=1)
    @torch.no_grad()
    def observe(self,e):self.error.mul_(.95).add_(e,alpha=.05)


def make(arm,seed):torch.manual_seed(seed);np.random.seed(seed);return ContextTrace(arm)

@torch.no_grad()
def score(model,x,y,contexts,oracle=False):
    model.eval();before=state_hash(model);hits=0
    for start in range(0,len(x),256):
        bx=x[start:start+256];by=y[start:start+256]
        cx=contexts[by//2] if oracle else prototype_module.infer_prototype(contexts)(bx)
        logits=model(bx,cx,False);assert logits.shape[1]==10
        hits+=int((logits.argmax(-1)==by).sum())
    assert before==state_hash(model)
    return hits/len(x)

def train(arm,seed,lr,order,x,y,indices,contexts):
    model=make(arm,seed);opt=torch.optim.Adam(model.parameters(),lr=lr)
    steps=presentations=0;t=time.perf_counter()
    for stage,g in enumerate(order):
        for epoch in range(2):
            gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            for ids in indices[g][torch.randperm(len(indices[g]),generator=gen)].split(128):
                model.train();bx,by=x[ids],y[ids];cx=contexts[g].expand(len(ids),-1)
                opt.zero_grad(set_to_none=True)
                if arm=='native_max':
                    loss=F.cross_entropy(model(bx,cx),by);feedback=None
                else:
                    # Explicit pilot compute adaptation:16 current examples for error probes.
                    n=min(16,len(ids))
                    with torch.set_grad_enabled(arm=='trace_aux'):
                        alts=model.alternatives(bx[:n],cx[:n])
                        targets=by[:n,None,None,None].expand(alts.shape[:-1]).reshape(-1)
                        e=F.cross_entropy(alts.reshape(-1,10),targets,reduction='none').reshape(alts.shape[:-1])
                        feedback=e.detach().mean(0)
                    loss=F.cross_entropy(model(bx,cx),by)
                    if arm=='trace_aux':loss=loss+.5*e.mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                model.apply(rezero_weights)
                if feedback is not None:model.observe(feedback)
                steps+=1;presentations+=len(ids)
    assert presentations==110000
    return model,dict(seconds=time.perf_counter()-t,presentations=presentations,steps=steps,
      optimizer_bytes=sum(v.numel()*v.element_size() for z in opt.state.values() for v in z.values() if torch.is_tensor(v)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[101]);args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(seeds=args.seeds,tuning_seed=113,rates=[.0005,.001,.003],arms=ARMS,architecture='Same official Numenta2x64,5context segments,50%feedforward sparsity,5%kWinners,784D prototype input for all arms',selection='native absmax versus global per-layer/per-neuron error EMA .95 argmin; CE or CE+.5counterfactual CE',counterfactual_probe_batch=16,training='two epochs/pair,batch128,continuousAdam,clip5,110k selected-path presentations',inference='prototype vector enters dendritic segments for every arm. NO archived-route lookup. Autonomous upstream nearest-input-prototype context; oracle provides true-pair PROTOTYPE ONLY, not historical branch selections.',limits='Hybrid adaptation of error selector into a different architecture, not identical original branch network. Single-seed paired-order pilot. Feedback and aux use16currentexamples per128batch to bound compute, same probe rule for bothtracearms. Different training compute. Global trace may still limit retrieval; test labels never update traces. No claim newly solved routing.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw=load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=np.random.default_rng(20261101);vi=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==c))[:500] for c in range(10)]);ti=np.setdiff1d(np.arange(len(y)),vi)
    assert not np.intersect1d(vi,ti).size
    np.savez(out/'split_indices.npz',train=ti,validation=vi)
    vx,vy=x[vi],y[vi];x,y=x[ti],y[ti];indices=[torch.where(y//2==g)[0] for g in range(5)];contexts=torch.stack([x[ix].mean(0) for ix in indices]);torch.save(contexts,out/'contexts.pt')
    trials=[];selected={}
    for arm in ARMS:
        for lr in protocol['rates']:
            m,tr=train(arm,113,lr,list(range(5)),x,y,indices,contexts)
            acc=score(m,vx,vy,contexts);trials.append(dict(arm=arm,lr=lr,validation_accuracy=acc,**tr));(out/'validation_trials.json').write_text(json.dumps(trials,indent=2));print('TUNE',arm,lr,acc,flush=True)
        selected[arm]=max([z for z in trials if z['arm']==arm],key=lambda z:(z['validation_accuracy'],-z['lr']))['lr']
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    rows=[]
    for seed in args.seeds:
        for name,order in [('forward',list(range(5))),('reverse',list(range(4,-1,-1)))]:
            for arm in ARMS:
                m,tr=train(arm,seed,selected[arm],order,x,y,indices,contexts)
                scores={mode:score(m,tx,ty,contexts,mode=='oracle_context') for mode in ['autonomous','oracle_context']}
                modelbytes=sum(v.numel()*v.element_size() for v in m.state_dict().values())
                row=dict(arm=arm,seed=seed,order=name,rate=selected[arm],scores=scores,parameters=sum(p.numel() for p in m.parameters()),retained_bytes=modelbytes+contexts.numel()*contexts.element_size(),**tr)
                rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));torch.save(m.state_dict(),out/f'{seed}_{name}_{arm}.pt');print('FINAL',seed,name,arm,scores,flush=True)
    lines=['# Matched context-input segment-selection pilot','','Same Numenta architecture, raw pixels and prototype-to-dendrite inputs in every arm. No archived routes. One seed, two orders, no seed SD.','','| Selector and loss | Autonomous | Oracle context | Retained MB |','|---|---:|---:|---:|']
    for arm in ARMS:
        rr=[z for z in rows if z['arm']==arm]
        lines.append(f"| {arm} | {100*st.mean(z['scores']['autonomous'] for z in rr):.2f}% | {100*st.mean(z['scores']['oracle_context'] for z in rr):.2f}% | {rr[0]['retained_bytes']/1e6:.3f} |")
    lines+=['','Oracle context provides the true task prototype but does NOT restore historical error traces or branch choices. Therefore these oracle numbers differ in meaning from earlier archived-route oracle results.','','Full protocol and source snapshot accompany the results. Prototype/error tensors are frozen during inference. Learning rates selected on separate seed113 validation before final tests. Counterfactual error probes use16currentexamples per minibatch; auxiliary and trace-CE share that rule. All models share initial network tensors for a given seed. This is a pilot hybrid architecture, not a full native-paper reproduction or general method ranking.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
