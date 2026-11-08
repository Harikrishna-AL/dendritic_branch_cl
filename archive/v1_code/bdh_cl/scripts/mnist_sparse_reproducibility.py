"""Factorial replication of restricted branch inputs in the original four-branch MLP."""
import json,time,math,statistics,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.mnist_adaptive_branches import SOURCE,train_pair
from bdh_cl.scripts.mnist_quadratic_branches import InteractionBranches,evaluate

class CompactProjection(nn.Module):
    def __init__(self,dense,k,seed):
        super().__init__();self.inputs=dense.in_features
        gen=torch.Generator().manual_seed(seed+20000)
        self.register_buffer('indices',torch.stack([torch.randperm(self.inputs,generator=gen)[:k] for _ in range(dense.out_features)]))
        self.weight=nn.Parameter(dense.weight.detach().gather(1,self.indices)*math.sqrt(self.inputs/k))
        self.bias=nn.Parameter(dense.bias.detach().clone())
    def forward(self,x):
        # Compact persistent parameters; dense temporary matrix uses optimized GEMM.
        # This is connection sparsity, not a sparse-compute speed claim.
        matrix=self.weight.new_zeros(len(self.weight),self.inputs).scatter(1,self.indices,self.weight)
        return F.linear(x,matrix,self.bias)

class SparseBranches(InteractionBranches):
    def __init__(self,normalized,k,seed):
        super().__init__('normalized' if normalized else 'original',branches=4)
        if k<784:self.first=CompactProjection(self.first,k,seed)


def summary(out,results):
    rows=[];lines=['# Sparse-input MLP reproducibility','', 'Split MNIST; original 168-neuron, four-branch MLP; three seeds × forward/reverse orders, two epochs per task. No replay, pretraining, task ID or saved routes in image-only inference. Test set is used for fixed, prespecified evaluation, not tuning.','', '| Normalization | Inputs per branch | Image-only raw acc | Image-only tanh acc | Oracle acc | Parameters | Model MB | Forgetting pp | Train s |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for norm in [False,True]:
        for k in [784,392,196]:
            rs=[r for r in results if r['normalized']==norm and r['inputs']==k]
            if len(rs)!=6:continue
            metrics={}
            for mode in ['strongest_linear','strongest_tanh','oracle']:
                vals=[statistics.mean(r['stages'][-1][mode]['overall']*100 for r in rs if r['seed']==seed) for seed in [101,102,103]]
                metrics[mode]=dict(mean=statistics.mean(vals),seed_sd=statistics.stdev(vals))
            drop=statistics.mean(statistics.mean(r['stages'][i]['strongest_linear']['pairs'][i]-r['stages'][-1]['strongest_linear']['pairs'][i] for i in range(4))*100 for r in rs)
            base=[r for r in results if r['normalized']==norm and r['inputs']==784]
            diffs=[(r['stages'][-1]['strongest_linear']['overall']-next(b for b in base if b['seed']==r['seed'] and b['order']==r['order'])['stages'][-1]['strongest_linear']['overall'])*100 for r in rs]
            seed_diffs=[statistics.mean(d for d,r in zip(diffs,rs) if r['seed']==s) for s in [101,102,103]]
            mean=statistics.mean(seed_diffs);half=4.30265273*statistics.stdev(seed_diffs)/math.sqrt(3)
            row=dict(normalized=norm,inputs=k,metrics=metrics,parameters=rs[0]['parameters'],model_bytes=rs[0]['model_bytes'],forgetting_pp=drop,training_seconds=statistics.mean(r['training_seconds'] for r in rs),paired_seed_mean_delta_pp=mean,paired_seed_t95_interval=[mean-half,mean+half],positive_seed_order_pairs=sum(d>0 for d in diffs))
            rows.append(row)
            fmt=lambda mode:f"{metrics[mode]['mean']:.2f} ± {metrics[mode]['seed_sd']:.2f}%"
            lines.append(f"| {norm} | {k} | {fmt('strongest_linear')} | {fmt('strongest_tanh')} | {fmt('oracle')} | {row['parameters']:,} | {row['model_bytes']/1e6:.3f} | {drop:.2f} | {row['training_seconds']:.1f} |")
    lines+=['','Accuracy ± SD uses three seed-level averages over the two orders. Memory includes compact weights and int64 connection indices; excludes optimizer, temporary matrices, activations and oracle archives. Runtime is training only on CPU, two threads. Raw inference retains the winning raw response; tanh inference retains its bounded response, matching training. Oracle uses true task identity only to select archived routes, with the full ten-class head.','', 'Paired raw image-only differences against dense, with normalization matched:','']
    for r in rows:
        if r['inputs']==784:continue
        lo,hi=r['paired_seed_t95_interval'];lines.append(f"- normalized={r['normalized']}, {r['inputs']} inputs: {r['paired_seed_mean_delta_pp']:+.2f} pp; descriptive seed-level 95% t interval [{lo:+.2f}, {hi:+.2f}] pp; positive in {r['positive_seed_order_pairs']}/6 seed/order pairs.")
    lines+=['','Fixed shared learning rate; no independent per-arm tuning. Sparse branches access fixed random pixel subsets, which can overlap. This does not implement local receptive fields. Dense versus sparse keeps neuron/branch counts fixed, not parameter counts. Three seeds do not establish broad reproducibility across datasets.']
    (out/'summary.json').write_text(json.dumps(rows,indent=2));(out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/mnist_sparse_reproducibility_20261105'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    raw=load_raw('data_cache/mnist');split=np.load(SOURCE/'split_indices.npz')
    def ds(im,lab):return (torch.tensor(im.copy()).float().reshape(-1,784)/255-.13062755)/.30810780,torch.tensor(lab.copy(),dtype=torch.long)
    x,y=ds(raw['train_images'][split['train']],raw['train_labels'][split['train']]);tx,ty=ds(raw['test_images'],raw['test_labels'])
    protocol=dict(seeds=[101,102,103],orders=['forward','reverse'],inputs=[784,392,196],normalization=[False,True],neurons=168,branches=4,epochs=2,batch_size=128,learning_rate=.0005,train_examples=len(y),test_examples=len(ty),loss='CE + .5 mean single-neuron substitution CE',gate='EMA .95 label-based training trace; image-only largest absolute response inference',sparse_implementation='Compact learnable weights plus int64 indices; temporary dense scatter for GEMM',selection='Prespecified factorial, no test-based hyperparameter selection')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.out/'source.py').write_bytes(Path(__file__).read_bytes());results=[]
    for seed in [101,102,103]:
        for order in [list(range(5)),list(reversed(range(5)))]:
            for norm in [False,True]:
                for k in [784,392,196]:
                    torch.manual_seed(seed);m=SparseBranches(norm,k,seed);opt=torch.optim.Adam(m.parameters(),lr=.0005);routes=torch.zeros(5,168,dtype=torch.uint8);stages=[];seconds=0
                    for stage,pair in enumerate(order):
                        mask=y//2==pair;start=time.perf_counter();train_pair(m,opt,x[mask],y[mask],seed,stage);seconds+=time.perf_counter()-start;routes[pair]=m.gates().argmax(-1).to(torch.uint8);stages.append(evaluate(m,tx,ty,order,routes))
                    r=dict(seed=seed,order=order,normalized=norm,inputs=k,stages=stages,training_seconds=seconds,parameters=sum(p.numel() for p in m.parameters()),model_bytes=sum(v.numel()*v.element_size() for v in m.state_dict().values()),optimizer_bytes=sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)))
                    results.append(r);(a.out/'results.json').write_text(json.dumps(results,indent=2));summary(a.out,results);print(seed,order,norm,k,{mode:round(v['overall']*100,2) for mode,v in stages[-1].items()},flush=True)
    print((a.out/'REPORT.md').read_text(),flush=True)
if __name__=='__main__':main()
