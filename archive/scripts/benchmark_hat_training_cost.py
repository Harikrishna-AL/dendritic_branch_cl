"""Serial CPU training timer matching branch efficiency timer boundaries."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.rawpixel_gating_baselines import construct,predict,ROOT


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/branch_compute_efficiency_20261101'
    assert len(json.loads((out/'results.json').read_text()))==36
    raw=load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    split=np.load(out/'split_indices.npz');x,y=x[split['train']],y[split['train']]
    tx=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    contexts=torch.load(out/'contexts.pt',weights_only=True);indices=[torch.where(y//2==g)[0] for g in range(5)]
    rows=[]
    for seed in [101,102,103]:
        for name,order in [('forward',list(range(5))),('reverse',list(range(4,-1,-1)))]:
            m=construct('hat',seed);seconds=0.;steps=0
            for stage,g in enumerate(order):
                opt=torch.optim.SGD(m.parameters(),lr=.1)
                for epoch in range(2):
                    gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
                    shuffled=indices[g][torch.randperm(len(indices[g]),generator=gen)]
                    m.train()
                    for batchno,ids in enumerate(shuffled.split(128)):
                        start=time.perf_counter()
                        bx,by=x[ids],y[ids];opt.zero_grad(set_to_none=True)
                        s=(400-1/400)*(batchno*128/len(shuffled))+1/400
                        loss=F.cross_entropy(m(bx,g,s=s),by)+.75*m.regularizer(m.masks(g,s))
                        loss.backward();m.restrict()
                        with torch.no_grad():
                            for n,p in m.net.named_parameters():
                                if n.startswith('e'):p.grad.mul_(400/s*(torch.cosh((s*p).clamp(-50,50))+1)/(torch.cosh(p)+1))
                        torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                        with torch.no_grad():
                            for n,p in m.net.named_parameters():
                                if n.startswith('e'):p.clamp_(-6,6)
                        seconds+=time.perf_counter()-start;steps+=1
                begin=time.perf_counter();m.consolidate(g);seconds+=time.perf_counter()-begin
            ids=torch.cdist(tx,contexts).argmin(-1)
            acc=float((predict(m,tx,ids)==ty).float().mean())
            oracle=float((predict(m,tx,ty//2)==ty).float().mean())
            rows.append(dict(seed=seed,order=name,training_seconds=seconds,autonomous=acc,oracle=oracle,steps=steps))
            print('HAT',seed,name,round(seconds,3),round(acc*100,2),flush=True)
    (out/'hat_timing.json').write_text(json.dumps(rows,indent=2))
    (out/'hat_timing_source.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
