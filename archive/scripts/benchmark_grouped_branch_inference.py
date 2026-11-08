"""Selected-weight inference with samples grouped by retrieved route."""
import json
import statistics as st
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts.branch_compute_efficiency import construct,predictions,ROOT
from bdh_cl.data.mnist import load_raw

@torch.no_grad()
def grouped_predict(model,x,contexts,routes):
    tasks=torch.cat([torch.cdist(bx,contexts).argmin(-1) for bx in x.split(256)])
    out=torch.empty(len(x),dtype=torch.long)
    weights=model.first.weight.reshape(168,4,784);bias=model.first.bias.reshape(168,4);neurons=torch.arange(168)
    for g in tasks.unique():
        positions=torch.where(tasks==g)[0]
        selected=routes[g].long()
        w=weights[neurons,selected];b=bias[neurons,selected]
        for pos in positions.split(256):
            hidden=torch.tanh(F.linear(x[pos],w,b))
            out[pos]=model.head(hidden).argmax(-1)
    return out


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/branch_compute_efficiency_20261101'
    # Do not overlap timing with the training comparator.
    assert (out/'hat_timing.json').exists()
    raw=load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    contexts=torch.load(out/'contexts.pt',weights_only=True);rows=[]
    for seed in [101,102,103]:
        for order in ['forward','reverse']:
            ck=torch.load(out/f'{seed}_{order}_cached_exact.pt',weights_only=True)
            m=construct('branch_ce',seed);m.load_state_dict(ck['model']);m.eval();routes=ck['routes']
            expected=predictions(m,'branch_ce',x,contexts,routes)
            actual=grouped_predict(m,x,contexts,routes)
            assert torch.equal(expected,actual)
            times={'dense':[],'grouped_selected':[]}
            for repeat in range(15):
                for mode in (['dense','grouped_selected'] if repeat%2==0 else ['grouped_selected','dense']):
                    begin=time.perf_counter()
                    if mode=='dense':predictions(m,'branch_ce',x,contexts,routes)
                    else:grouped_predict(m,x,contexts,routes)
                    times[mode].append(time.perf_counter()-begin)
            rows.append(dict(seed=seed,order=order,identical_predictions=True,median_seconds={k:st.median(v) for k,v in times.items()}))
    (out/'grouped_inference.json').write_text(json.dumps(rows,indent=2))
    (out/'grouped_inference_source.py').write_bytes(Path(__file__).read_bytes())
    print(rows)
if __name__=='__main__':main()
