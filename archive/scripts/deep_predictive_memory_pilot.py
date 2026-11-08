"""Two nonlinear layers: branch/central predictive state, versus history-only routing."""
import argparse,copy,json,statistics,time,tracemalloc
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.scripts.branch_short_memory_pilot import stream
ARMS=('branch_memory','history_only','central_memory')

class Memory(nn.Module):
    def __init__(self,width,streams=16):
        super().__init__()
        for name in ('previous','cross','energy'):self.register_buffer(name,torch.zeros(streams,width))
        self.register_buffer('seen',torch.zeros(width,dtype=torch.bool))
    @torch.no_grad()
    def write(self,q,start=0):
        end=start+q.shape[1];seen=self.seen[start:end]
        self.cross[:,start:end].mul_(.95).add_(self.previous[:,start:end]*q*.05*seen)
        self.energy[:,start:end].mul_(.95).add_(self.previous[:,start:end].square()*.05*seen)
        self.previous[:,start:end].copy_(q);self.seen[start:end]=True
    def reset(self):
        for n in ('previous','cross','energy'):getattr(self,n).zero_()
        self.seen.zero_()

class Model(nn.Module):
    def __init__(self,arm='branch_memory',streams=16):
        super().__init__();self.arm=arm
        self.first=nn.Linear(2,32);self.second=nn.Linear(8,32)
        if arm=='central_memory':self.memories=nn.ModuleList([Memory(64,streams)])
        else:self.memories=nn.ModuleList([Memory(32,streams),Memory(32,streams)])
    def state(self,layer):
        if self.arm=='central_memory':return self.memories[0],32*layer
        return self.memories[layer],0
    def scores(self,q,historical,layer,erase):
        if self.arm=='history_only':
            return (historical[:,:-1]*historical[:,1:]).sum(1)/(historical[:,:-1].square().sum(1)+.01)
        memory,start=self.state(layer)
        if erase:return torch.zeros_like(q)
        return memory.cross[:,start:start+32]/(memory.energy[:,start:start+32]+.01)
    def predict(self,x,erase=False):
        history=x.reshape(-1,8,2)
        q1=self.first(history[:,-1]);historical1=self.first(history)
        score1=self.scores(q1,historical1,0,erase).reshape(-1,8,4)
        winners1=score1.argmax(-1);mask1=F.one_hot(winners1,4).float()
        hidden=torch.tanh((q1.reshape(-1,8,4)*mask1).sum(-1))
        # Recompute an eight-step hidden history using a historical prefix-only gate.
        histscore=(historical1[:,:-1]*historical1[:,1:]).sum(1)/(historical1[:,:-1].square().sum(1)+.01)
        histmask=F.one_hot(histscore.reshape(-1,8,4).argmax(-1),4).float()
        historicalhidden=torch.tanh((historical1.reshape(-1,8,8,4)*histmask[:,None]).sum(-1))
        q2=self.second(hidden);historical2=self.second(historicalhidden)
        score2=self.scores(q2,historical2,1,erase);winners2=score2.argmax(-1)
        prediction=q2.gather(1,winners2[:,None]).squeeze(1)
        return prediction,(q1.detach(),q2.detach()),(winners1,winners2)
    @torch.no_grad()
    def commit(self,q,erase=False):
        if erase or self.arm=='history_only':self.reset();return
        for i,values in enumerate(q):m,start=self.state(i);m.write(values,start)
    def reset(self):
        for m in self.memories:m.reset()

def train(model,seed,block,order):
    opt=torch.optim.Adam(model.parameters(),lr=.003);losses=[];states=[]
    for t,(x,y,_) in enumerate(stream(seed,block*len(order),block,16,order)):
        pred,q,_=model.predict(x);losses.append(F.mse_loss(pred,y));model.commit(q)
        if (t+1)%8==0:
            opt.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses=[]
        if (t+1)%block==0:states.append(copy.deepcopy(model.state_dict()))
    return states
@torch.no_grad()
def evaluate(model,seed,erase=False,order=(0,1,0,1)):
    model.reset();rows=[{'errors':[],'routes1':[],'routes2':[]} for _ in order]
    for t,(x,y,_) in enumerate(stream(seed,256*len(order),256,16,order)):
        pred,q,w=model.predict(x,erase);model.commit(q,erase);r=rows[t//256]
        r['errors'].append((pred-y).square().mean().item());r['routes1'].append(w[0].tolist());r['routes2'].append(w[1].tolist())
    return [{'regime':order[i],'mse':statistics.mean(r['errors'][32:]),'early_mse':statistics.mean(r['errors'][:32]),
            'usage1':F.one_hot(torch.tensor(r['routes1'][32:]),4).float().mean((0,1)).tolist(),
            'usage2':(torch.bincount(torch.tensor(r['routes2'][32:]).flatten(),minlength=32).float()/(224*16)).tolist()} for i,r in enumerate(rows)]
@torch.no_grad()
def benchmark(model):
    x=next(stream(99,1,1))[0];model.reset()
    for _ in range(20):_,q,_=model.predict(x);model.commit(q)
    tracemalloc.start();start=time.perf_counter()
    for _ in range(300):_,q,_=model.predict(x);model.commit(q)
    elapsed=time.perf_counter()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return {'milliseconds_per_step':elapsed/300*1000,'python_peak_bytes':peak,
            'runtime_state_bytes':sum(b.numel()*b.element_size() for b in model.buffers()),
            'parameters':sum(p.numel() for p in model.parameters()),
            'note':'CPU two threads;tracemalloc excludes native tensor allocations;not GPU throughput or total tensor peak memory'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='0,1,2');p.add_argument('--block',type=int,default=4096)
    args=p.parse_args();torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);records=[]
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'arms':ARMS,
        'architecture':'8first-layer neurons x4branches,input2,tanh pooledwinner;second32branches,input8,one outputwinner;384trainable parameters in all arms',
        'state':'64branch responses x3scalar traces x16streams =3072 float32 scalars;central stores identical information in one router;history-only no functional persistentstate',
        'input':'eight observed steps;no taskIDs or boundary flags;both training orders;single continuous optimizer',
        'history_control':'recomputes per-layer histories using current weights;second-layerhistorical hidden routing uses whole available observed eight-step history,not future targets',
        'training':'Adam .003 every8steps;4096steps perregime;singleA4096/singleB4096/joint4096 controls',
        'evaluation':'frozenweights A/B/A/B andreverse;erasure every step;32stepwarmup',
        'cost':'CPU inference latency with warmup;exactstatebytes;Pythonallocationpeak excludes native tensors andisnot full peakmemory',
        'limits':'centralcontrol mathematicallyequivalent;stateless baseline functionalstate inherentlylower;more depth adds changed routing dynamics;no claim of dendrite-specific advantage'},indent=2))
    for seed in map(int,args.seeds.split(',')):
        for order in ((0,1),(1,0)):
            for arm in ARMS:
                torch.manual_seed(seed);initial=Model(arm);row={'seed':seed,'order':order,'arm':arm,'controls':{}}
                for label,block,schedule in [('singleA',args.block,(0,)),('singleB',args.block,(1,)),('joint',128,(0,1)*16),('sequential',args.block,order)]:
                    model=copy.deepcopy(initial);states=train(model,seed+1000,block,schedule)
                    if label=='sequential':
                        model.load_state_dict(states[0]);row['stage1']=evaluate(model,seed+3000)
                        model.load_state_dict(states[-1]);row['live']=evaluate(model,seed+3000)
                        model.load_state_dict(states[-1]);row['erased']=evaluate(model,seed+3000,True)
                        model.load_state_dict(states[-1]);row['reverse_eval']=evaluate(model,seed+4000,order=(1,0,1,0))
                        row['benchmark']=benchmark(model);torch.save(states,out/f'{seed}_{order[0]}_{arm}.pt')
                    else:row['controls'][label]=evaluate(model,seed+5000,order=schedule if label!='joint' else (0,1,0,1))
                records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
                print(seed,order,arm,'live',round(statistics.mean(t['mse'] for t in row['live']),3),flush=True)
    lines=['# Two-layer predictive memory','','Means across seeds;lower MSE better. Both training orders.','','| Order | Arm | Old MSE | New MSE | Joint avg | Erased avg | Overlap L1 / L2 | CPU ms/step |','|---|---|---:|---:|---:|---:|---:|---:|']
    for order in ((0,1),(1,0)):
        for arm in ARMS:
            rs=[r for r in records if tuple(r['order'])==order and r['arm']==arm]
            def avg(f):return statistics.mean(f(r) for r in rs)
            def overlap(r,layer):
                a=(torch.tensor(r['live'][0][f'usage{layer}'])+torch.tensor(r['live'][2][f'usage{layer}']))/2;b=(torch.tensor(r['live'][1][f'usage{layer}'])+torch.tensor(r['live'][3][f'usage{layer}']))/2
                return torch.minimum(a,b).sum(-1).mean().item() if layer==1 else torch.minimum(a,b).sum().item()
            values=[avg(lambda r:statistics.mean(t['mse'] for t in r['live'] if t['regime']==i)) for i in order]
            values+=[avg(lambda r:statistics.mean(t['mse'] for t in r['controls']['joint'])),avg(lambda r:statistics.mean(t['mse'] for t in r['erased']))]
            lines.append(f'| {order} | {arm} | '+' | '.join(f'{v:.3f}' for v in values)+f' | {avg(lambda r:overlap(r,1)):.3f} / {avg(lambda r:overlap(r,2)):.3f} | {avg(lambda r:r["benchmark"]["milliseconds_per_step"]):.3f} |')
    lines+=['','Persistent runtime state:12288bytes plus64bools, identical branch/central budgets. CPU Pythonallocationpeak is not native tensor/GPU peak memory.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
