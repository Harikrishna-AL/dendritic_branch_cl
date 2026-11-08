"""Constrained raw-pixel HAT, TWTA and FTN adaptations, not paper reproductions."""
import argparse
import ast
import hashlib
import json
import math
import time
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw

ROOT=Path(__file__).resolve().parents[2]
ARMS=['hat','twta','ftn_slow']

def load_hat():
    path=ROOT/'external/hat/src/networks/mlp_hat.py'
    tree=ast.parse(path.read_text())
    body=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    scope={'torch':torch}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),scope)
    return scope['Net']

class HAT(nn.Module):
    def __init__(self):
        super().__init__()
        # Native network and masks, shared 10-way head for unrestricted Class-IL.
        self.net=load_hat()([1,28,28],[(i,2) for i in range(5)],nhid=428)
        self.net.last=nn.ModuleList([nn.Linear(428,10)])
        self.net.taskcla=[(0,10)]
        self.register_buffer('used1',torch.zeros(1,428))
        self.register_buffer('used2',torch.zeros(1,428))
        self.started=False
    def forward(self,x,g,s=400):
        return self.net(torch.tensor([g]),x,s=s)[0][0]
    def masks(self,g,s=400):
        return self.net.mask(torch.tensor([g]),s=s)
    def regularizer(self,masks):
        if not self.started:
            return sum(m.sum() for m in masks)/856
        free=[1-self.used1,1-self.used2]
        return sum((m*a).sum() for m,a in zip(masks,free))/sum(a.sum() for a in free).clamp_min(1e-8)
    def restrict(self):
        if not self.started:return
        for name,param in self.net.named_parameters():
            view=self.net.get_view_for(name,[self.used1,self.used2])
            if view is not None and param.grad is not None:param.grad.mul_(1-view)
        # Shared head is updated as in single-head adaptation, unlike native task heads.
    @torch.no_grad()
    def consolidate(self,g):
        a,b=self.masks(g)
        self.used1.maximum_(a) if hasattr(self.used1,'maximum_') else self.used1.copy_(torch.maximum(self.used1,a))
        self.used2.copy_(torch.maximum(self.used2,b));self.started=True

class TWTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=166; self.j=4
        self.first=nn.Linear(784,664)
        self.head=nn.Linear(664,10)
        self.posterior=nn.Parameter(torch.empty(5,166,4))
        nn.init.xavier_normal_(self.first.weight);nn.init.zeros_(self.first.bias)
        nn.init.xavier_normal_(self.head.weight);nn.init.zeros_(self.head.bias)
        for t in range(5):nn.init.xavier_normal_(self.posterior[t])
    def forward(self,x,g,tau=None):
        if tau is None:
            gate=F.one_hot(self.posterior[g].argmax(-1),4).to(x.dtype)
        else:
            gate=F.gumbel_softmax(self.posterior[g],tau=tau,hard=True,dim=-1)
        # Linear competing units retain their separate downstream connections.
        return self.head((self.first(x).reshape(-1,166,4)*gate).flatten(1))

class FTN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.ParameterList([nn.Parameter(torch.empty(64,10,784))]+[nn.Parameter(torch.empty(64,10,10)) for _ in range(3)])
        self.biases=nn.ParameterList([nn.Parameter(torch.zeros(64,10)) for _ in range(4)])
        self.scalar=nn.Parameter(torch.empty(64,10))
        self.scalar_bias=nn.Parameter(torch.zeros(64))
        self.readout=nn.Parameter(torch.empty(10,64))
        self.register_buffer('masks',torch.zeros(5,64))
        for w in self.weights:
            for a in w:nn.init.kaiming_uniform_(a,a=math.sqrt(5))
        nn.init.uniform_(self.scalar,-1/math.sqrt(10),1/math.sqrt(10))
        nn.init.uniform_(self.readout,-1/8,1/8)
    def forward_mask(self,x,mask):
        # Select active slots for hard masks, all slots for differentiable proposals.
        ids=torch.arange(64) if mask.requires_grad else torch.where(mask!=0)[0]
        h=F.linear(x,self.weights[0][ids].reshape(-1,784),self.biases[0][ids].flatten()).reshape(len(x),len(ids),10)
        h=F.relu(h)
        for w,b in zip(self.weights[1:],self.biases[1:]):h=F.relu(torch.einsum('bhi,hoi->bho',h,w[ids])+b[ids])
        z=(h*self.scalar[ids]).sum(-1)+self.scalar_bias[ids]
        z=F.dropout(z,.2,training=self.training)*mask[ids]
        return F.linear(z,self.readout[:,ids])
    def forward(self,x,g):return self.forward_mask(x,self.masks[g])
    def configure(self,x,y):
        # Eq/Algorithm1, cold-start mask, supervised CURRENT TRAINING batch only.
        training=self.training;self.eval()
        logits=torch.zeros(64,requires_grad=True)
        opt=torch.optim.Adam([logits],lr=.2)
        for _ in range(20):
            opt.zero_grad()
            loss=F.cross_entropy(self.forward_mask(x,logits.sigmoid()),y)
            logits.grad=torch.autograd.grad(loss,logits)[0]
            opt.step()
        sm=logits.detach().reshape(1,1,8,8)
        for _ in range(15):sm=F.avg_pool2d(F.pad(sm,(1,1,1,1),mode='circular'),3,stride=1)
        mask=torch.zeros(64);mask[sm.flatten().topk(8).indices]=1
        self.train(training)
        return mask

def construct(arm,seed):
    torch.manual_seed(seed);np.random.seed(seed)
    return {'hat':HAT,'twta':TWTA,'ftn_slow':FTN}[arm]()

def statehash(model):
    h=hashlib.sha256()
    for z in model.state_dict().values():h.update(z.detach().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def predict(model,x,ids):
    model.eval();outputs=torch.empty(len(x),dtype=torch.long)
    for start in range(0,len(x),256):
        bx=x[start:start+256];groups=ids[start:start+256]
        result=outputs[start:start+256]
        for g in groups.unique():
            mask=groups==g
            logits=model(bx[mask],int(g))
            assert logits.shape[1]==10
            result[mask]=logits.argmax(1)
    return outputs

def train(arm,seed,rate,order,x,y,vx,vy,indices):
    model=construct(arm,seed)
    stages=[];steps=presentations=0
    begin=time.perf_counter()
    for stage,g in enumerate(order):
        # Native task-reset SGD/Adam, no momentum/weight decay in SGD.
        opt=(torch.optim.Adam(model.parameters(),lr=rate) if arm=='ftn_slow' else torch.optim.SGD(model.parameters(),lr=rate))
        ids_task=indices[g]
        for epoch in range(2):
            gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            shuffled=ids_task[torch.randperm(len(ids_task),generator=gen)]
            if arm=='ftn_slow':
                model.masks[g]=model.configure(x[shuffled[:128]],y[shuffled[:128]])
            model.train()
            for batchno,ids in enumerate(shuffled.split(128)):
                bx,by=x[ids],y[ids]
                opt.zero_grad(set_to_none=True)
                if arm=='hat':
                    s=(400-1/400)*(batchno*128/len(shuffled))+1/400
                    loss=F.cross_entropy(model(bx,g,s=s),by)+.75*model.regularizer(model.masks(g,s))
                elif arm=='twta':
                    progress=(epoch*len(shuffled)+batchno*128)/(2*len(shuffled))
                    for group in opt.param_groups:group['lr']=rate*(1-progress)
                    loss=F.cross_entropy(model(bx,g,tau=.67+(.01-.67)*progress),by)
                else:loss=F.cross_entropy(model(bx,g),by)
                loss.backward()
                if arm=='hat':
                    model.restrict()
                    with torch.no_grad():
                        for name,p in model.net.named_parameters():
                            if name.startswith('e'):
                                p.grad.mul_(400/s*(torch.cosh((s*p).clamp(-50,50))+1)/(torch.cosh(p)+1))
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                # FTN: explicitly suppress Adam updates on inactive slots, including moments.
                if arm=='ftn_slow':
                    inactive=model.masks[g]==0
                    old=[p.detach().clone() for p in model.parameters()]
                opt.step()
                if arm=='ftn_slow':
                    with torch.no_grad():
                        for p,prev in zip(model.parameters(),old):
                            if p is model.readout:p[:,inactive]=prev[:,inactive]
                            else:p[inactive]=prev[inactive]
                            for v in opt.state[p].values():
                                if torch.is_tensor(v) and v.shape==p.shape:
                                    if p is model.readout:v[:,inactive]=0
                                    else:v[inactive]=0
                if arm=='hat':
                    with torch.no_grad():
                        for n,p in model.net.named_parameters():
                            if n.startswith('e'):p.clamp_(-6,6)
                steps+=1;presentations+=len(ids)
        if arm=='hat':model.consolidate(g)
        mask=vy//2==g
        pred=predict(model,vx[mask],torch.full_like(vy[mask],g))
        stages.append({'group':g,'validation_acquisition':float((pred==vy[mask]).float().mean())})
    assert presentations==110000
    return model,dict(steps=steps,presentations=presentations,seconds=time.perf_counter()-begin,stages=stages,
                     optimizer_state_bytes=sum(v.numel()*v.element_size() for z in opt.state.values() for v in z.values() if torch.is_tensor(v)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--arms',nargs='+',default=ARMS,choices=ARMS);p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103]);args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'results.json').exists()
    protocol=dict(arms=args.arms,seeds=args.seeds,epochs_per_task=2,batch=128,train_count=55000,validation_count=5000,
                  rates={'hat':[.01,.05,.1],'twta':[.01,.05,.1],'ftn_slow':[.0003,.001,.003]},tuning_seed=113,
                  input='784 standardized raw pixels, no CNN/pretrained encoder/PCA',
                  inference='Common nearest-input-task-mean routing, no test labels; oracle task route separate, all ten outputs eligible',
                  source_status={'hat':'official Net/mask/get_view_for, custom modern CPU single-10-class-head training adapter',
                    'twta':'paper reconstruction, no official implementation verified; Eq5 straight-through hard Gumbel-Softmax, separate winner outputs; task logits retained',
                    'ftn_slow':'paper reconstruction, no official implementation verified. D8 H64 inner10 L4, ReLU private MLPs, dropout .2, no shared bias. Native D32 H1024 inner8 L8 scaled for total parameter budget. Slow3x3 circular15steps, top8. Configure current training batch20 Adam steps lr.2 at each epoch. Inactive Adam updates explicitly suppressed.'},
                  restrictions='No replay, test-label feedback, test-time mask optimization or supplied test task IDs. Known training boundaries allowed.',
                  limits='Short-budget adaptations, not native paper reproductions. FTN native labelled-batch recovery excluded from primary label-free evaluation. HAT shared-head single-Class-IL differs from native separate task-head objective. No equal-FLOP claim.',
                  references=['https://github.com/joansj/hat','https://arxiv.org/html/2407.10758','https://arxiv.org/html/2604.24637'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    raw=load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=np.random.default_rng(20261101);vi=np.concatenate([rng.permutation(np.flatnonzero(y.numpy()==c))[:500] for c in range(10)]);ti=np.setdiff1d(np.arange(len(y)),vi)
    assert not np.intersect1d(ti,vi).size
    np.savez(out/'split_indices.npz',train=ti,validation=vi)
    vx,vy=x[vi],y[vi];x,y=x[ti],y[ti]
    indices=[torch.where(y//2==g)[0] for g in range(5)];contexts=torch.stack([x[z].mean(0) for z in indices]);torch.save(contexts,out/'contexts.pt')
    trials=[];selected={}
    for arm in args.arms:
        for rate in protocol['rates'][arm]:
            model,trace=train(arm,113,rate,list(range(5)),x,y,vx,vy,indices)
            ids=torch.cdist(vx,contexts).argmin(1);before=statehash(model);pred=predict(model,vx,ids);assert before==statehash(model)
            value=float((pred==vy).float().mean());trials.append(dict(arm=arm,rate=rate,validation_accuracy=value,**trace))
            (out/'validation_trials.json').write_text(json.dumps(trials,indent=2));print('TUNE',arm,rate,round(value*100,2),flush=True)
        selected[arm]=max([r for r in trials if r['arm']==arm],key=lambda r:(r['validation_accuracy'],-r['rate']))['rate']
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    # All settings frozen before first final test.
    records=[]
    for seed in args.seeds:
        for ordername,order in [('forward',list(range(5))),('reverse',list(range(4,-1,-1)))]:
            for arm in args.arms:
                model,trace=train(arm,seed,selected[arm],order,x,y,vx,vy,indices)
                before=statehash(model);scores={}
                for mode,ids in [('prototype',torch.cdist(tx,contexts).argmin(1)),('oracle',ty//2),('current',torch.full_like(ty,order[-1]))]:
                    pred=predict(model,tx,ids);scores[mode]=float((pred==ty).float().mean())
                assert before==statehash(model)
                payload=sum(v.numel()*v.element_size() for v in model.state_dict().values())+contexts.numel()*contexts.element_size()
                row=dict(arm=arm,seed=seed,order=ordername,rate=selected[arm],scores=scores,parameters=sum(v.numel() for v in model.parameters()),retained_payload_bytes=payload,**trace)
                records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2));torch.save(model.state_dict(),out/f'{seed}_{ordername}_{arm}.pt')
                print('FINAL',seed,ordername,arm,{k:round(v*100,2) for k,v in scores.items()},flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
