"""From-scratch CIFAR pilot: branch allocation and genuinely compact sparse inputs.
No replay/pretraining/task identity at inference. Training uses known task boundaries.
Validation selects configurations; official test is evaluated after selection only.
"""
import json,time,argparse,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data import cifar10,cifar100

class Model(nn.Module):
    def __init__(self,classes,neurons,branches,sparse=False,plain=False):
        super().__init__();self.n=neurons;self.b=branches;self.plain=plain
        self.encoder=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Flatten())
        self.k=256 if sparse else 1024
        self.weight=nn.Parameter(torch.empty(neurons*branches,self.k));nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
        self.bias=nn.Parameter(torch.zeros(neurons*branches))
        if sparse:self.register_buffer('indices',torch.stack([torch.randperm(1024)[:self.k] for _ in range(neurons*branches)]))
        self.head=nn.Linear(neurons,classes);self.register_buffer('error',torch.full((neurons,branches),math.log(classes)))
    def responses(self,x):
        h=F.layer_norm(self.encoder(x),(1024,))
        z=(h[:,self.indices]*self.weight).sum(-1)+self.bias if hasattr(self,'indices') else F.linear(h,self.weight,self.bias)
        return z.reshape(-1,self.n,self.b)
    def forward(self,x,mode='trace',counter=False,routes=None):
        raw=self.responses(x);z=raw.tanh()
        if mode=='image':gate=raw.abs().argmax(-1)
        elif mode=='oracle':gate=routes
        else:gate=self.error.argmin(-1)[None].expand(len(x),-1)
        h=z.gather(2,gate[:,:,None]).squeeze(-1);logits=self.head(h)
        if counter:return logits,logits[:,None,None,:]+(z-h[:,:,None])[:,:,:,None]*self.head.weight.T[None,:,None,:]
        return logits

def norm(x):
    return (x.float()/255-torch.tensor([.4914,.4822,.4465])[None,:,None,None])/torch.tensor([.247,.2435,.2616])[None,:,None,None]

@torch.no_grad()
def evaluate(m,x,y,groups,routes=None):
    m.eval();pred=[];oracle=[]
    for ids in torch.arange(len(y)).split(128):
        bx=norm(x[ids]);pred.append(m(bx,'image').argmax(1))
        if routes is not None:oracle.append(m(bx,'oracle',routes=routes[y[ids]//groups]).argmax(1))
    p=torch.cat(pred);pairs=[float((p[y//groups==i]==y[y//groups==i]).float().mean()) for i in range(int(y.max())//groups+1)]
    return dict(accuracy=float((p==y).float().mean()),task_accuracy=pairs,oracle_accuracy=float((torch.cat(oracle)==y).float().mean()) if oracle else None)

def run(dataset,config,data,args):
    x,y,vx,vy,tx,ty=data;c=10 if dataset=='cifar10' else 100;g=2 if c==10 else 10;tasks=c//g
    n,b,sparse,plain=config;torch.manual_seed(args.seed);m=Model(c,n,b,sparse,plain);opt=torch.optim.Adam(m.parameters(),lr=.0005)
    routes=torch.zeros(tasks,n,dtype=torch.long);history=[];start=time.perf_counter()
    for stage in range(tasks):
        ix=torch.where(y//g==stage)[0]
        for epoch in range(args.epochs):
            order=ix[torch.randperm(len(ix),generator=torch.Generator().manual_seed(args.seed*10000+stage*100+epoch))]
            m.train()
            for ids in order.split(64):
                bx,by=norm(x[ids]),y[ids];opt.zero_grad(set_to_none=True)
                if plain:loss=F.cross_entropy(m(bx),by)
                else:
                    logits,alt=m(bx,counter=True);target=by[:,None,None].expand(alt.shape[:-1]);err=F.cross_entropy(alt.reshape(-1,c),target.reshape(-1),reduction='none').reshape(alt.shape[:-1]);loss=F.cross_entropy(logits,by)+.5*err.mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
                if not plain:
                    with torch.no_grad():m.error.mul_(.95).add_(err.detach().mean(0),alpha=.05)
        routes[stage]=m.error.argmin(-1)
        val=evaluate(m,vx,vy,g);history.append(val['task_accuracy']);print(dataset,config,'stage',stage,'val',round(val['accuracy']*100,2),flush=True)
    elapsed=time.perf_counter()-start;val=evaluate(m,vx,vy,g,routes)
    forget=np.mean([max(row[i] for row in history[i:])-history[-1][i] for i in range(tasks-1)])
    statebytes=sum(t.numel()*t.element_size() for t in m.state_dict().values());optimbytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
    name=('plain' if plain else 'sparse' if sparse else 'dense')+f'_n{n}_b{b}'
    result=dict(dataset=dataset,name=name,seed=args.seed,epochs=args.epochs,neurons=n,branches=b,sparse=sparse,parameters=sum(p.numel() for p in m.parameters()),inference_bytes=statebytes,optimizer_bytes=optimbytes,diagnostic_route_bytes=routes.numel()*routes.element_size(),training_seconds=elapsed,validation=val,forgetting=float(forget),validation_history=history)
    torch.save(dict(state=m.state_dict(),routes=routes,config=config),args.out/f'{dataset}_{name}.pt')
    return result,m,routes

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=101);p.add_argument('--epochs',type=int,default=2);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);records=[]
    for dataset in ['cifar10','cifar100']:
        if dataset=='cifar10':
            d=cifar10.load('data_cache/cifar10');data=[d['images'][d['train']],d['labels'][d['train']],d['images'][d['validation']],d['labels'][d['validation']],d['test_images'],d['test_labels']]
            configs=[(256,1,False,True),(64,4,False,False),(32,8,False,False),(64,8,False,False),(64,4,True,False),(64,8,True,False)]
        else:
            train,val,test,_=cifar100.load('data_cache/cifar100');data=[*train,*val,*test]
            configs=[(512,1,False,True),(64,8,False,False),(32,16,False,False),(64,16,False,False),(64,8,True,False),(64,16,True,False)]
        data=[torch.from_numpy(np.array(z,copy=True)).long() if i%2 else torch.from_numpy(np.array(z,copy=True)) for i,z in enumerate(data)]
        best=None
        for config in configs:
            r,m,routes=run(dataset,config,data,a);records.append(r);(a.out/'results.json').write_text(json.dumps(records,indent=2))
            if best is None or r['validation']['accuracy']>best[0]['validation']['accuracy']:best=(r,m,routes)
        r,m,routes=best;r['selected_test']=evaluate(m,data[4],data[5],2 if dataset=='cifar10' else 10,routes);(a.out/'results.json').write_text(json.dumps(records,indent=2));print('SELECTED',r,flush=True)
if __name__=='__main__':main()
