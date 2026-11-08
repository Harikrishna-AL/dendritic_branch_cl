"""Missing exemplar-free comparators under the existing two-epoch protocol.

PEC uses official neural modules. MIST is explicitly a paper reconstruction.
Covariance/tree comparators use task-0-only PCA50, charged as retained state.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import random
import statistics as st
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr
from scipy.linalg import cho_solve, cholesky
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from bdh_cl.scripts import continual_baseline_memory_suite as baseline
common=baseline.common
ROOT=common.ROOT
sys.path.insert(0,str(ROOT/'external/streaming_deps'))
from river import tree as river_tree
from datasketches import kll_floats_sketch


def official_pec(name):
    path=ROOT/'external/pec/backbone/pec_modules.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
    scope=dict(nn=nn,torch=torch,torchvision=torchvision,get_activation_from_name=lambda name:{'gelu':nn.GELU}[name])
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


class PEC(nn.Module):
    def __init__(self,seed):
        super().__init__()
        torch.manual_seed(seed)
        args=SimpleNamespace(pec_num_layers=2,pec_width=20,pec_teacher_width_multiplier=19.8,
                             pec_output_dim=99,pec_normalize_layers=True,pec_activation='gelu')
        config=dict(size=28,channels=1)
        cls=official_pec('PecMLP')
        self.teacher=cls(config,args,is_teacher=True).requires_grad_(False)
        self.students=nn.ModuleList([cls(config,args) for _ in range(10)])
        for student in self.students[1:]:student.load_state_dict(self.students[0].state_dict())

    def forward(self,x):
        target=self.teacher(x).detach()
        return -torch.stack([(student(x)-target).square().mean(1) for student in self.students],1)


def stream(indices,seed,order,epochs=2):
    for stage,group in enumerate(order):
        for epoch in range(epochs):
            gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            ix=indices[group]
            for ids in ix[torch.randperm(len(ix),generator=gen)].split(128):yield ids.numpy()


def fit_pec(x,y,indices,seed,order,config,epochs=2):
    model=PEC(seed)
    opts=[torch.optim.Adam(student.parameters(),lr=config['lr']) for student in model.students]
    start=time.perf_counter()
    presentations=0
    for ids in stream(indices,seed,order,epochs):
        bx,by=x[ids],y[ids]
        with torch.no_grad():target=model.teacher(bx)
        for c in by.unique().tolist():
            mask=by==c
            opt=opts[c]
            opt.zero_grad(set_to_none=True)
            loss=F.mse_loss(model.students[c](bx[mask]),target[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.students[c].parameters(),5.)
            opt.step()
        presentations+=len(ids)
    params=list(model.parameters())
    numeric=baseline.tensor_bytes(params)
    optimizer_bytes=sum(sum(baseline.tensor_bytes(v for v in state.values() if torch.is_tensor(v)) for state in opt.state.values()) for opt in opts)
    stats=dict(parameters=sum(p.numel() for p in params),trainable_parameters=sum(p.numel() for p in params if p.requires_grad),
               persistent_numeric_bytes=numeric,projection_bytes=0,optimizer_state_bytes=optimizer_bytes,
               current_presentations=presentations,training_seconds=time.perf_counter()-start,buffer_images=0,
               scope='Official PEC MLPs, shared frozen teacher counted once, 529145 total parameters; common Adam/batch128/two-epoch adaptation')
    return model,stats


@torch.no_grad()
def predict_pec(model,x):
    model.eval()
    return torch.cat([model(b).argmax(1) for b in x.split(256)])


class Projection:
    def __init__(self,first_samples):
        a=np.asarray(first_samples,dtype=np.float64)
        self.mean=a.mean(0)
        _,_,vt=np.linalg.svd(a-self.mean,full_matrices=False)
        self.components=vt[:50].copy()
    def __call__(self,x):return (np.asarray(x,dtype=np.float64)-self.mean)@self.components.T
    def nbytes(self):return self.mean.nbytes+self.components.nbytes


def load_checkpoint(path):
    """Load our local -m CLI archives with stable class-name resolution."""
    classes={name:globals()[name] for name in ['Projection','GaussianStats','RidgeStats','MistLeaf','MistBranch','Mist']}
    class LocalUnpickler(pickle.Unpickler):
        def find_class(self,module,name):
            if module=='__main__' and name in classes:return classes[name]
            return super().find_class(module,name)
    with Path(path).open('rb') as handle:return LocalUnpickler(handle).load()


class GaussianStats:
    def __init__(self,dimension=50):
        self.count=np.zeros(10,dtype=np.float64)
        self.mean=np.zeros((10,dimension),dtype=np.float64)
        self.m2=np.zeros((10,dimension,dimension),dtype=np.float64)
        self.precision=self.logdet=None
        self.mode=None
    def update(self,x,y):
        for c in np.unique(y):
            z=x[y==c]
            n=len(z);mu=z.mean(0);centered=z-mu
            delta=mu-self.mean[c];old=self.count[c];total=old+n
            self.m2[c]+=centered.T@centered+np.outer(delta,delta)*old*n/total
            self.mean[c]+=delta*n/total
            self.count[c]=total
    def prepare(self,mode,shrink):
        self.mode=mode
        d=self.mean.shape[1]
        if mode=='slda':
            cov=self.m2.sum(0)/max(self.count.sum()-np.count_nonzero(self.count),1)
            cov=cov[None]
        else:cov=self.m2/np.maximum(self.count[:,None,None]-1,1)
        cov=(1-shrink)*cov+shrink*np.eye(d)[None]
        factors=[cholesky(c,lower=True) for c in cov]
        self.precision=np.stack([cho_solve((factor,True),np.eye(d)) for factor in factors])
        self.logdet=np.array([2*np.log(np.diag(factor)).sum() for factor in factors])
    def predict(self,x):
        difference=x[:,None,:]-self.mean[None]
        precision=np.broadcast_to(self.precision,(10,)+self.precision.shape[-2:])
        distance=np.einsum('ncd,cde,nce->nc',difference,precision,difference,optimize=True)
        scores=-.5*(distance+self.logdet)+np.log(np.maximum(self.count,1)/max(self.count.sum(),1))
        scores[:,self.count==0]=-np.inf
        return scores.argmax(1)
    def nbytes(self):return sum(a.nbytes for a in [self.count,self.mean,self.m2,self.precision,self.logdet] if a is not None)


class RidgeStats:
    """Exact ridge/RLS sufficient statistics, no stored training examples."""
    def __init__(self,dimension=50):
        self.gram=np.zeros((dimension+1,dimension+1),dtype=np.float64)
        self.cross=np.zeros((dimension+1,10),dtype=np.float64)
        self.head=None
    def update(self,x,y):
        phi=np.column_stack([x,np.ones(len(x))])
        self.gram+=phi.T@phi
        self.cross+=phi.T@np.eye(10)[y]
    def prepare(self,ridge):self.head=np.linalg.solve(self.gram+ridge*np.eye(len(self.gram)),self.cross)
    def predict(self,x):return (np.column_stack([x,np.ones(len(x))])@self.head).argmax(1)
    def nbytes(self):return sum(a.nbytes for a in [self.gram,self.cross,self.head] if a is not None)


def gaussian_log_scores(x,count,mean,var):
    variance=np.maximum(var,1e-6)
    return -.5*np.sum(np.log(2*np.pi*variance)+(x-mean)**2/variance,axis=-1)+np.log(count+1)


class MistLeaf:
    def __init__(self,d=50,k=64):
        self.count=np.zeros(10,dtype=np.float64)
        self.mean=np.zeros((10,d),dtype=np.float64)
        self.m2=np.zeros((10,d),dtype=np.float64)
        self.sketches={}
        self.d,self.k=d,k
        self.since_attempt=0
    def __getstate__(self):
        state=dict(vars(self))
        state['sketches']={c:[s.serialize() for s in values] for c,values in self.sketches.items()}
        return state
    def __setstate__(self,state):
        state=dict(state)
        state['sketches']={c:[kll_floats_sketch.deserialize(s) for s in values] for c,values in state['sketches'].items()}
        self.__dict__.update(state)
    def update(self,x,y):
        old=self.count[y];new=old+1
        delta=x-self.mean[y]
        self.mean[y]+=delta/new
        self.m2[y]+=delta*(x-self.mean[y])
        self.count[y]=new
        if y not in self.sketches:self.sketches[y]=[kll_floats_sketch(self.k) for _ in range(self.d)]
        for j,s in enumerate(self.sketches[y]):s.update(float(x[j]))
        self.since_attempt+=1
    def variance(self):return self.m2/np.maximum(self.count[:,None]-1,1)
    def best_split(self,delta,tie):
        active=np.flatnonzero(self.count>0)
        if len(active)<2 or len(self.sketches)<2:return None
        n=self.count.sum();base=1-np.square(self.count/n).sum()
        candidates=[];variance=self.variance()
        for j in range(self.d):
            medians=sorted({float(s[j].get_quantile(.5)) for s in self.sketches.values() if s[j].n})
            best=None
            for threshold in [(a+b)/2 for a,b in zip(medians[:-1],medians[1:])]:
                left=np.zeros(10)
                for c in active:
                    if c in self.sketches and self.sketches[c][j].n:
                        fraction=self.sketches[c][j].get_rank(threshold)
                    else:
                        fraction=ndtr((threshold-self.mean[c,j])/math.sqrt(max(variance[c,j],1e-6)))
                    left[c]=fraction*self.count[c]
                right=self.count-left;nl=left.sum();nr=right.sum()
                if min(nl,nr)<1:continue
                gain=base-(nl/n)*(1-np.square(left/nl).sum())-(nr/n)*(1-np.square(right/nr).sum())
                if best is None or gain>best[0]:best=(gain,j,threshold)
            if best is not None:candidates.append(best)
        if len(candidates)<2:return None
        candidates.sort(reverse=True)
        m=max(len(active)-1,1)
        radius=math.sqrt(32*math.log(2*self.d*m/delta)/n)
        if candidates[0][0]-candidates[1][0]>radius or radius<tie:return candidates[0]
        return None
    def split(self,j,threshold,alpha):
        children=[]
        variance=self.variance()
        std=np.sqrt(np.maximum(variance[:,j],1e-6))
        z=(threshold-self.mean[:,j])/std
        density=np.exp(-.5*z*z)/math.sqrt(2*math.pi)
        for left in [True,False]:
            child=MistLeaf(self.d,self.k)
            mass=ndtr(z) if left else ndtr(-z)
            mass=np.clip(mass,1e-12,1)
            ratio=density/mass
            child.count=alpha*mass*self.count
            child.mean=self.mean.copy()
            child.mean[:,j]=self.mean[:,j]+(-1 if left else 1)*std*ratio
            childvar=variance.copy()
            childvar[:,j]=variance[:,j]*np.maximum(1+(-1 if left else 1)*z*ratio-ratio**2,1e-10)
            # Carry variance as a weighted prior; count can be fractional.
            child.m2=childvar*np.maximum(child.count[:,None]-1,1)
            children.append(child)
        return children
    def score(self,x,mode,epsilon=1e-5,beta=.5):
        result=gaussian_log_scores(x,self.count,self.mean,self.variance())
        result[self.count<=0]=-np.inf
        if mode=='mist_k':
            result=np.log(self.count+1)
            for c in np.flatnonzero(self.count>0):
                if c not in self.sketches:continue # Class prior until feature evidence arrives.
                for j,s in enumerate(self.sketches[c]):
                    if s.n<2:continue
                    r=s.get_rank(float(x[j]));lo=max(r-.25,0);hi=min(r+.25,1)
                    h=max(beta*(s.get_quantile(hi)-s.get_quantile(lo)),epsilon)
                    probability=(s.get_rank(float(x[j]+h))-s.get_rank(float(x[j]-h))+epsilon)/(2*h+epsilon)
                    result[c]+=math.log(max(probability,1e-300))
            result[self.count<=0]=-np.inf
        return result
    def nbytes(self):
        return self.count.nbytes+self.mean.nbytes+self.m2.nbytes+sum(len(s.serialize()) for values in self.sketches.values() for s in values)+24


class MistBranch:
    def __init__(self,feature,threshold,children):self.feature,self.threshold,self.children=feature,threshold,children


class Mist:
    def __init__(self,config):self.root=MistLeaf(k=config.get('k',64));self.config=config;self.splits=0
    def locate(self,x):
        node=self.root;parent=None;side=None
        while isinstance(node,MistBranch):
            parent=node;side=int(x[node.feature]>node.threshold);node=node.children[side]
        return node,parent,side
    def update(self,x,y):
        leaf,parent,side=self.locate(x)
        leaf.update(x,int(y))
        if leaf.since_attempt>=self.config.get('grace',200):
            leaf.since_attempt=0
            split=leaf.best_split(self.config.get('delta',.1),self.config.get('tie',.05))
            if split is not None:
                _,j,threshold=split
                branch=MistBranch(j,threshold,leaf.split(j,threshold,self.config.get('alpha',.6)))
                if parent is None:self.root=branch
                else:parent.children[side]=branch
                self.splits+=1
    def predict(self,x,mode='mist_g',epsilon=1e-5,beta=.5):
        return np.array([self.locate(row)[0].score(row,mode,epsilon,beta).argmax() for row in x])
    def nbytes(self):
        def visit(node):return 24+sum(visit(c) for c in node.children) if isinstance(node,MistBranch) else node.nbytes()
        return visit(self.root)+32
    def canonicalize_sketches(self):
        # DataSketches queries lazily sort level0. Normalize this representation
        # before checking that inference leaves the learned state unchanged.
        def visit(node):
            if isinstance(node,MistBranch):
                for child in node.children:visit(child)
            else:
                for values in node.sketches.values():
                    for sketch in values:
                        if sketch.n:sketch.get_quantile(.5)
        visit(self.root)


def river_numeric_bytes(obj,visited=None):
    """Logical numerical payload; Python object/allocator overhead separate."""
    if visited is None:visited=set()
    if obj is None or isinstance(obj,(str,bytes,type)):return 0
    if isinstance(obj,(bool,np.bool_)):return 1
    if isinstance(obj,(int,float,np.integer,np.floating)):return 8
    if id(obj) in visited:return 0
    visited.add(id(obj))
    if isinstance(obj,np.ndarray):return obj.nbytes
    if isinstance(obj,random.Random):return 0 # RNG state excluded consistently with older arms.
    if isinstance(obj,dict):return sum(river_numeric_bytes(k,visited)+river_numeric_bytes(v,visited) for k,v in obj.items())
    if isinstance(obj,(list,tuple,set)):return sum(river_numeric_bytes(v,visited) for v in obj)
    if hasattr(obj,'__dict__'):return river_numeric_bytes(vars(obj),visited)
    raise TypeError(f'Unaudited numeric state type: {type(obj)}')


def fit_classical(method,x,y,indices,seed,order,config,epochs=2):
    start=time.perf_counter()
    if method in ['slda','sqda']:model=GaussianStats()
    elif method=='rls':model=RidgeStats()
    elif method=='vfdt':
        model=river_tree.HoeffdingTreeClassifier(grace_period=config['grace'],split_criterion='gini',
            delta=config['delta'],tau=.05,leaf_prediction='nb',max_size=100.,memory_estimate_period=1000000)
    else:model=Mist(config)
    presentations=0
    for ids in stream(indices,seed,order,epochs):
        bx,by=x[ids],y[ids]
        if method in ['slda','sqda','rls']:model.update(bx,by)
        elif method=='vfdt':
            for row,label in zip(bx,by):model.learn_one(dict(enumerate(map(float,row))),int(label))
        else:
            for row,label in zip(bx,by):model.update(row,label)
        presentations+=len(ids)
    if method in ['slda','sqda']:model.prepare(method,config['shrinkage'])
    if method=='rls':model.prepare(config['ridge'])
    numerical=river_numeric_bytes(model) if method=='vfdt' else model.nbytes()
    stats=dict(persistent_numeric_bytes=numerical,model_numeric_bytes=numerical,optimizer_state_bytes=0,
               current_presentations=presentations,training_seconds=time.perf_counter()-start,buffer_images=0,
               projection_bytes=0,parameters=0,
               scope='PCA50 task-0-only transform; no replay; local statistical implementation' if method in ['slda','sqda','rls'] else
                     'River 0.26.1 Gaussian-splitter VFDT, NB leaves; no resets' if method=='vfdt' else
                     'MIST paper reconstruction; not official code or verified paper reproduction')
    if method=='vfdt':stats.update(nodes=model.n_nodes,python_model_heap_bytes=model._raw_memory_usage)
    if method=='mist':stats.update(nodes=2*model.splits+1,sketch_accounting='Serialized KLL payload plus FP64 moments and numeric tree fields; not native allocator RAM')
    return model,stats


def predict_classical(method,model,x,config):
    if method=='vfdt':return np.array([model.predict_one(dict(enumerate(map(float,row)))) for row in x])
    if method in ['mist_g','mist_k']:return model.predict(x,method,config.get('epsilon',1e-5),config.get('beta',.5))
    return np.concatenate([model.predict(b) for b in np.array_split(x,max(1,len(x)//512))])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--methods',nargs='+',default=['pec','slda','sqda','rls','vfdt','mist'])
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    configs=dict(pec=[dict(lr=lr) for lr in [.0005,.001,.003,.01]],
                 slda=[dict(shrinkage=s) for s in [1e-4,.001,.01]],
                 sqda=[dict(shrinkage=s) for s in [1e-4,.001,.01]],
                 rls=[dict(ridge=s) for s in [.1,1.,10.]],
                 vfdt=[dict(delta=d,grace=g) for d,g in [(.95,200),(1e-7,200),(.95,1000)]],
                 mist=[dict(k=64,grace=200,delta=.1,tie=.05,alpha=.6)])
    if args.smoke:configs={k:v[:1] for k,v in configs.items()}
    protocol=dict(configs=configs,methods=args.methods,smoke=args.smoke,epochs=2,batch=128,seeds=[101,102,103],orders=common.ORDERS,
                  selection='Seed113 forward validation only. Selections persisted before official-test inference.',
                  representation='PEC: standardized784 pixels. Classical/tree methods: PCA50 fitted on first1000 presented samples of initial task only; mean/components retained and charged.',
                  scope='Shared55k/5k/10k split,2epochs/task,no test task hints,no historical replay. This is not the strict single-pass leaderboard protocol.',
                  memory='Allocated/logical numerical payload. All independent neural parameters including shared frozen teacher once, covariances/caches, PCA, tree numeric fields and encoded KLL state. Python heap reported separately for VFDT; no universal peak-RAM claim.',
                  mist_reconstruction='Equations3–5 and Algorithms1–3, arXiv2605.11617v2. No verified public official code. Table tie=.05 used; split attempts each200 new leaf arrivals; confidence uses inherited+observed class mass; empty sketch split-CDF falls back to inherited Gaussian; empty MIST-K likelihood contributes class prior only. Numeric floors1e-6; fractional weighted priors; KLL C++ RNG not explicitly seedable.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    revision=subprocess.check_output(['git','-C',str(ROOT/'external/pec'),'rev-parse','HEAD'],text=True).strip()
    (out/'upstream.json').write_text(json.dumps(dict(pec_revision=revision,river=importlib.metadata.version('river'),
        datasketches=importlib.metadata.version('datasketches'),pec_sha256=hashlib.sha256((ROOT/'external/pec/backbone/pec_modules.py').read_bytes()).hexdigest()),indent=2))
    (out/'PEC_LICENSE').write_bytes((ROOT/'external/pec/LICENSE').read_bytes())
    raw=common.load_raw(str(ROOT/'data_cache/mnist'))
    x=(torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=(torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    split=np.load(ROOT/'results/matched_numenta_20261031/split_indices.npz')
    vx,vy=x[split['validation']],y[split['validation']];x,y=x[split['train']],y[split['train']]
    indices=[torch.nonzero(y//2==k,as_tuple=True)[0] for k in range(5)]
    if args.smoke:indices=[v[:128] for v in indices];vx,vy=vx[:200],vy[:200];tx,ty=tx[:200],ty[:200]
    cache={}
    def projected(seed,order):
        key=(seed,tuple(order))
        if key not in cache:
            first=np.concatenate(list(stream(indices,seed,order,epochs=1))[:8])[:1000]
            # Restrict the initial fitting block to task0 even in tiny smoke runs.
            first=first[np.isin(first,indices[order[0]].numpy())]
            projection=Projection(x[first].numpy())
            cache[key]=(projection,projection(x.numpy()),projection(vx.numpy()),projection(tx.numpy()))
        return cache[key]
    read=lambda name,default:json.loads((out/name).read_text()) if (out/name).exists() else default
    trials=read('validation_trials.json',[]);selected=read('selected_hyperparameters.json',{})
    for method in args.methods:
        if method in selected:continue
        for config in configs[method]:
            if any(r['method']==method and r['config']==config for r in trials):continue
            if method=='pec':
                model,stats=fit_pec(x,y,indices,113,common.ORDERS['forward'],config)
                pred=predict_pec(model,vx).numpy()
            else:
                projection,z,vz,_=projected(113,common.ORDERS['forward'])
                model,stats=fit_classical(method,z,y.numpy(),indices,113,common.ORDERS['forward'],config)
                if method=='mist':
                    # Calibrate only the sketch likelihood hyperparameters;
                    # the tree's training state is independent of predictor.
                    candidates=[]
                    for epsilon,beta in [(1.,.5),(1e-5,.5),(1e-5,1.)]:
                        cfg=dict(config,epsilon=epsilon,beta=beta)
                        candidate_accuracy=float((predict_classical('mist_k',model,vz,cfg)==vy.numpy()).mean())
                        candidates.append((candidate_accuracy,cfg))
                        trials.append(dict(method='mist_k',config=cfg,validation_accuracy=candidate_accuracy,**stats))
                    k_accuracy,k_config=max(candidates,key=lambda r:r[0])
                    selected['mist_k']=k_config
                    pred=predict_classical('mist_g',model,vz,config)
                else:pred=predict_classical(method,model,vz,config)
                stats['projection_bytes']=projection.nbytes();stats['persistent_numeric_bytes']+=projection.nbytes()
            accuracy=float((pred==vy.numpy()).mean())
            trials.append(dict(method=method,config=config,validation_accuracy=accuracy,**stats))
            (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
            print('TUNE',method,config,round(100*accuracy,2),round(stats['training_seconds'],2),flush=True)
        selected[method]=max((r for r in trials if r['method']==method),key=lambda r:r['validation_accuracy'])['config']
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    rows=read('results.json',[])
    for method in args.methods:
        for seed in ([101] if args.smoke else [101,102,103]):
            for order_name,order in common.ORDERS.items():
                if any(r['method']==('mist_g' if method=='mist' else method) and r['seed']==seed and r['order']==order_name for r in rows):continue
                config=selected[method]
                if method=='pec':
                    model,stats=fit_pec(x,y,indices,seed,order,config)
                    before=common.state_hash(model);pred=predict_pec(model,tx)
                    assert before==common.state_hash(model)
                    torch.save(model.state_dict(),out/f'pec_{seed}_{order_name}.pt')
                    predictions=[(method,pred,config)]
                else:
                    projection,z,_,tz=projected(seed,order)
                    model,stats=fit_classical(method,z,y.numpy(),indices,seed,order,config)
                    stats['projection_bytes']=projection.nbytes();stats['persistent_numeric_bytes']+=projection.nbytes()
                    modes=['mist_g','mist_k'] if method=='mist' else [method]
                    predictions=[]
                    if method=='mist':model.canonicalize_sketches()
                    before=pickle.dumps(model,protocol=5)
                    for mode in modes:
                        cfg=selected['mist_k'] if mode=='mist_k' else config
                        pred=torch.tensor(predict_classical(mode,model,tz,cfg),dtype=torch.long)
                        predictions.append((mode,pred,cfg))
                    assert before==pickle.dumps(model,protocol=5),'Evaluation changed learned state'
                    payload=pickle.dumps(dict(model=model,projection=projection),protocol=5)
                    stats['serialized_checkpoint_bytes']=len(payload)
                    (out/f'{method}_{seed}_{order_name}.pkl').write_bytes(payload)
                for mode,pred,cfg in predictions:
                    scores=common.metrics(pred,ty,order)
                    rows.append(dict(method=mode,seed=seed,order=order_name,config=cfg,scores=scores,**stats))
                    (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                    print('FINAL',mode,seed,order_name,round(100*scores['accuracy'],2),stats['persistent_numeric_bytes'],flush=True)
                # Keep only one transformed cohort at a time; statistics don't
                # retain any training/test input arrays or PCA fitting samples.
                cache.clear()
    for method in sorted({r['method'] for r in rows}):
        values=[st.mean(100*r['scores']['accuracy'] for r in rows if r['method']==method and r['seed']==seed) for seed in sorted({r['seed'] for r in rows if r['method']==method})]
        print('SUMMARY',method,st.mean(values),st.stdev(values) if len(values)>1 else 0,flush=True)


if __name__=='__main__':main()
