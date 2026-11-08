"""From-scratch CNN Split CIFAR-10 routing diagnostic and matched PEC control."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bdh_cl.data.cifar10 import load,MD5
from bdh_cl.scripts.split_mnist_usefulness import BranchClassifier
from bdh_cl.scripts.leaderboard_missing_comparison import official_pec

ROOT=Path(__file__).resolve().parents[2]
ORDERS={'forward':list(range(5)),'reverse':list(range(4,-1,-1))}
MEAN=torch.tensor([.4914,.4822,.4465])[None,:,None,None]
STD=torch.tensor([.2470,.2435,.2616])[None,:,None,None]


def normalize(x):return (x.float()/255-MEAN)/STD
def nbytes(values):return sum(v.numel()*v.element_size() for v in values)
def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class VisionBranch(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.encoder=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),
                                   nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Flatten())
        self.branch=BranchClassifier('local',inputs=1024,neurons=122,branches=4,gate='hard')
    def forward(self,x,counterfactual=False):return self.branch(self.encoder(x),counterfactual)
    def observe(self,error):self.branch.observe(error)
    def route_logits(self,x,routes):
        z=torch.tanh(self.branch.first(self.encoder(x))).reshape(-1,122,4)
        idx=routes.long()[None,:,:,None].expand(len(x),-1,-1,-1)
        hidden=z[:,None].expand(-1,len(routes),-1,-1).gather(3,idx).squeeze(3)
        return self.branch.head(hidden)


class VisionPEC(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        args=SimpleNamespace(pec_conv_layers=[(16,3,2),(32,3,2)],pec_teacher_width_multiplier=117/16,
                             pec_normalize_layers=True,pec_activation='gelu',pec_conv_reduce_spatial_to=2,
                             pec_output_dim=99)
        cls=official_pec('PecCNN');dataset=dict(size=32,channels=3)
        self.teacher=cls(dataset,args,is_teacher=True).requires_grad_(False)
        self.students=nn.ModuleList([cls(dataset,args) for _ in range(10)])
        for student in self.students[1:]:student.load_state_dict(self.students[0].state_dict())
    def forward(self,x):
        target=self.teacher(x).detach()
        return -torch.stack([(student(x)-target).square().mean(1) for student in self.students],1)


class ContextRidge:
    """Online boundary-supervised context router on stable random image features."""
    def __init__(self,seed):
        g=torch.Generator().manual_seed(seed+4000)
        self.projection=torch.randn(192,128,generator=g)/math.sqrt(192)
        self.gram=torch.zeros(129,129);self.cross=torch.zeros(129,5)
    def features(self,x):
        pooled=F.avg_pool2d(x,4).flatten(1)
        return torch.cat([torch.tanh(pooled@self.projection),torch.ones(len(x),1)],1)
    def observe(self,x,group):
        features=self.features(x)
        self.gram.add_(features.T@features)
        targets=F.one_hot(torch.full((len(x),),group,dtype=torch.long),5).float()
        self.cross.add_(features.T@targets)
    def solve(self,ridge):return torch.linalg.solve(self.gram+ridge*torch.eye(129),self.cross)
    def state(self):return dict(projection=self.projection,gram=self.gram,cross=self.cross)
    def nbytes(self,head=None):return nbytes([self.projection,self.gram,self.cross]+([head] if head is not None else []))


def batches(indices,seed,stage,epoch):
    gen=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
    return indices[torch.randperm(len(indices),generator=gen)].split(128)


def optimizer_bytes(opt):
    return sum(nbytes(v for v in state.values() if torch.is_tensor(v)) for state in opt.state.values())


@torch.no_grad()
def current_predictions(model,x):
    model.eval()
    return torch.cat([model(normalize(b)).argmax(1) for b in x.split(256)])


@torch.no_grad()
def evaluate_branch(model,x,y,contexts,routes,router,settings):
    model.eval();before=state_hash(model)
    head=router.solve(settings.get('ridge',1.))
    outputs=[];prototype=[];ridge_ids=[];current=[]
    for raw in x.split(256):
        bx=normalize(raw)
        outputs.append(model.route_logits(bx,routes))
        distance=contexts.square().sum(1)[None]-2*bx.flatten(1)@contexts.T
        prototype.append(distance.argmin(1));ridge_ids.append((router.features(bx)@head).argmax(1))
        current.append(model(bx).argmax(1))
    logits=torch.cat(outputs);proto=torch.cat(prototype);learned=torch.cat(ridge_ids)
    ids=torch.arange(len(y));group=y//2;route_predictions=logits.argmax(2)
    logprob=(logits/settings.get('temperature',1.)).log_softmax(2)
    entropy=-(logprob.exp()*logprob).sum(2)
    entropy_ids=entropy.argmin(1)
    top=logits.topk(2,dim=2).values;margin_ids=(top[:,:,0]-top[:,:,1]).argmax(1)
    random_ids=torch.randint(5,(len(y),),generator=torch.Generator().manual_seed(9191))
    predicted=dict(current=torch.cat(current),prototype=route_predictions[ids,proto],
                   learned_context=route_predictions[ids,learned],entropy=route_predictions[ids,entropy_ids],
                   margin=route_predictions[ids,margin_ids],oracle=route_predictions[ids,group],
                   wrong_route=route_predictions[ids,(group+1)%5],random_route=route_predictions[ids,random_ids])
    scores={name:dict(accuracy=float((pred==y).float().mean()),pair_accuracy=[float((pred[y//2==k]==y[y//2==k]).float().mean()) for k in range(5)]) for name,pred in predicted.items()}
    scores['bank_any_correct']=dict(accuracy=float((route_predictions==y[:,None]).any(1).float().mean()))
    scores['context_pair_accuracy']={name:float((chosen==group).float().mean()) for name,chosen in [('prototype',proto),('learned',learned),('entropy',entropy_ids),('margin',margin_ids)]}
    assert state_hash(model)==before
    return scores


def fit_branch(arm,seed,rate,order,x,y,vx,vy,indices,epochs=5,joint=False):
    model=VisionBranch(seed);opt=torch.optim.Adam(model.parameters(),lr=rate)
    router=ContextRidge(seed);contexts=torch.zeros(5,3072);routes=torch.zeros(5,122,dtype=torch.uint8)
    acquisition=[];start=time.perf_counter();steps=clipped=0
    groups=[None] if joint else order
    for stage,group in enumerate(groups):
        ix=torch.cat(indices) if joint else indices[group]
        if not joint:
            # Same boundary privilege as the MNIST prototype router: only the
            # current task's images contribute to its context; never test labels.
            total=torch.zeros(3072)
            for ids in ix.split(256):total+=normalize(x[ids]).flatten(1).sum(0)
            contexts[group]=total/len(ix)
        for epoch in range(epochs):
            model.train()
            for ids in batches(ix,seed,stage,epoch):
                bx,by=normalize(x[ids]),y[ids]
                opt.zero_grad(set_to_none=True)
                if arm=='aux':
                    logits,candidates=model(bx,True)
                    targets=by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                    errors=F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1])
                    feedback=errors.detach().mean(0);loss=F.cross_entropy(logits,by)+.5*errors.mean()
                else:
                    with torch.no_grad():
                        _,candidates=model(bx,True)
                        targets=by[:,None,None].expand(candidates.shape[:-1]).reshape(-1)
                        feedback=F.cross_entropy(candidates.reshape(-1,10),targets,reduction='none').reshape(candidates.shape[:-1]).mean(0)
                    loss=F.cross_entropy(model(bx),by)
                assert torch.isfinite(loss)
                loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.)
                clipped+=int(norm>5);opt.step();model.observe(feedback);steps+=1
                if not joint:router.observe(bx,group)
        if not joint:
            routes[group]=model.branch.error.argmin(1).to(torch.uint8)
            mask=vy//2==group
            acquisition.append(dict(group=group,validation_accuracy=float((current_predictions(model,vx[mask])==vy[mask]).float().mean())))
            print('PHASE',arm,seed,'joint' if joint else group,round(100*acquisition[-1]['validation_accuracy'],2),flush=True)
    stats=dict(parameters=sum(p.numel() for p in model.parameters()),model_bytes=nbytes(model.state_dict().values()),
               optimizer_state_bytes=optimizer_bytes(opt),route_bytes=nbytes([routes]),prototype_bytes=nbytes([contexts]),
               router_bytes=router.nbytes(router.solve(1.)) if not joint else 0,
               training_seconds=time.perf_counter()-start,current_presentations=epochs*sum(len(ix) for ix in indices),steps=steps,clipped_steps=clipped,
               acquisition=acquisition,joint=joint)
    stats['prototype_persistent_bytes']=stats['model_bytes']+stats['route_bytes']+stats['prototype_bytes']
    stats['learned_router_persistent_bytes']=stats['model_bytes']+stats['route_bytes']+stats['router_bytes']
    return model,contexts,routes,router,stats


def fit_pec(seed,rate,order,x,y,indices,epochs=5):
    model=VisionPEC(seed);opts=[torch.optim.Adam(student.parameters(),lr=rate) for student in model.students]
    start=time.perf_counter();clipped=steps=0
    for stage,group in enumerate(order):
        ix=indices[group]
        # The frozen teacher and per-image InstanceNorm are deterministic here
        # (no augmentation). Reuse only CURRENT-task targets across its epochs;
        # discard this temporary cache when the task ends. No old replay.
        with torch.no_grad():
            target_cache=torch.cat([model.teacher(normalize(x[ids])) for ids in ix.split(128)])
        for epoch in range(epochs):
            for positions in batches(torch.arange(len(ix)),seed,stage,epoch):
                ids=ix[positions]
                bx,by=normalize(x[ids]),y[ids]
                target=target_cache[positions]
                for c in by.unique().tolist():
                    mask=by==c;opt=opts[c];opt.zero_grad(set_to_none=True)
                    loss=F.mse_loss(model.students[c](bx[mask]),target[mask]);loss.backward()
                    norm=nn.utils.clip_grad_norm_(model.students[c].parameters(),5.);clipped+=int(norm>5);opt.step()
                steps+=1
        del target_cache
        print('PHASE pec',seed,group,flush=True)
    return model,dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        model_bytes=nbytes(model.state_dict().values()),optimizer_state_bytes=sum(optimizer_bytes(opt) for opt in opts),
        training_seconds=time.perf_counter()-start,current_presentations=epochs*sum(len(ix) for ix in indices),
        frozen_teacher_presentations=sum(len(ix) for ix in indices),steps=steps,clipped_steps=clipped)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--methods',nargs='+',default=['aux','ce','pec','joint_aux','joint_ce'])
    p.add_argument('--stage',choices=['all','tune','final'],default='all')
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    data=load(ROOT/'data_cache/cifar10');x=torch.tensor(data['images'][data['train']].copy());y=torch.tensor(data['labels'][data['train']])
    vx=torch.tensor(data['images'][data['validation']].copy());vy=torch.tensor(data['labels'][data['validation']])
    tx=torch.tensor(data['test_images'].copy());ty=torch.tensor(data['test_labels'])
    indices=[torch.nonzero(y//2==k,as_tuple=True)[0] for k in range(5)]
    epochs=1 if args.smoke else args.epochs
    if args.smoke:
        indices=[ix[:128] for ix in indices]
        vi=torch.cat([torch.nonzero(vy==c,as_tuple=True)[0][:10] for c in range(10)])
        ti=torch.cat([torch.nonzero(ty==c,as_tuple=True)[0][:10] for c in range(10)])
        vx,vy=vx[vi],vy[vi];tx,ty=tx[ti],ty[ti]
    rates=[.001] if args.smoke else [.0005,.001,.003]
    pec_rates=[.001] if args.smoke else [.0005,.001,.003,.01]
    configs={name:(pec_rates if name=='pec' else rates) for name in args.methods}
    protocol=dict(dataset='Official CIFAR-10, MD5 '+MD5,split='45000 fit /5000 validation (500/class), RNG20261102 /official10000 test',
        pairs=[[2*k,2*k+1] for k in range(5)],orders=ORDERS,seeds=[201,202,203],tuning_seed=213,epochs=epochs,batch128=128,
        optimizer='Continuous Adam, clipping5; no reset between tasks; no augmentation or external pretraining; CPU2threads',
        normalization='Standard RGB channel means[.4914,.4822,.4465], std[.2470,.2435,.2616]',
        branch='Conv3->16->32->64, all3x3 stride2 padding1 ReLU, flatten1024; 122neurons x4tanh branches; shared10-way head; errorEMA.95, hard argmin gate. Auxiliary=.5 mean single-neuron branch-substitution CE.',
        ce_control='Identical architecture/error feedback/archives but no auxiliary-loss gradient. Not an ordinary fine-tuning CNN.',
        routers='Phase-end branch indices saved per task. Prototype: full standardized input task-mean nearest distance. Learned: fixed128random tanh features of8x8pooled RGB, online ridge predicting training phase; uses boundary labels, no input replay. Entropy/margin: compare all5routes, all10logits eligible. Current global gate/random/cyclicwrong controls.',
        oracle='True class-pair selects historical route only as diagnostic. All10outputclasses eligible. bank_any_correct is stronger label-informed best-of5 correctness diagnostic; neither is deployable.',
        training_boundary_privilege='Both archived route acquisition and context targets use known training boundaries. This does not claim fully unsupervised task discovery.',
        tuning='Each method independently selects LR maximizing final validation prototype accuracy (branches) or plain accuracy (PEC/joint). Then choose learned-router ridge .1/1/10 and entropy temp .5/1/2 using same validation run. All selections saved before final-test evaluation.',
        joint='Offline joint reference: same225000sample presentations at5epochs, allclassesmixed; separatelyLRtuned; not a continual-learning result.',
        pec='Verbatim official PecCNN modules: students16/32channels, sharedteacher117/234channels, GELU/InstanceNorm, adaptive2x2pool, output99. Count all10students+sharedfrozenteacher once. Frozen per-image teacher targets cached only for current task epochs then discarded, no old replay. Different intrinsic architecture; totalparameter matching, notequalFLOPs.',
        memory='Allocated numerical model/EMA + route/context/router tensors. Adam extra; no peakRAM/systemallocator/datasetcache/temporaryactivation claim.',
        configs=configs,smoke=args.smoke,execution_stage=args.stage)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    np.savez(out/'split_indices.npz',train=data['train'],validation=data['validation'])
    read=lambda name,default:json.loads((out/name).read_text()) if (out/name).exists() else default
    trials=read('validation_trials.json',[]);selected=read('selected_hyperparameters.json',{})
    if args.stage=='final':assert all(method in selected for method in args.methods),'Freeze validation selections before final evaluation'
    for method in args.methods:
        if method in selected:continue
        for rate in configs[method]:
            if any(r['method']==method and r['lr']==rate for r in trials):continue
            if method=='pec':
                model,stats=fit_pec(213,rate,ORDERS['forward'],x,y,indices,epochs)
                accuracy=float((current_predictions(model,vx)==vy).float().mean());settings={}
            else:
                joint=method.startswith('joint_');arm=method.removeprefix('joint_')
                model,contexts,routes,router,stats=fit_branch(arm,213,rate,ORDERS['forward'],x,y,vx,vy,indices,epochs,joint)
                if joint:accuracy=float((current_predictions(model,vx)==vy).float().mean());settings={}
                else:
                    ridge_trials=[]
                    for ridge in [.1,1.,10.]:
                        score=evaluate_branch(model,vx,vy,contexts,routes,router,dict(ridge=ridge))
                        ridge_trials.append(dict(ridge=ridge,accuracy=score['learned_context']['accuracy']))
                    ridge=max(ridge_trials,key=lambda r:r['accuracy'])['ridge']
                    entropy_trials=[]
                    for temperature in [.5,1.,2.]:
                        score=evaluate_branch(model,vx,vy,contexts,routes,router,dict(ridge=ridge,temperature=temperature))
                        entropy_trials.append(dict(temperature=temperature,accuracy=score['entropy']['accuracy']))
                    temperature=max(entropy_trials,key=lambda r:r['accuracy'])['temperature']
                    settings=dict(ridge=ridge,temperature=temperature,ridge_trials=ridge_trials,entropy_trials=entropy_trials)
                    accuracy=score['prototype']['accuracy']
            trials.append(dict(method=method,lr=rate,validation_accuracy=accuracy,router_settings=settings,**stats))
            (out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
            print('TUNE',method,rate,round(100*accuracy,2),round(stats['training_seconds'],1),flush=True)
        best=max((r for r in trials if r['method']==method),key=lambda r:r['validation_accuracy'])
        selected[method]=dict(lr=best['lr'],**best['router_settings'])
    (out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    if args.stage=='tune':
        print('TUNING COMPLETE',args.methods,flush=True)
        return
    rows=read('results.json',[])
    for method in args.methods:
        for seed in ([201] if args.smoke else [201,202,203]):
            names=['joint'] if method.startswith('joint_') else list(ORDERS)
            for name in names:
                if any(r['method']==method and r['seed']==seed and r['order']==name for r in rows):continue
                config=selected[method];order=ORDERS['forward'] if name=='joint' else ORDERS[name]
                if method=='pec':
                    model,stats=fit_pec(seed,config['lr'],order,x,y,indices,epochs)
                    before=state_hash(model);pred=current_predictions(model,tx);assert before==state_hash(model)
                    scores=dict(autonomous=dict(accuracy=float((pred==ty).float().mean())))
                    state=dict(model=model.state_dict())
                else:
                    joint=method.startswith('joint_');arm=method.removeprefix('joint_')
                    model,contexts,routes,router,stats=fit_branch(arm,seed,config['lr'],order,x,y,vx,vy,indices,epochs,joint)
                    if joint:
                        before=state_hash(model);pred=current_predictions(model,tx);assert before==state_hash(model)
                        scores=dict(autonomous=dict(accuracy=float((pred==ty).float().mean())))
                    else:scores=evaluate_branch(model,tx,ty,contexts,routes,router,config)
                    state=dict(model=model.state_dict(),contexts=contexts,routes=routes,router=router.state(),
                               router_head=router.solve(config.get('ridge',1.)))
                torch.save(state,out/f'{method}_{seed}_{name}.pt')
                rows.append(dict(method=method,seed=seed,order=name,config=config,scores=scores,**stats))
                (out/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
                print('FINAL',method,seed,name,{k:round(100*v['accuracy'],2) for k,v in scores.items() if 'accuracy' in v},flush=True)
    print('COMPLETE',len(rows),'runs',flush=True)


if __name__=='__main__':main()
