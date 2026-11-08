"""Matched trainable-CNN branch-configuration vs HAT experiment.

Uses official DCNet losses/augmentation/optimizers/readout for both arms.
CNN architecture, initialization and CNN HAT protection are shared. Only
branch allocation/gating/protection differs. No replay or test-time task IDs.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import resource
import statistics as st
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from bdh_cl.data.cifar100 import load

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'external/DCNet'
DEFAULT_OUT=ROOT/'results/cnn_branch_comparison_20261027'
SIZES={'cnn16':(16,32,64),'cnn32':(32,64,128)}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def tensor_bytes(obj):
    if torch.is_tensor(obj):return obj.numel()*obj.element_size()
    if isinstance(obj,dict):return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj,(tuple,list)):return sum(tensor_bytes(v) for v in obj)
    return 0


def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Logger:
    def print(self,*values):print(*values,flush=True)


def load_official():
    sys.path.insert(0,str(UPSTREAM));os.chdir(UPSTREAM)
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
    from models.base_model import BaseModel
    return BaseModel


def model_class(BaseModel):
    class MatchedModel(BaseModel):
        def __init__(self,P,channels,arm,seed):
            P.embedding_dim=128
            super().__init__(128,P,num_classes=10)
            self.channels=channels;self.arm=arm;self.width=128;self.branches=10
            # All shared weights/gates initialize identically across arms.
            with torch.random.fork_rng():
                torch.manual_seed(seed+10000)
                self.convs=nn.ModuleList()
                previous=3
                for channel in channels:
                    self.convs.append(nn.Conv2d(previous,channel,3,stride=2,padding=1,bias=True));previous=channel
                self.branch=nn.Linear(previous,1280)
                self.encoder_ec=nn.ModuleList([nn.ParameterList([nn.Parameter(torch.randn(1,c)) for _ in range(10)]) for c in channels])
            if arm=='dcnet_hat':
                with torch.random.fork_rng():
                    torch.manual_seed(seed+20000)
                    self.branch_ec=nn.ParameterList([nn.Parameter(torch.randn(1,1280)) for _ in range(10)])
            else:
                self.register_buffer('routes',torch.full((10,128),-1,dtype=torch.long))
                self.register_buffer('used',torch.zeros(128,10,dtype=torch.bool))

        def encoder_masks(self,task,s):
            return [torch.sigmoid(s*layer[task]) for layer in self.encoder_ec]

        def encoder(self,task,x,s):
            masks=self.encoder_masks(task,s);x=(x-.5)/.5
            for conv,mask in zip(self.convs,masks):x=F.relu(conv(x))*mask.reshape(1,-1,1,1)
            return x.mean((2,3)),masks

        def branch_mask(self,task,s):
            if self.arm=='dcnet_hat':return torch.sigmoid(s*self.branch_ec[task])
            assert bool((self.routes[task]>=0).all())
            return F.one_hot(self.routes[task],10).float().flatten().reshape(1,-1)

        def penultimate(self,task,inputs,s,all_features=False):
            x,masks=self.encoder(task,inputs,s)
            branches=torch.tanh(self.branch(x)).reshape(-1,128,10)
            mask=self.branch_mask(task,s)
            return (branches*mask.reshape(1,128,10)).sum(-1),masks+[mask]

        @torch.no_grad()
        def allocate(self,task,x):
            if self.arm=='dcnet_hat':return
            f,_=self.encoder(task,x,1.)
            response=self.branch(f).tanh().reshape(-1,128,10)
            route=response.square().mean(0).masked_fill(self.used,-torch.inf).argmax(-1)
            self.routes[task].copy_(route);self.used[torch.arange(128,device=x.device),route]=True

    return MatchedModel


def masks_after_task(P,model,previous):
    current={f'encoder{i}':v.detach() for i,v in enumerate(model.encoder_masks(P.t,P.smax))}
    current['branch']=model.branch_mask(P.t,P.smax).detach()
    current['projection']=model.simclr_layer.mask(P.t,P.smax).detach()
    cumulative={k:v if previous is None else torch.maximum(previous[k],v) for k,v in current.items()}
    back={}
    for i,conv in enumerate(model.convs):
        post=cumulative[f'encoder{i}'].flatten()
        if i==0:weight=1-post[:,None,None,None]
        else:
            pre=cumulative[f'encoder{i-1}'].flatten()
            weight=1-torch.minimum(post[:,None],pre[None,:])[:,:,None,None]
        back[f'convs.{i}.weight']=weight;back[f'convs.{i}.bias']=1-post
    post=cumulative['branch'].flatten();pre=cumulative['encoder2'].flatten()
    # Config arm freezes previously assigned branch rows exactly. HAT arm
    # uses standard input/output gate connection protection.
    back['branch.weight']=1-post[:,None] if model.arm=='branch_config' else 1-torch.minimum(post[:,None],pre[None,:])
    back['branch.bias']=1-post
    pre=post.reshape(128,10).max(-1).values;post=cumulative['projection'].flatten()
    back['simclr_layer.fc1.weight']=1-torch.minimum(post[:,None],pre[None,:])
    back['simclr_layer.fc1.bias']=1-post
    parameters=dict(model.named_parameters())
    # Materialize masks to match convolution gradients on MPS.
    back={name:mask.expand_as(parameters[name]).contiguous() for name,mask in back.items()}
    return cumulative,back


@torch.no_grad()
def predictions(model,x,stages,smax):
    # Every candidate task is evaluated. Labels/true task IDs are absent.
    joint=[];cosine=[]
    for task in range(stages):
        j=0;c=0
        for rotation in range(4):
            _,aux,_=model(task,torch.rot90(x,rotation,(2,3)),s=smax,joint=True,penultimate=True)
            j=j+aux['joint'][:,10*rotation:10*(rotation+1)]/4
            basis=model.prototypes[40*task+10*rotation:40*task+10*(rotation+1)]
            c=c+F.normalize(aux['feature'],dim=1)@basis.T/4
        joint.append(j);cosine.append(c)
    return {'joint':torch.cat(joint,1),'prototype_cosine':torch.cat(cosine,1)}


@torch.no_grad()
def evaluate(model,x,y,stages,P,device,batch_size=64):
    before=state_digest(model);model.eval()
    totals={name:{'correct':0,'route':0,'oracle':0,'task_correct':[0]*stages} for name in ['joint','prototype_cosine']}
    counts=[int((y//10==t).sum()) for t in range(stages)]
    for start in range(0,len(y),batch_size):
        by=y[start:start+batch_size];scores=predictions(model,x[start:start+batch_size].to(device),stages,P.smax)
        for name,score in scores.items():
            assert bool(torch.isfinite(score).all())
            score=score.cpu();pred=score.argmax(-1);v=totals[name]
            v['correct']+=int((pred==by).sum());v['route']+=int((pred//10==by//10).sum())
            oracle=score.reshape(-1,stages,10)[torch.arange(len(by)),by//10].argmax(-1)+10*(by//10)
            v['oracle']+=int((oracle==by).sum())
            for task in range(stages):v['task_correct'][task]+=int(((pred==by)&(by//10==task)).sum())
    assert state_digest(model)==before
    return {name:{'accuracy':v['correct']/len(y),'routing_accuracy':v['route']/len(y),'oracle_task_diagnostic':v['oracle']/len(y),'task_accuracy':[a/b for a,b in zip(v['task_correct'],counts)]} for name,v in totals.items()}


def shared_signature(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items():
        if name.startswith(('branch_ec','routes','used')):continue
        h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def verify(Model,parse_args,channels,seed,device):
    models=[]
    for arm in ['branch_config','dcnet_hat']:
        torch.manual_seed(seed);P=parse_args(default=True);P.n_tasks=10;P.n_cls_per_task=10
        models.append(Model(P,channels,arm,seed).to(device))
    assert shared_signature(models[0])==shared_signature(models[1])
    x=torch.rand(16,3,32,32,device=device)
    for model in models:
        model.allocate(0,x)
        before=state_digest(model);feature,masks=model.penultimate(0,x,1.)
        assert feature.shape==(16,128) and len(masks)==4
        with torch.no_grad():
            h=(x-.5)/.5
            for i,conv in enumerate(model.convs):h=F.relu(conv(h))*model.encoder_masks(0,1.)[i].reshape(1,-1,1,1)
            b=model.branch(h.mean((2,3))).tanh().reshape(-1,128,10)
            manual=(b*model.branch_mask(0,1.).reshape(1,128,10)).sum(-1)
            assert torch.allclose(manual,feature,atol=1e-6)
        assert state_digest(model)==before
        P=model.P;P.t=0;cum,back=masks_after_task(P,model,None)
        for name,param in model.named_parameters():
            if name in back:assert torch.broadcast_shapes(param.shape,back[name].shape)==param.shape
        assert any(p.requires_grad for p in model.convs.parameters())
        # Verify a real update changes the shared CNN (not a frozen encoder).
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        old=model.convs[0].weight.detach().clone();optimizer.zero_grad();feature.square().mean().backward();optimizer.step()
        assert not torch.equal(old,model.convs[0].weight)
        if model.arm=='branch_config':
            first=model.routes[0].clone();model.allocate(1,x);assert (first!=model.routes[1]).all()
    del models
    if device.type=='mps':torch.mps.empty_cache()
    return {'shared_CNN_branch_projection_and_head_initialization_equal':True,'forward_matches_direct_calculation':True,'masks_broadcast_correctly':True,'CNN_weights_change_under_training':True,'configuration_routes_disjoint':True}


def summary_report(root):
    protocol=json.loads((root/'protocol.json').read_text());rows=[];failed=[];progress=[]
    for path in sorted(root.glob('*/result.json')):rows.append(json.loads(path.read_text()))
    for path in sorted(root.glob('*/failed.json')):failed.append({'run':path.parent.name,**json.loads(path.read_text())})
    for path in sorted(root.glob('*/progress.json')):
        if not (path.parent/'result.json').exists():progress.append({'run':path.parent.name,**json.loads(path.read_text())})
    complete=(root/'completion.json').exists()
    status=json.loads((root/'status.json').read_text()).get('status','running') if (root/'status.json').exists() else 'running'
    report=ROOT/'reports/cnn_branch_comparison_20261027/REPORT.md';report.parent.mkdir(parents=True,exist_ok=True)
    lines=['# Matched trainable small-CNN continual-learning comparison','', 'Status: '+('finished' if complete else status)+f'; {len(rows)} complete runs, {len(failed)} failures.', '',
        'Question: with the same trainable CNN, branch feature layer, losses, augmentations, optimizer and task heads, does saved disjoint branch configuration differ from learned HAT branch gating?', '',
        f"CNNs: 16–32–64 and 32–64–128 channels; three stride-2 3×3 convolutions, ReLU, task masks and global average pooling. No normalization running statistics, external pretraining or frozen shared encoder. Both models have 128 groups × 10 tanh branches after the CNN. Seeds {protocol['seeds']}; one fixed class order. {protocol['feature_epochs']} representation + {protocol['ood_epochs']} classifier epochs per task; batch {protocol['batch_size']}.", '',
        'Training uses known task boundaries, only current-class data, and no old-example/latent/generated replay. Inference evaluates every learned task configuration and predicts across all seen classes without a supplied true task ID. Official DCNet rotation-ensemble logits are the primary readout; prototype-cosine scores are a predeclared secondary readout, not selected on test accuracy.', '',
        '| CNN | Method | Seeds finished | Readout | Incremental avg (%) | Final (%) | Oracle diagnostic (%) | Continuation MB |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for size in SIZES:
        for arm in ['branch_config','dcnet_hat']:
            group=[r for r in rows if r['size']==size and r['arm']==arm]
            if not group:continue
            for rule in ['joint','prototype_cosine']:
                final=[100*r['history'][-1][rule]['accuracy'] for r in group];avg=[100*r['average_incremental'][rule] for r in group];oracle=[100*r['history'][-1][rule]['oracle_task_diagnostic'] for r in group]
                sd=f' ± {st.stdev(final):.2f}' if len(final)>1 else ''
                lines.append(f'| {size} | {arm} | {len(group)} | {rule} | {st.mean(avg):.2f} | {st.mean(final):.2f}{sd} | {st.mean(oracle):.2f} | {st.mean(r["continuation_bytes"] for r in group)/1e6:.3f} |')
    lines += ['', '## Exact comparison and limits','',
        'The branch_config arm saves one unused branch index per group at each known task boundary and freezes its previous branch rows. The dcnet_hat arm learns task-specific sigmoid branch masks and protects connections with HAT. Both use identical CNN HAT masks/protection formulas, task-specific DCNet projection heads, IOE/DAC losses, LARC/SGD, current-task augmentation and classifier training. This extends the branch mechanism into the DCNet training/readout framework; it is not the earlier fixed-code/Adam raw-pixel pilot.', '',
        'These are small-CNN architectural variants. They do not reproduce published DCNet’s ResNet/700+100-epoch result. Different effective gating and readout-module parameter counts are reported; the CNN, branch-layer dimensions, shared initialization, examples, optimizer and presentation budgets are matched. Hard branch configurations and soft HAT masks have different active-computation patterns. Historical CNN feature protection is HAT-style and selective, not a guarantee that all old feature values remain exactly invariant.', '',
        'The CNN is trainable during representation learning and fixed during current-task classifier fitting, matching the two-phase protocol. All evaluation is read-only. The oracle columns use true tasks only after scores are computed, for diagnostics; they are not task-ID-free accuracy.', '',
        'Continuation bytes count actual model tensors, cumulative/update masks, retained class-direction tensors and concentration summaries, excluding datasets/current activations and optimizer workspaces. Full contiguous update masks are used for reliable MPS convolution updates and counted. They could be reconstructed from cumulative gates to reduce memory; the reported footprint is implementation-specific. Process peak RSS includes Python, cached dataset, allocator and runtime; it is not pure model RAM. Each run is a fresh process so peak RSS is attributable to that process. Run wall time and phase times are retained in JSON, not asserted to be equal FLOPs.', '',
        'Fit/unused-validation/test counts: 45,000/5,000/10,000 on the official cached CIFAR-100 archive; same fixed 10×10 class order as prior experiments. No validation or test selection. Native data-independent DCNet class directions are reused; no pretrained image representation. Model sizes/budget chosen from hardware profiling only.', '',
        '## Current progress','', '```json',json.dumps(progress,indent=2),'```', '', '## Failures','', '```json',json.dumps(failed,indent=2),'```', '',
        '[Protocol](../../results/cnn_branch_comparison_20261027/protocol.json) · [Results directory](../../results/cnn_branch_comparison_20261027) · [Source](../../bdh_cl/scripts/cnn_branch_comparison.py)']
    tmp=report.with_suffix('.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(report)
    save(root/'aggregate.json',{'completed':rows,'failures':failed,'in_progress':progress,'complete':complete})


def worker(args):
    root=Path(args.out).resolve();folder=root/f'{args.size}_{args.arm}_seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and not args.smoke:raise RuntimeError('GPU unavailable: refusing silent overnight CPU fallback')
    baseline_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    train,_,test,order=load(ROOT/'data_cache/cifar100');x=torch.from_numpy(train[0].copy()).float()/255;y=torch.from_numpy(train[1].copy()).long();tx=torch.from_numpy(test[0].copy()).float()/255;ty=torch.from_numpy(test[1].copy()).long()
    BaseModel=load_official();Model=model_class(BaseModel)
    from common.common import parse_args
    from common.sgd_hat import SGD_hat
    from common.lars_optimizer import LARC
    from training.scheduler import GradualWarmupScheduler
    from training.train_IOE_DAC import train as train_features
    from training.train_OOD_classifier import train as train_ood
    from models.classifier import get_simclr_augmentation
    from utils.util_loss import SupConLoss,CompLoss
    channels=SIZES[args.size]
    if args.smoke:save(folder/'verification.json',verify(Model,parse_args,channels,args.seed,device))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    P=parse_args(default=True);P.n_tasks=10;P.n_cls_per_task=10;P.total_cls=100;P.n_gpus=1;P.image_size=(32,32,3);P.logger=Logger();P.smax=700;P.lamb0=1.5;P.lamb1=1;P.batch_size=args.batch_size;P.amp=False
    model=Model(P,channels,args.arm,args.seed).to(device);initial_shared=shared_signature(model)
    augmentation=get_simclr_augmentation(P,(32,32,3)).to(device);criterion=nn.CrossEntropyLoss().to(device);supcon=SupConLoss(temperature=.1).to(device)
    # Model-specific initialization must not change paired augmentation streams.
    random.seed(args.seed+30000);np.random.seed(args.seed+30000);torch.manual_seed(args.seed+30000)
    pmask=None;backmask=None;history=[];aggregation=[];start=time.perf_counter();first_task=0;elapsed_before=0.;phase_seconds={'representation':0.,'classifier':0.,'evaluation':0.}
    cnn_updates=[]
    latest=folder/'latest.pt'
    if latest.exists() and not args.profile and not args.smoke:
        state=torch.load(latest,map_location='cpu',weights_only=True);model.load_state_dict(state['model']);pmask={k:v.to(device) for k,v in state['p_mask'].items()};backmask={k:v.to(device) for k,v in state['mask_back'].items()};history=state['history'];aggregation=state['aggregation'];first_task=state['stages'];elapsed_before=state['elapsed'];phase_seconds=state['phase_seconds'];cnn_updates=state.get('cnn_updates',[]);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);n=state['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        if device.type=='mps':torch.mps.set_rng_state(state['mps_rng'])
    save(folder/'run.json',{'size':args.size,'channels':channels,'arm':args.arm,'seed':args.seed,'initial_shared_sha256':initial_shared,'device':str(device),'parameters':sum(p.numel() for p in model.parameters()),'CNN_convolution_parameters':sum(p.numel() for p in model.convs.parameters()),'baseline_RSS_bytes':baseline_rss,'resumed_after_tasks':first_task,'training_stream_seed':args.seed+30000})
    for task in range(first_task,2 if args.smoke else 10):
        P.t=task;P.mode='train_IOE_DAC';P.epochs=2 if args.smoke else args.feature_epochs;P.DAC_epoch=1 if args.smoke else max(1,round(P.epochs*400/700));P.temp=.1;P.temp_sim=.2
        P.logout=str(folder/f'task{task}');Path(P.logout).mkdir(parents=True,exist_ok=True)
        if task:
            P.mean_all=float(np.mean(aggregation));fraction=sum(v.sum() for v in pmask.values())/sum(v.numel() for v in pmask.values());P.temp=max(.1*(1-float(fraction)),.05)
        comp=CompLoss(P,temperature=P.temp).to(device)
        for name,param in model.named_parameters():param.grad=None;param.hat=None if backmask is None else backmask.get(name)
        optimizer=LARC(SGD_hat(model.parameters(),lr=.1,momentum=.9,weight_decay=0),eps=1e-8,trust_coefficient=.001)
        cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,P.epochs);scheduler=GradualWarmupScheduler(optimizer,multiplier=10.,total_epoch=1 if args.smoke else min(3,max(1,P.epochs//5)),after_scheduler=cosine)
        linear_opt=torch.optim.Adam(model.linear.parameters(),lr=.001)
        ids=torch.nonzero(y//10==task).flatten();cx=x[ids];cy=y[ids]
        if args.smoke:
            take=torch.cat([torch.nonzero(cy==c).flatten()[:2] for c in range(10*task,10*(task+1))]);cx=cx[take];cy=cy[take]
        elif args.profile:cx=cx[:args.batch_size*12];cy=cy[:args.batch_size*12]
        model.allocate(task,cx[:512].to(device))
        cnn_before={name:p.detach().clone() for name,p in model.named_parameters() if name.startswith('convs.')}
        if args.arm=='branch_config':
            used_before=pmask['branch'].flatten().bool() if pmask is not None else torch.zeros(1280,dtype=torch.bool,device=device)
            old_branch=model.branch.weight[used_before].detach().clone();old_bias=model.branch.bias[used_before].detach().clone()
        loader=DataLoader(TensorDataset(cx,cy),batch_size=10 if args.smoke else args.batch_size,shuffle=True,num_workers=0)
        for epoch in range(1,P.epochs+1):
            begin=time.perf_counter();train_features(P,epoch,model,criterion,supcon,comp,optimizer,scheduler,loader,pmask,backmask,simclr_aug=augmentation,linear=model.linear,linear_optim=linear_opt)
            if device.type=='mps':torch.mps.synchronize()
            seconds=time.perf_counter()-begin;phase_seconds['representation']+=seconds
            save(folder/'progress.json',{'task':task+1,'phase':'representation','epoch':epoch,'seconds':seconds})
            if not args.smoke and not args.profile:summary_report(root)
            if args.profile:
                save(folder/'profile.json',{'device':str(device),'seconds':seconds,'batches':len(loader),'raw_images':len(loader.dataset),'parameters':sum(p.numel() for p in model.parameters()),'GPU_current_bytes':torch.mps.current_allocated_memory() if device.type=='mps' else None,'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});return
        if args.arm=='branch_config':
            assert torch.equal(old_branch,model.branch.weight[used_before]);assert torch.equal(old_bias,model.branch.bias[used_before])
        cnn_delta=sum(float((p.detach()-cnn_before[name]).square().sum()) for name,p in model.named_parameters() if name.startswith('convs.'))**.5
        if backmask is not None:
            for name,param in model.named_parameters():
                if name in cnn_before and name in backmask:
                    protected=(backmask[name]==0).expand_as(param)
                    assert torch.equal(cnn_before[name][protected],param.detach()[protected]), {'name':name,'max_protected_change':float((cnn_before[name][protected]-param.detach()[protected]).abs().max()),'protected_count':int(protected.sum()),'attached_mask_matches':torch.equal(param.hat,backmask[name])}
        aggregation.append(float(np.load(Path(P.logout)/'md'/f'aggregation_mean_{task}.npy')))
        pmask,backmask=masks_after_task(P,model,pmask)
        cnn_updates.append({'task':task+1,'CNN_weight_change_L2':cnn_delta,'cumulative_channel_gate_means':[float(pmask[f'encoder{i}'].mean()) for i in range(3)],'exactly_protected_CNN_weights_unchanged':True})
        P.mode='train_OOD_classifier';P.epochs=1 if args.smoke else args.ood_epochs
        for epoch in range(1,P.epochs+1):
            begin=time.perf_counter();train_ood(P,epoch,model,criterion,supcon,comp,optimizer,scheduler,loader,pmask,backmask,simclr_aug=augmentation,linear=model.linear,linear_optim=linear_opt)
            if device.type=='mps':torch.mps.synchronize()
            phase_seconds['classifier']+=time.perf_counter()-begin
            save(folder/'progress.json',{'task':task+1,'phase':'classifier','epoch':epoch})
        seen=ty//10<=task;ex=tx[seen];ey=ty[seen]
        if args.smoke:
            take=torch.cat([torch.nonzero(ey==c).flatten()[:2] for c in range(10*(task+1))]);ex=ex[take];ey=ey[take]
        begin=time.perf_counter();result=evaluate(model,ex,ey,task+1,P,device,64);phase_seconds['evaluation']+=time.perf_counter()-begin;history.append(result)
        assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point())
        state={'model':{k:v.cpu() for k,v in model.state_dict().items()},'p_mask':{k:v.cpu() for k,v in pmask.items()},'mask_back':{k:v.cpu() for k,v in backmask.items()},'stages':task+1,'history':history,'aggregation':aggregation,'cnn_updates':cnn_updates,'phase_seconds':phase_seconds,'elapsed':elapsed_before+time.perf_counter()-start,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:])}
        if device.type=='mps':state['mps_rng']=torch.mps.get_rng_state().cpu()
        torch.save(state,folder/'latest.tmp');(folder/'latest.tmp').replace(latest);save(folder/'history.json',history);save(folder/'cnn_updates.json',cnn_updates)
        print('STAGE',args.size,args.arm,args.seed,task+1,json.dumps({k:v['accuracy'] for k,v in result.items()}),flush=True)
        if args.smoke and task==1:
            from evals.evals import cil
            class Tracker:
                def update(self,*values):pass
            P.mode='cil';P.cil_tracker=Tracker();before=state_digest(model)
            err=cil(P,model,{0:DataLoader(TensorDataset(ex,ey),batch_size=10)})
            assert abs((100-err)/100-result['joint']['accuracy'])<1e-6 and state_digest(model)==before
            save(folder/'smoke_complete.json',{'passed':True,'two_tasks_both_phases':True,'official_joint_evaluator_matches':True,'finite_state':True,'read_only_inference':True,'old_config_branch_rows_unchanged':args.arm=='branch_config'});return
        if not args.profile and not args.smoke:summary_report(root)
    # Count a unique model prototype copy plus the criterion's retained copy.
    continuation=tensor_bytes(model.state_dict())+tensor_bytes(pmask)+tensor_bytes(backmask)+2*tensor_bytes(model.prototypes)+8*len(aggregation)
    # Primary joint inference can discard training-only reference/projection modules.
    inference=sum(tensor_bytes(v) for k,v in model.state_dict().items() if not k.startswith(('linear.','simclr_layer.')))
    record={'size':args.size,'arm':args.arm,'seed':args.seed,'history':history,'average_incremental':{name:st.mean(h[name]['accuracy'] for h in history) for name in ['joint','prototype_cosine']},'continuation_bytes':continuation,'model_bytes':tensor_bytes(model.state_dict()),'primary_inference_tensor_bytes':inference,'seconds':elapsed_before+time.perf_counter()-start,'phase_seconds':phase_seconds,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'parameters':sum(p.numel() for p in model.parameters()),'shared_init_sha256':initial_shared,'cnn_updates':cnn_updates,'mean_acquisition':{name:st.mean(h[name]['task_accuracy'][-1] for h in history) for name in ['joint','prototype_cosine']},'forgetting':{name:st.mean(max(h[name]['task_accuracy'][j] for h in history[j:-1])-history[-1][name]['task_accuracy'][j] for j in range(9)) for name in ['joint','prototype_cosine']}}
    save(folder/'result.json',record);summary_report(root)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(DEFAULT_OUT));parser.add_argument('--size',choices=list(SIZES),default='cnn16');parser.add_argument('--arm',choices=['branch_config','dcnet_hat'],default='branch_config');parser.add_argument('--seed',type=int,default=701);parser.add_argument('--feature-epochs',type=int,default=20);parser.add_argument('--ood-epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--smoke',action='store_true');parser.add_argument('--profile',action='store_true')
    args=parser.parse_args();args.out=str((ROOT/args.out).resolve())
    try:worker(args)
    except BaseException as error:
        folder=Path(args.out).resolve()/f'{args.size}_{args.arm}_seed{args.seed}';save(folder/'failed.json',{'type':type(error).__name__,'error':str(error)})
        raise

if __name__=='__main__':main()
