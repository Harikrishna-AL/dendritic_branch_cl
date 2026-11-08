"""Run official DCNet modules on the branch pilot's exact CIFAR split.

Portability patches in external/DCNet only select MPS/CPU and omit unused
optional imports. All official model, feature/OOD training, losses and HAT
functions are called directly. Short-budget settings are explicit.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from bdh_cl.data.cifar100 import load
ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'external/DCNet'

class Logger:
    def print(self,*args):
        print(*args,flush=True)

def count_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(count_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(count_bytes(v) for v in value)
    return 0

def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def digest(model):
    h=hashlib.sha256()
    for v in model.state_dict().values():h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

@torch.no_grad()
def predict(model,images,stages,smax):
    # Official evals.evals.cil scoring: sum matching rotation logits / 4,
    # concatenate all task heads, argmax. No task IDs or labels are inputs.
    outputs=[]
    for task in range(stages):
        score=0
        for rotation in range(4):
            _,aux,_=model(task,torch.rot90(images,rotation,(2,3)),s=smax,joint=True,penultimate=True)
            score=score+aux['joint'][:,10*rotation:10*(rotation+1)]/4
        outputs.append(score)
    return torch.cat(outputs,1)

@torch.no_grad()
def evaluate(model,x,y,stages,smax,device,batch_size):
    before=digest(model); model.eval();correct=0;route=0;oracle=0;perclass=torch.zeros(100);counts=torch.zeros(100)
    for a in range(0,len(y),batch_size):
        by=y[a:a+batch_size];score=predict(model,x[a:a+batch_size].to(device),stages,smax).cpu()
        pred=score.argmax(-1);correct+=int((pred==by).sum());route+=int((pred//10==by//10).sum())
        own=score.reshape(-1,stages,10)[torch.arange(len(by)),by//10].argmax(-1)+by//10*10
        oracle+=int((own==by).sum())
        for c in by.unique():perclass[c]+=(pred[by==c]==c).sum();counts[c]+=(by==c).sum()
    assert digest(model)==before
    return {'accuracy':correct/len(y),'routing_accuracy':route/len(y),'oracle_task_diagnostic':oracle/len(y),'per_class_accuracy':(perclass[:10*stages]/counts[:10*stages]).tolist(),'test_count':len(y)}


def write_report(out):
    protocol=json.loads((out/'protocol.json').read_text())
    raw=protocol.get('architecture')=='raw_branches'
    destination=ROOT/('reports/dcnet_raw_baseline_20261027' if raw else 'reports/dcnet_official_baseline_20261026')
    destination.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((out/'protocol.json').read_text())
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
    complete=(out/'completion.json').exists()
    lines=[('# Raw-pixel DCNet variant versus the branch pilot' if raw else '# Official-module DCNet baseline versus the branch pilot'),'',
        'Status: '+('complete' if complete else 'running; no complete baseline score yet')+'.','',
        'Official source: https://github.com/Tianqi-Wang1/DCNet at commit '+protocol['commit']+'.', '',
        f"DCNet budget: {protocol['feature_epochs']} representation epochs + {protocol['ood_epochs']} OOD-classifier epochs per task, batch {protocol['batch_size']}, device {protocol['device']}. This is not the published 700+100-epoch recipe. Training uses the official ResNet, losses, augmentation, HAT and OOD classifier functions. DAC begins at epoch {protocol['DAC_epoch']}; warmup is shortened to one epoch.", '',
        'Both experiments use the same 45,000 fit images, 10,000 official test images, ten equal class blocks and class permutation. The 5,000 validation images are unused. Task boundaries are known during training; no true task ID is supplied at inference and no old observations are rehearsed. Backbones, batch sizes, augmentation and compute differ, so this is a same-data comparison, not a resource-matched experiment.', '',
        '| Method | Incremental average (%) | Final accuracy (%) | Persistent tensor state |',
        '|---|---:|---:|---:|',
        '| Branch pilot, head probabilities | 20.39 | 12.64 ± 0.27 | 15.81 MB |',
        '| Branch pilot, cosine | 21.59 | 13.91 ± 0.20 | 15.81 MB |']
    if rows:
        aa=[100*r['average_incremental'] for r in rows];ff=[100*r['final_accuracy'] for r in rows]
        sd=f" ± {st.stdev(ff):.2f}" if len(ff)>1 else ' (one seed)'
        lines.append(f"| DCNet, {len(rows)} completed seeds | {st.mean(aa):.2f} | {st.mean(ff):.2f}{sd} | {rows[0]['continuation_bytes']/1e6:.2f} MB |")
    else:lines.append('| DCNet | Pending | Pending | Pending |')
    lines += ['', 'Progress: '+json.dumps(progress)+'.','',
        '## Stage progress (not final benchmark scores)','', '| Seed | Completed tasks | Latest seen-class accuracy (%) |','|---|---:|---:|']
    for f in sorted(out.glob('seed*_history.json')):
        history=json.loads(f.read_text());lines.append(f"| {f.name.split('_')[0]} | {len(history)} | {100*history[-1]['accuracy']:.2f} |")
    lines += ['', '## Verification and portability','',
        'Two-task smoke training passed both official phases and HAT updates; independent official cil evaluation matched the input-only prediction wrapper. Inference state remained unchanged and model values were finite. The optimizer synchronization patch was checked against original LARC for multiple steps, clipping on/off, zero norms, weight decay and momentum.', '',
        'Patches select MPS instead of CUDA, remap CUDA-saved tensor prototypes onto the selected device, use FP32 crop parameters, omit unused faiss/diffdist imports, and replace GPU-scalar Python branches in LARC with equivalent tensor operations. The exact patch is saved with the run. MPS numerical reproducibility is not the original CUDA environment.', '',
        'DCNet inference combines the matching logit slices from four rotated inputs for every learned task, concatenates all task outputs and takes argmax. No per-head softmax is added. The previous branch pilot uses different scoring rules, as labelled above.', '',
        'Persistent memory counts model tensors, cumulative masks, dense update masks and prototype tensors retained by this runner; peak GPU memory and optimizer workspace are separate and not measured. Checkpoints resume at completed task boundaries, preserving random states. Full result means are reported only for completed ten-task seeds.', '',
        '[Runner](../../bdh_cl/scripts/dcnet_official_baseline.py) · [Run directory](../../'+str(out.relative_to(ROOT))+') · [Branch pilot](../dcnet_branch_pilot_20261026/REPORT.md)']
    if raw:
        lines=[line.replace('official ResNet, losses','matched raw-pixel branch layer, official losses').replace('Backbones, batch sizes, augmentation and compute differ','Both feature layers have 3072 inputs and 128×10 tanh branches, with paired first-layer initialization; gates, projections, augmentation and compute differ') for line in lines]
        lines.extend(['','This replaces the ResNet with the same first-layer dimensions as the branch pilot, not with a pretrained or CNN encoder. DCNet HAT sigmoid gates can activate multiple branches per group, whereas the pilot uses one disjoint branch per group. Its official 128→128→256 projection and learned task heads add parameters. First-layer dimensions are matched; the complete models and active compute are not identical. Raw HAT connection masks use the native min(input_gate, output_gate) rule with group input protection defined by the maximum branch gate.',''])
    target=destination/'REPORT.md';temporary=target.with_suffix('.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(target)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--architecture',choices=['resnet18','raw_branches'],default='resnet18');parser.add_argument('--seeds',default='701,702,703');parser.add_argument('--feature-epochs',type=int,default=5);parser.add_argument('--ood-epochs',type=int,default=5);parser.add_argument('--profile',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args();
    if args.smoke: args.seeds='701';args.feature_epochs=2;args.ood_epochs=1;args.batch_size=10
    out=(ROOT/args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    train,valid,test,order=load(ROOT/'data_cache/cifar100')
    x=torch.from_numpy(train[0].copy()).float()/255;y=torch.from_numpy(train[1].copy()).long();tx=torch.from_numpy(test[0].copy()).float()/255;ty=torch.from_numpy(test[1].copy()).long()
    # Official pickle contains tensor storages saved on CUDA; load on CPU first.
    torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
    sys.path.insert(0,str(UPSTREAM));os.chdir(UPSTREAM)
    from common.common import parse_args
    from models import classifier as C
    from common.sgd_hat import SGD_hat
    from common.lars_optimizer import LARC
    from training.scheduler import GradualWarmupScheduler
    from training.train_IOE_DAC import train as train_features
    from training.train_OOD_classifier import train as train_ood
    from utils.util_loss import SupConLoss,CompLoss
    from mask_ops import cum_mask,freeze_mask,cum_mask_simclr,freeze_mask_simclr
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()
    (out/'portability.patch').write_text(subprocess.check_output(['git','diff'],cwd=UPSTREAM,text=True))
    save(out/'protocol.json',{'architecture':args.architecture,'upstream':'https://github.com/Tianqi-Wang1/DCNet','commit':commit,'device':str(device),'torch':torch.__version__,'class_order':order,'seeds':args.seeds,'fit':45000,'unused_validation':5000,'test':10000,'feature_epochs':args.feature_epochs,'ood_epochs':args.ood_epochs,'DAC_epoch':max(1,round(args.feature_epochs*400/700)),'warmup_epochs':1,'batch_size':args.batch_size,'inference':'Official rotation-logit ensemble across all learned heads; no supplied task ID','replay':False,'scope':('Raw-pixel DCNet architectural ablation: same branch-layer dimensions and images/classorder as pilot, with native DCNet projection/loss/augmentation and HAT. Not published DCNet.' if args.architecture=='raw_branches' else 'Short-budget official-module baseline; not published700+100epoch reproduction; same images/classorder as branch pilot, different backbone, augmentations and compute'),'patches':'MPS selection, CUDA pickle CPU remapping, FP32 crop parameters, unused imports omitted, equivalent tensorized LARC scalar decisions. Official model/training/loss/HAT/inference formulas preserved.'})
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    (out/'runner_source.py').write_text(Path(__file__).read_text())
    if not args.profile and not args.smoke:write_report(out)
    for seed in map(int,args.seeds.split(',')):
        if any(r['seed']==seed for r in rows): continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        P=parse_args(default=True);P.seed=seed;P.n_tasks=10;P.n_cls_per_task=10;P.total_cls=100;P.n_gpus=1;P.image_size=(32,32,3);P.logger=Logger();P.epochs=args.feature_epochs;P.DAC_epoch=max(1,round(P.epochs*400/700));P.warmup=1;P.lamb0=1.5;P.lamb1=1;P.batch_size=args.batch_size;P.amp=False
        if args.architecture=='raw_branches':
            from bdh_cl.scripts.dcnet_raw_branches import RawBranchDCNet,update_masks,verify_raw_model
            model=RawBranchDCNet(P,seed).to(device);save(out/f'seed{seed}_raw_verification.json',verify_raw_model(model))
            (out/'raw_architecture_source.py').write_text((ROOT/'bdh_cl/scripts/dcnet_raw_branches.py').read_text())
        else:
            model=C.get_classifier(P,'resnet18',n_classes=10);model=C.get_shift_classifer(10,model,4).to(device)
        augmentation=C.get_simclr_augmentation(P,image_size=(32,32,3)).to(device)
        criterion=nn.CrossEntropyLoss().to(device);supcon=SupConLoss(temperature=P.temp).to(device)
        pmask=None;backmask=None;history=[];start=time.perf_counter();aggregation=[]
        first_task=0;elapsed_before=0.
        latest=out/f'seed{seed}_latest.pt'
        if latest.exists() and not args.smoke and not args.profile:
            state=torch.load(latest,map_location='cpu',weights_only=True);model.load_state_dict(state['model']);first_task=state['stages'];aggregation=state['aggregation']
            pmask={k:v.to(device) for k,v in state['p_mask'].items()};backmask={k:v.to(device) for k,v in state['mask_back'].items()};history=state.get('history') or json.loads((out/f'seed{seed}_history.json').read_text());elapsed_before=state.get('elapsed_seconds',0.)
            if 'torch_rng' in state:
                torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);n=state['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
                if device.type=='mps':torch.mps.set_rng_state(state['mps_rng'])
            print('RESUME',seed,'completed tasks',first_task,flush=True)
        for task in range(first_task,10):
            P.t=task;P.mode='train_IOE_DAC';P.epochs=args.feature_epochs;P.temp=.1;P.temp_sim=.2
            P.logout=str(out/f'seed{seed}'/f'task{task}');Path(P.logout).mkdir(parents=True,exist_ok=True)
            if task:
                P.mean_all=float(np.mean(aggregation))
                reg=sum(v.sum() for v in pmask.values())/sum(v.numel() for v in pmask.values())
                P.temp=max(.1*(1-float(reg)),.05)
            comp=CompLoss(P,temperature=P.temp).to(device)
            for name,param in model.named_parameters():param.grad=None;param.hat=None if backmask is None else backmask.get(name)
            optimizer=LARC(SGD_hat(model.parameters(),lr=.1,momentum=.9,weight_decay=0),eps=1e-8,trust_coefficient=.001)
            cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,P.epochs)
            scheduler=GradualWarmupScheduler(optimizer,multiplier=10.,total_epoch=1,after_scheduler=cosine)
            linear=model.linear;linear_opt=torch.optim.Adam(linear.parameters(),lr=.001,betas=(.9,.999),weight_decay=0)
            ids=torch.nonzero(y//10==task).flatten();cx=x[ids];cy=y[ids]
            if args.smoke:
                take=torch.cat([torch.nonzero(cy==c).flatten()[:2] for c in range(10*task,10*(task+1))]);cx=cx[take];cy=cy[take]
            loader=DataLoader(TensorDataset(cx,cy),batch_size=args.batch_size,shuffle=True,num_workers=0)
            if args.profile:
                loader=DataLoader(TensorDataset(cx[:args.batch_size*2],cy[:args.batch_size*2]),batch_size=args.batch_size,shuffle=False)
                P.DAC_epoch=1
            for epoch in range(1,P.epochs+1):
                epoch_start=time.perf_counter()
                train_features(P,epoch,model,criterion,supcon,comp,optimizer,scheduler,loader,pmask,backmask,simclr_aug=augmentation,linear=linear,linear_optim=linear_opt)
                if device.type=='mps':torch.mps.synchronize()
                elapsed=time.perf_counter()-epoch_start
                save(out/'progress.json',{'seed':seed,'task':task+1,'phase':'representation','epoch':epoch,'seconds':elapsed})
                if not args.profile and not args.smoke:write_report(out)
                print('PROGRESS',seed,task+1,'representation',epoch,round(elapsed,2),flush=True)
                if args.profile:
                    save(out/'profile.json',{'seconds':elapsed,'batches':len(loader),'raw_images':len(loader.dataset),'device':str(device),'parameters':sum(p.numel() for p in model.parameters())});return
            aggregation.append(float(np.load(Path(P.logout)/'md'/f'aggregation_mean_{task}.npy')))
            # Identical boundary mask updates to official train.py.
            if args.architecture=='raw_branches':pmask,backmask=update_masks(P,task,model,pmask)
            else:
                pmask=cum_mask(P,task,model,pmask);backmask=freeze_mask(P,task,model,pmask);pmask=cum_mask_simclr(P,task,model,pmask);backmask=freeze_mask_simclr(P,task,model,pmask,backmask)
            P.epochs=args.ood_epochs;P.mode='train_OOD_classifier'
            for epoch in range(1,P.epochs+1):
                train_ood(P,epoch,model,criterion,supcon,comp,optimizer,scheduler,loader,pmask,backmask,simclr_aug=augmentation,linear=linear,linear_optim=linear_opt)
                save(out/'progress.json',{'seed':seed,'task':task+1,'phase':'OOD','epoch':epoch})
            seen=ty//10<=task;ex=tx[seen];ey=ty[seen]
            if args.smoke:
                take=torch.cat([torch.nonzero(ey==c).flatten()[:2] for c in range(10*(task+1))]);ex=ex[take];ey=ey[take]
            result=evaluate(model,ex,ey,task+1,P.smax,device,32);history.append(result)
            model_bytes=count_bytes(model.state_dict());continuation=model_bytes+count_bytes(pmask)+count_bytes(backmask)+count_bytes(model.prototypes)+count_bytes(comp.prototypes)+8*len(aggregation)
            state={'model':{k:v.cpu() for k,v in model.state_dict().items()},'p_mask':{k:v.cpu() for k,v in pmask.items()},'mask_back':{k:v.cpu() for k,v in backmask.items()},'aggregation':aggregation,'history':history,'stages':task+1,'seed':seed,'elapsed_seconds':elapsed_before+time.perf_counter()-start,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:])}
            if device.type=='mps':state['mps_rng']=torch.mps.get_rng_state().cpu()
            torch.save(state,out/f'seed{seed}_latest.tmp');(out/f'seed{seed}_latest.tmp').replace(out/f'seed{seed}_latest.pt')
            save(out/f'seed{seed}_history.json',history)
            if not args.profile and not args.smoke:write_report(out)
            print('STAGE',seed,task+1,'accuracy',round(result['accuracy']*100,2),'oracle',round(result['oracle_task_diagnostic']*100,2),flush=True)
            if args.smoke and task==1:
                from evals.evals import cil
                class Tracker:
                    def update(self,*args): pass
                P.mode='cil';P.cil_tracker=Tracker();before=digest(model)
                err=cil(P,model,{0:DataLoader(TensorDataset(ex,ey),batch_size=10)})
                assert abs((100-err)/100-result['accuracy'])<1e-6
                assert digest(model)==before
                assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
                save(out/'verification.json',{'two_task_training_complete':True,'both_official_training_phases':True,'HAT_boundary_updates':True,'finite_model_state':True,'prediction_matches_official_cil_evaluator':True,'inference_state_unchanged':True,'note':'Tiny synthetic-size data subset; smoke scores are not benchmark results.'});return
        model_bytes=count_bytes(model.state_dict());continuation=model_bytes+count_bytes(pmask)+count_bytes(backmask)+count_bytes(model.prototypes)+count_bytes(comp.prototypes)+8*len(aggregation) if first_task<10 else model_bytes+count_bytes(pmask)+count_bytes(backmask)+2*count_bytes(model.prototypes)+8*len(aggregation)
        rows.append({'seed':seed,'history':history,'average_incremental':sum(h['accuracy'] for h in history)/10,'final_accuracy':history[-1]['accuracy'],'model_bytes':model_bytes,'continuation_bytes':continuation,'seconds':elapsed_before+time.perf_counter()-start});save(out/'results.json',rows)
    save(out/'completion.json',{'complete':True,'runs':len(rows)});write_report(out)

if __name__=='__main__':main()
