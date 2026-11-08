"""Exp16: CIFAR-100 base-50 pretraining and value-branch adaptation pilot."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bdh_cl.data.cifar100 import load
from bdh_cl.models.vision_bdh import VisionBDH
from bdh_cl.cl import wblog


def dump(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def images(x,augment=False):
    x=x.float()/255.
    if augment:
        B=len(x);padded=F.pad(x,(4,4,4,4),mode='reflect')
        offsets=torch.randint(0,9,(B,2),device=x.device)
        yy=offsets[:,0,None,None]+torch.arange(32,device=x.device)[None,:,None]
        xx=offsets[:,1,None,None]+torch.arange(32,device=x.device)[None,None,:]
        x=padded.permute(0,2,3,1)[torch.arange(B,device=x.device)[:,None,None],yy,xx].permute(0,3,1,2)
        flip=torch.rand(B,1,1,1,device=x.device)<.5
        x=torch.where(flip,x.flip(-1),x)
    return (x-.5)/.5


def select(data,low,high):
    x,y=data;mask=(y>=low)&(y<high);return x[mask],y[mask]


@torch.no_grad()
def evaluate(model,data,seen,batch=128):
    model.eval();x,y=select(data,0,seen);correct=[];loss=[];usage_sum=None
    for start in range(0,len(y),batch):
        target=y[start:start+batch]
        out=model(images(x[start:start+batch]))[:,:seen]
        if model.router is not None and model.mode=='gaussian':
            usage=torch.stack(model.router.observations).mean(0)*len(target)
            usage_sum=usage if usage_sum is None else usage_sum+usage
        correct.append((out.argmax(-1)==target).float())
        loss.append(F.cross_entropy(out,target,reduction='sum'))
    correct=torch.cat(correct)
    classes=[float(correct[y==c].mean()) for c in range(seen)]
    result={'accuracy':float(correct.mean()),'loss':float(torch.stack(loss).sum()/len(y)),'class_accuracy':classes}
    if usage_sum is not None:
        usage=usage_sum/len(y)
        result['gate']={'rejection':float(1-usage.sum(-1).mean()),'usage':[float(usage[...,k].mean()) for k in range(model.branches)]}
    return result


def optimizer(model,lr):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.01,betas=(.9,.95))


def train_epoch(model,data,seen,opt,epoch,total,lr,device,batch=128):
    model.train();x,y=data;order=torch.randperm(len(y),device=device)
    # Epoch-level cosine schedule, 5 epoch warmup for base training / 1 for short pilots.
    warm=min(5,max(1,total//10))
    rate=lr*min(1.,(epoch+1)/warm)*(.1+.9*.5*(1+math.cos(math.pi*epoch/total)))
    for group in opt.param_groups:group['lr']=rate
    losses=[];usage_sum=None;sync(device);start=time.perf_counter()
    for start_ix in range(0,len(y),batch):
        ix=order[start_ix:start_ix+batch]
        opt.zero_grad(set_to_none=True)
        logits=model(images(x[ix],True))[:,:seen]
        loss=F.cross_entropy(logits,y[ix]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step();model.commit();losses.append(loss.detach())
        if model.router is not None and model.mode=='gaussian':
            usage=torch.stack(model.router.observations).mean(0)*len(ix)
            usage_sum=usage if usage_sum is None else usage_sum+usage
    sync(device)
    result={'train/loss':float(torch.stack(losses).mean()),'train/seconds':time.perf_counter()-start,'train/lr':rate}
    if usage_sum is not None:
        usage=usage_sum/len(y)
        result.update(model.router.metrics())
        result['gate/rejection']=float(1-usage.sum(-1).mean())
        for k in range(model.branches):result[f'gate/branch_{k}_usage']=float(usage[...,k].mean())
    return result


def start_log(a,name,config):
    wblog.init({'enabled':not a.no_wandb,'project':'bdh-cl','group':'exp16_cifar100_value_branches','job_type':a.phase},name=name,config=config)
    if wblog.active():
        try:
            wblog._RUN.define_metric('epoch')
            wblog._RUN.define_metric('train/*',step_metric='epoch')
            wblog._RUN.define_metric('validation/*',step_metric='epoch')
            wblog._RUN.define_metric('cl/*',step_metric='stage')
            return wblog._RUN.url
        except Exception:pass
    return None


def pretrain(a,train,valid,test,order,device,out):
    torch.manual_seed(a.seed)
    model=VisionBDH().to(device);opt=optimizer(model,a.lr)
    config=vars(a)|{'class_order':order,'architecture':'BDH D128 H4 N512 depth4, 64 4x4 RGB patches + final cls, learned positions; standard query/value projections'}
    report={'config':config,'epochs':[],'wandb_url':start_log(a,f'base50-seed{a.seed}',config)}
    train=select(train,0,50);best=-1;best_epoch=0
    epochs=2 if a.smoke else a.epochs
    if a.smoke:train=tuple(t[:128] for t in train)
    for epoch in range(epochs):
        metrics=train_epoch(model,train,50,opt,epoch,epochs,a.lr,device)
        ev=evaluate(model,valid,50)
        metrics.update({'epoch':epoch+1,'validation/accuracy':ev['accuracy'],'validation/loss':ev['loss']})
        report['epochs'].append(metrics);wblog.log(metrics)
        if ev['accuracy']>best:
            best=ev['accuracy'];best_epoch=epoch+1
            torch.save({'model':model.state_dict(),'epoch':best_epoch,'validation_accuracy':best,'class_order':order},out/'base_best.pt')
        report['best_validation_accuracy']=best
        dump(out/'pretrain.json',report)
        print(json.dumps(metrics),flush=True)
        if not a.smoke and epoch+1-best_epoch>=20 and epoch>=29:break
    ckpt=torch.load(out/'base_best.pt',map_location=device,weights_only=False);model.load_state_dict(ckpt['model'])
    report['base_test']=evaluate(model,test,50)
    # Quality gate prevents meaningless adaptation of an essentially untrained model.
    report['ready_for_adaptation']=best>=.20
    report['quality_threshold']=.20
    dump(out/'pretrain.json',report)
    wblog.summary({'best_validation_accuracy':best,'base_test_accuracy':report['base_test']['accuracy'],'ready_for_adaptation':report['ready_for_adaptation']})
    wblog.finish()
    print('PRETRAIN COMPLETE',report['ready_for_adaptation'],flush=True)


def adapt(a,train,valid,test,order,device,out):
    source=Path(a.pretrained)
    base_report=json.loads((source/'pretrain.json').read_text())
    if not a.smoke and not base_report['ready_for_adaptation']:
        print('ADAPTATION HELD: base validation accuracy below 20%; inspect pretraining before proceeding.',flush=True)
        dump(out/'adaptation_held.json',{'reason':'pretraining quality gate','base_validation':base_report['best_validation_accuracy']})
        return
    ckpt=torch.load(source/'base_best.pt',map_location=device,weights_only=False)
    if ckpt['class_order']!=order:raise ValueError('class order mismatch')
    epochs=1 if a.smoke else a.adapt_epochs
    arms=a.arms.split(',')
    all_results={}
    for arm in arms:
        torch.manual_seed(a.seed+100)
        baseline=arm=='finetune';gaussian='gaussian' in arm
        model=VisionBDH(branches=0 if baseline else 3,mode='gaussian' if gaussian else 'open').to(device)
        missing,unexpected=model.load_state_dict(ckpt['model'],strict=False)
        assert not unexpected
        assert all(k.startswith(('branch_','router.')) for k in missing)
        if not baseline:
            # Base-training inputs only, no labels to the router. Offline calibration is explicit.
            base_x,_=select(train,0,50)
            ix=torch.randperm(len(base_x),device=device)[:64]
            model.eval()
            with torch.no_grad():model(images(base_x[ix]),calibrate=True)
        model.set_backbone_trainable(not arm.startswith('frozen'))
        config=vars(a)|{'arm':arm,'class_order':order,'pretraining_epoch':ckpt['epoch'],'branch_count':0 if baseline else 3,'classifier_policy':'all seen rows trainable in every arm','backbone_frozen':arm.startswith('frozen')}
        report={'config':config,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'stages':[],'wandb_url':start_log(a,f'{arm}-seed{a.seed}',config)}
        report['initial_base_test']=evaluate(model,test,50)
        maxima=np.array(report['initial_base_test']['class_accuracy']+[0.]*50)
        for stage in range(1,6):
            seen=50+10*stage;new=select(train,seen-10,seen)
            if a.smoke:new=tuple(t[:128] for t in new)
            opt=optimizer(model,a.adapt_lr);history=[]
            # Identical minibatch/augmentation RNG sequence in every arm.
            torch.manual_seed(a.seed+1000+stage)
            for epoch in range(epochs):
                metrics=train_epoch(model,new,seen,opt,epoch,epochs,a.adapt_lr,device)
                route_metrics={}
                ev=evaluate(model,valid,seen)
                row=metrics|route_metrics|{'epoch':(stage-1)*epochs+epoch+1,'stage':stage,'validation/accuracy':ev['accuracy'],'validation/base_accuracy':float(np.mean(ev['class_accuracy'][:50])),'validation/new_accuracy':float(np.mean(ev['class_accuracy'][-10:]))}
                history.append(row);wblog.log(row)
                print(arm,json.dumps(row),flush=True)
            ev=evaluate(model,test,seen)
            old=seen-10;accuracy=np.array(ev['class_accuracy'])
            forgetting=float(np.mean(maxima[:old]-accuracy[:old]))
            maxima[:seen]=np.maximum(maxima[:seen],accuracy)
            record={'stage':stage,'seen_classes':seen,'test':ev,'history':history,'base_accuracy':float(accuracy[:50].mean()),'old_accuracy':float(accuracy[:old].mean()),'new_accuracy':float(accuracy[-10:].mean()),'forgetting':forgetting}
            if gaussian:record['gate']=ev['gate']
            report['stages'].append(record)
            wblog.log({'stage':stage,'cl/accuracy':ev['accuracy'],'cl/base_accuracy':record['base_accuracy'],'cl/old_accuracy':record['old_accuracy'],'cl/new_accuracy':record['new_accuracy'],'cl/forgetting':forgetting})
            dump(out/f'{arm}.json',report)
            torch.save(model.state_dict(),out/f'{arm}_latest.pt')
        if wblog.active():
            try:
                import wandb
                series=[[report['initial_base_test']['accuracy']]+[r['base_accuracy'] for r in report['stages']], [report['initial_base_test']['accuracy']]+[r['new_accuracy'] for r in report['stages']], [report['initial_base_test']['accuracy']]+[r['test']['accuracy'] for r in report['stages']]]
                wblog.log({'plots/retention_and_acquisition':wandb.plot.line_series(xs=list(range(6)),ys=series,keys=['base 50','new classes','all seen'],title='CIFAR-100 retention and acquisition',xname='stage')})
            except Exception as exc:print('Plot logging failed:',type(exc).__name__,flush=True)
        wblog.summary({'final_accuracy':report['stages'][-1]['test']['accuracy'],'final_base_accuracy':report['stages'][-1]['base_accuracy']})
        wblog.finish();all_results[arm]=report
        del model,opt
        if device.type=='cuda':torch.cuda.empty_cache()
    dump(out/'all_results.json',all_results)
    print('ADAPTATION COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pretrain','adapt'],required=True)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--pretrained')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--adapt-epochs',type=int,default=10);p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--adapt-lr',type=float,default=3e-4);p.add_argument('--device',default='cuda')
    p.add_argument('--arms',default='finetune,frozen_open,frozen_gaussian,trainable_open,trainable_gaussian')
    p.add_argument('--smoke',action='store_true');p.add_argument('--no-wandb',action='store_true')
    a=p.parse_args();torch.set_num_threads(4);device=torch.device(a.device)
    train,valid,test,order=load(a.data)
    if a.smoke:
        def small(data):
            ix=np.concatenate([np.flatnonzero(data[1]==c)[:2] for c in range(100)])
            return data[0][ix],data[1][ix]
        valid=small(valid);test=small(test)
    datasets=[tuple(torch.tensor(t.copy(),device=device) for t in data) for data in (train,valid,test)]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.phase=='pretrain':pretrain(a,*datasets,order,device,out)
    else:adapt(a,*datasets,order,device,out)

if __name__=='__main__':main()
