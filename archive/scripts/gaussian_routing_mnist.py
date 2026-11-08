"""Exp15: routing-only Split-MNIST, matched samples and dendritic initialization."""
import argparse, dataclasses, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bdh_cl.config import preset,as_dict
from bdh_cl.data.mnist import build_pools
from bdh_cl.models.gaussian_routed_bdh import RoutedBDH
from bdh_cl.cl.runner import make_optimizer,lr_at,set_lr


def tensors(images,labels,device):
    im=torch.tensor(images.copy(),device=device)
    lookup=torch.tensor(list(b'0123456789abcdef'),device=device)
    pix=lookup[(im.long().flatten(1)>>4)]
    prompt=torch.cat((torch.full((len(im),1),100,device=device),torch.full((len(im),1),124,device=device),pix,torch.full((len(im),1),62,device=device)),1).long()
    return prompt,im,torch.tensor(labels.copy(),device=device).long()


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model,tests,stage,bs):
    model.eval(); acc=[]; routes=[]
    for x,im,y in tests:
        pred=[]; assignments=[]
        for j in range(0,len(y),bs):
            logits=model(x[j:j+bs],im[j:j+bs])[:,-1,48:48+2*(stage+1)]
            pred.append(logits.argmax(-1))
            if model.last_assignments is not None:assignments.append(model.last_assignments.cpu())
        acc.append(float((torch.cat(pred)==y).float().mean()))
        routes.append(torch.cat(assignments).tolist() if assignments else None)
    return acc,routes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--device',default='cuda');p.add_argument('--smoke',action='store_true')
    p.add_argument('--arms',default='baseline,sum,centroid,gaussian')
    p.add_argument('--out',default='bdh_cl/results/gaussian_routing')
    a=p.parse_args();torch.set_num_threads(4)
    device=torch.device(a.device);bs=32;side=14
    steps=3 if a.smoke else a.steps;neval=20 if a.smoke else 400
    pools=build_pools(ROOT/'data_cache/mnist',side=side)
    trains=[tensors(p.train_images,p.train_labels,device) for p in pools]
    tests=[]
    for pool in pools:
        rng=np.random.default_rng(900)
        ix=np.concatenate([rng.choice(np.flatnonzero(pool.test_labels==c),neval//2,False) for c in pool.classes])
        tests.append(tensors(pool.test_images[ix],pool.test_labels[ix],device))
    spec,_,ts=preset('tiny')
    ts=dataclasses.replace(ts,batch_size=bs,steps_per_task=steps,warmup_steps=min(60,steps//5),seed=a.seed)
    report={'seed':a.seed,'steps':steps,'batch_size':bs,'evaluation_per_pair':neval,'arms':{},
            'protocol':'Split-MNIST 01/23/45/67/89; prompt-only; seen-class global head in training/evaluation; no replay; no task IDs to router; optimizer reset per stage; fixed initial thresholds; no recruitment or synaptic protection',
            'router':'per-head 16D fixed pixel projection; 3 specialists + shared branch; 512 initial training images from first phase only; online accepted-input statistics; uniform priors; variance floor 1e-4; 1st-percentile calibration score',
            'train':as_dict(ts)}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for arm in a.arms.split(','):
        torch.manual_seed(a.seed)
        s=spec if arm=='baseline' else dataclasses.replace(spec,mlp_internal_dim_multiplier=11,neuron_unit='dendritic',neuron_unit_kwargs={'n_branches':4,'soma':'sum'})
        model=RoutedBDH(s,arm,side).to(device)
        rng=np.random.default_rng(a.seed)
        calibration=rng.choice(len(trains[0][2]),512,False)
        sync(device);start=time.perf_counter()
        if model.router is not None:model.router.fit(trains[0][1][calibration],arm)
        sync(device)
        rec={'parameters':sum(p.numel() for p in model.parameters()),'router_bytes':sum(b.numel()*b.element_size() for b in model.router.buffers()) if model.router else 0,'calibration_seconds':time.perf_counter()-start,'stages':[]}
        reference_routes={}
        for stage,(x,im,y) in enumerate(trains):
            opt=make_optimizer(model,ts);model.train();losses=[]
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            sync(device);start=time.perf_counter()
            for step in range(steps):
                ix=torch.tensor(rng.integers(len(y),size=bs),device=device)
                set_lr(opt,lr_at(step,ts));opt.zero_grad(set_to_none=True)
                logits=model(x[ix],im[ix])[:,-1,48:48+2*(stage+1)]
                loss=F.cross_entropy(logits,y[ix]);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();model.update_router();losses.append(loss.detach())
            sync(device);elapsed=time.perf_counter()-start
            acc,routes=evaluate(model,tests,stage,bs)
            route_stats=[]
            if model.router is not None:
                for task,assign in enumerate(routes):
                    arr=np.asarray(assign)
                    if task==stage:reference_routes[task]=arr.copy()
                    route_stats.append({'usage_by_head':[np.bincount(arr[:,h],minlength=4).tolist() for h in range(s.n_head)],'rejected_fraction':float((arr==0).mean()),'agreement_with_acquisition':float((arr==reference_routes[task]).mean()) if task in reference_routes else None})
            rec['stages'].append({'accuracy_by_pair':acc,'seen_accuracy':float(np.mean(acc[:stage+1])),'new_accuracy':acc[stage],'old_accuracy':float(np.mean(acc[:stage])) if stage else None,'train_seconds':elapsed,'loss':float(torch.stack(losses).mean()),'routing':route_stats,'assignments':routes,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None})
            report['arms'][arm]=rec
            (out/f'seed{a.seed}.json').write_text(json.dumps(report,indent=2)+'\n')
            print(f'seed={a.seed} arm={arm} stage={stage} accuracy={acc} train_seconds={elapsed:.2f}',flush=True)
        torch.save(model.state_dict(),out/f'{arm}_seed{a.seed}.pt')
        del model,opt
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
