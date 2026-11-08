"""Full-trainable dendritic BDH, no replay: A -> B -> A, matched state arms."""
import argparse
import copy
import dataclasses
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bdh_cl.config import preset,as_dict
from bdh_cl.data.mnist import build_pools,render
from bdh_cl.models.compartment_bdh import CompartmentBDH
from bdh_cl.cl.runner import make_optimizer,lr_at,set_lr,git_hash


def inputs(pool,indices,device,test=False):
    images=pool.test_images if test else pool.train_images
    labels=pool.test_labels if test else pool.train_labels
    prompts=[render(images[i],int(labels[i]),pool.tag).prompt for i in indices]
    return torch.tensor([list(p.encode('ascii')) for p in prompts],device=device),torch.tensor(labels[indices].astype(np.int64),device=device)


def balanced(labels,classes,n,seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(labels==c),n//len(classes),replace=False) for c in classes])


@torch.no_grad()
def evaluate(model,pools,sets,device,bs):
    model.eval()
    out=[]
    for pool,ix in zip(pools,sets):
        right,total,restricted_right=0,0,0
        routing=[];ratios=[];head_routes=[];write_choices=[]
        for j in range(0,len(ix),bs):
            x,y=inputs(pool,ix[j:j+bs],device,True)
            logits=model(x)[0][:,-1,48:58]
            right+=int((logits.argmax(-1)==y).sum());total+=len(y)
            classes=torch.tensor(pool.classes,device=device)
            restricted_right+=int((classes[logits[:,classes].argmax(-1)]==y).sum())
            if model.last_routing is not None:
                routing.append(model.last_routing)
                head_routes.append(model.last_head_routing)
                write_choices.append(model.last_write_choices)
            ratios.append(model.last_read_ratio)
        out.append({'accuracy':right/total,'task_restricted_diagnostic':restricted_right/total,'routing':np.mean(routing,axis=0).tolist() if routing else None,
                    'routing_by_head':np.mean(head_routes,axis=0).tolist() if head_routes else None,
                    'hypothetical_write_choices_by_head':np.mean(write_choices,axis=0).tolist() if write_choices else None,
                    'read_fast_norm_ratio':float(np.mean(ratios))})
    return out


def permuted_environment(pool):
    """Same labels and images with one fixed spatial permutation in both splits."""
    order=np.random.default_rng(1234).permutation(pool.side*pool.side)
    def change(images):
        return images.reshape(len(images),-1)[:,order].reshape(images.shape)
    return dataclasses.replace(pool,name=pool.name+'_permuted',
                               train_images=change(pool.train_images),test_images=change(pool.test_images))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--domain-swap',action='store_true')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--steps',type=int,default=800)
    ap.add_argument('--arms',default='none,shared,uniform,selective')
    ap.add_argument('--eta',type=float,default=.01)
    ap.add_argument('--read-scale',type=float,default=.1)
    ap.add_argument('--out',default='bdh_cl/results/compartment_recurrence')
    args=ap.parse_args()
    torch.set_num_threads(4)
    device=torch.device(args.device)
    bs,side,steps,neval=(8,7,20,50) if args.smoke else (32,14,args.steps,200)
    spec,_,tspec=preset('tiny')
    spec=dataclasses.replace(spec,mlp_internal_dim_multiplier=11,neuron_unit='dendritic',
                             neuron_unit_kwargs={'n_branches':4,'soma':'sum'})
    tspec=dataclasses.replace(tspec,batch_size=bs,steps_per_task=steps,warmup_steps=min(80,steps//5),seed=args.seed)
    pools=build_pools(ROOT/'data_cache/mnist',side=side,splits=((0,1,2,3,4),(5,6,7,8,9)))
    if args.domain_swap:
        pools=[pools[0],permuted_environment(pools[0])]
    sets=[balanced(p.test_labels,p.classes,neval,args.seed+i) for i,p in enumerate(pools)]
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    report={'seed':args.seed,'git_hash':git_hash(),'device':str(device),'side':side,'spec':as_dict(spec),
            'train':as_dict(tspec),'domain_swap':args.domain_swap,'environment_definition':'same labels 0-4, fixed pixel permutation in B' if args.domain_swap else 'disjoint labels 0-4 versus 5-9',
            'n_eval_per_environment':neval,'schedule':['A','B','A'],
            'memory':{'decay':.999,'eta':args.eta,'read_scale':args.read_scale,'routing_temperature':5.,'routing':'learned branch-summary read gate; argmax write gate','read_normalization':'per-token/head recalled norm matched to fast attention, then scaled'},
            'protocol':'prompt-only digit classification, all ten class logits, full-trainable backbone, no replay/task IDs',
            'arms':{}}
    t0=time.perf_counter()
    for arm in args.arms.split(','):
        torch.manual_seed(args.seed)
        model=CompartmentBDH(spec,arm,eta=args.eta,read_scale=args.read_scale).to(device)
        rng=np.random.default_rng(args.seed)
        rec={'parameters':sum(p.numel() for p in model.parameters()),'state_bytes':model.memory_bytes(),'stages':[]}
        for stage,pool in enumerate([pools[0],pools[1],pools[0]]):
            before=evaluate(model,pools,sets,device,bs)
            # Memory ablations at return, before any A update.
            if stage==2 and model.slow_state is not None:
                saved=model.slow_state.detach().clone()
                model.slow_state.zero_()
                rec['return_zero_memory']=evaluate(model,pools,sets,device,bs)
                model.slow_state.copy_(saved.roll(1,dims=1))
                rec['return_permuted_memory']=evaluate(model,pools,sets,device,bs)
                model.slow_state.copy_(saved)
            opt=make_optimizer(model,tspec)
            model.train()
            curve=[];losses=[]
            st=time.perf_counter()
            for step in range(steps):
                set_lr(opt,lr_at(step,tspec))
                ix=rng.integers(len(pool.train_labels),size=bs)
                x,y=inputs(pool,ix,device)
                logits=model(x,update_memory=True)[0][:,-1,48:58]
                loss=F.cross_entropy(logits,y)
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();losses.append(float(loss.detach()))
                if (step+1)%100==0 or step==steps-1:
                    scores=evaluate(model,pools,sets,device,bs)
                    curve.append({'updates':step+1,'accuracy':[s['accuracy'] for s in scores]})
                    print(f'seed={args.seed} arm={arm} stage={stage} updates={step+1} loss={np.mean(losses[-100:]):.3f} A/B={curve[-1]["accuracy"]}',flush=True)
                    model.train()
            after=evaluate(model,pools,sets,device,bs)
            rec['stages'].append({'stage':stage,'before':before,'after':after,'curve':curve,'losses':losses,
                                  'elapsed_seconds':time.perf_counter()-st,
                                  'state_compartment_norms':model.slow_state.norm(dim=(-2,-1)).cpu().tolist() if model.slow_state is not None else None,
                                  'state_norm':float(model.slow_state.norm()) if model.slow_state is not None else 0.})
            report['arms'][arm]=rec
            report['elapsed_seconds']=time.perf_counter()-t0
            (out/f'seed{args.seed}.json').write_text(json.dumps(report,indent=2)+'\n')
        rec['retention_before_return']=rec['stages'][2]['before'][0]['accuracy']
        rec['B_after_learning']=rec['stages'][1]['after'][1]['accuracy']
        rec['mixed_final_accuracy']=np.mean([s['accuracy'] for s in rec['stages'][2]['after']])
        torch.save(model.state_dict(),out/f'{arm}_seed{args.seed}.pt')
        del model,opt
        if device.type=='cuda':torch.cuda.empty_cache()
        if device.type=='mps':torch.mps.empty_cache()
    report['elapsed_seconds']=time.perf_counter()-t0
    (out/f'seed{args.seed}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Completed in {report["elapsed_seconds"]:.1f}s: {out}',flush=True)

if __name__=='__main__':main()
