"""Hybrid reward/eligibility-trained routing; no counterfactuals or prototypes.

Routing synapses: analytic score-function update from actual sampled choices
and scalar correctness feedback. Branch computations/readout: ordinary Adam CE.
Independent images are independent trials; traces do not leak between trials.
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw,SPLITS


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(784,96);self.head=nn.Linear(24,10)
        # Updated explicitly with the local eligibility rule, never backprop.
        self.router=nn.Linear(784,96)
        self.router.requires_grad_(False)

    def forward(self,x,generator=None):
        with torch.no_grad():
            p=self.router(x).reshape(-1,24,4).softmax(-1)
            j=p.argmax(-1) if generator is None else torch.multinomial(p.reshape(-1,4),1,generator=generator).reshape(-1,24)
            tag=F.one_hot(j,4).float()-p
        b=torch.tanh(self.first(x).reshape(-1,24,4))
        h=b.gather(2,j[:,:,None]).squeeze(-1)
        return self.head(h),tag,p,j

    @torch.no_grad()
    def reinforce(self,x,tag,advantage,lr):
        # e_ijk = x_k*(chosen_ij-p_ij), retained in factored form until feedback.
        # Each example's scalar reward multiplies ITS OWN eligibility trace.
        credit=(tag*advantage[:,None,None]).reshape(-1,96)
        self.router.weight.add_(credit.T@x,alpha=lr/len(x))
        self.router.bias.add_(credit.mean(0),alpha=lr)


def build(arm,seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784,190),nn.Tanh(),nn.Linear(190,10)) if arm=='mlp' else Branches()


@torch.no_grad()
def evaluate(model,x,y):
    before={k:v.clone() for k,v in model.state_dict().items()}
    preds=[];entropies=[];choices=[]
    for bx in x.split(512):
        if isinstance(model,Branches):
            z,tag,p,j=model(bx);entropies.append(-(p*p.clamp_min(1e-9).log()).sum(-1).flatten());choices.append(j)
        else:z=model(bx)
        preds.append(z.argmax(-1))
    pred=torch.cat(preds)
    result={'accuracy':float((pred==y).float().mean()),'pairs':[
        float((pred[(y==a)|(y==b)]==y[(y==a)|(y==b)]).float().mean()) for a,b in SPLITS]}
    if entropies:
        result['routing_entropy']=float(torch.cat(entropies).mean())
        hist=F.one_hot(torch.cat(choices),4).float().mean(0)
        result['mean_dominant_branch_fraction']=float(hist.max(-1).values.mean())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    return result


def fit(arm,seed,lr,x,y,groups,epochs):
    model=build(arm,seed)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    initial=model.router.weight.detach().clone() if isinstance(model,Branches) else None
    baseline=0.;history=[];elapsed=0.;reward_sum=0.;samples=0
    for stage,indices in enumerate(groups):
        start=time.perf_counter()
        for epoch in range(epochs):
            rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch)
            route_rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch+10000000)
            feedback_rng=torch.Generator().manual_seed(seed*100000+stage*100+epoch+20000000)
            order=indices[torch.randperm(len(indices),generator=rng)]
            for ix in order.split(128):
                bx,by=x[ix],y[ix]
                opt.zero_grad(set_to_none=True)
                if isinstance(model,Branches):z,tag,p,j=model(bx,route_rng)
                else:z=model(bx)
                reward=(z.detach().argmax(-1)==by).float()
                advantage=reward-baseline
                baseline=.95*baseline+.05*float(reward.mean())
                # Label gradients only update branch responses/readout, not routing.
                loss=F.cross_entropy(z,by);loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5)
                opt.step()
                if isinstance(model,Branches):
                    assert model.router.weight.grad is None and model.router.bias.grad is None
                    if arm=='shuffled_reward':advantage=advantage[torch.randperm(len(bx),generator=feedback_rng)]
                    if arm!='frozen_router':model.reinforce(bx,tag,advantage,lr)
                reward_sum+=float(reward.sum());samples+=len(bx)
        elapsed+=time.perf_counter()-start
        # Stage snapshots used only by evaluation driver, never feedback.
        history.append({k:v.clone() for k,v in model.state_dict().items()})
    stats={'train_seconds':elapsed,'presentations':samples,'sampled_training_accuracy':reward_sum/samples,
           'parameters':sum(p.numel() for p in model.parameters()),
           'router_update_norm':float((model.router.weight-initial).norm()) if initial is not None else None}
    if arm=='frozen_router':assert stats['router_update_norm']==0
    return model,history,stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    protocol={'development_seed':60,'final_seeds':[61,62,63], 'arms':['reward_router','frozen_router','shuffled_reward','mlp'],
              'architecture':'24 neurons x4 tanh branches, shared10-class head; separate input-dependent linear routing synapses. Branch total150970 allocated weights; MLP784->190->10 has151060. Frozen router has same allocated weights but fewer updated weights.',
              'rule':'Sample one branch per neuron with softmax probabilities. Eligibility x*(onehot(choice)-probability), factored per example until its actual correctness reward arrives. Update routing weights by minibatch mean advantage*eligibility with explicit SGD. Baseline scalar EMA .95, prior to current batch. No alternative-branch prediction losses, prototypes, exemplars, task IDs or boundary signals.',
              'hybrid':'Branch content and classifier learn factual supervised CE via Adam .001, clip5. Router never receives CE gradients. Environment computes scalar0/1 correctness from training labels. Global scalar advantage credited to all24 sampled choices; not biological proof.',
              'trace_scope':'Immediate-feedback first experiment: eligibility held from decision until outcome update, reset per independent image. No cross-image EMA, no delayed-credit claim. Factored trace uses current input plus choice-minus-probability; no long-term input association table.',
              'selection':'Seed60, fixed stratified50k/10k train/router-validation split. Router rates .01/.1/1.0, five joint epochs; choose validation accuracy only. MLP and frozen controls fixed Adam setting. Test untouched until settings locked.',
              'progression':'Run full-data joint evaluation on fresh seeds after selection. If selected development validation accuracy>=90%, also run two epochs per pair on natural five-pair stream. Report all arms, including null results.',
              'evaluation':'Deterministic argmax branch at inference, no test updates or feedback. Stochastic routing during training. All10 outputs always compete. Model state checked unchanged.',
              'limits':'First routing-credit pilot, no evidence of trace timing benefit, no biologically plausible backbone update, separate routing synapses add capacity, one continual task order. Reused benchmark but fresh seeds; no claim of task separation or retention from reward learning alone.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    rng=torch.Generator().manual_seed(60)
    val=torch.cat([torch.nonzero(y==c,as_tuple=True)[0][torch.randperm(int((y==c).sum()),generator=rng)[:1000]] for c in range(10)])
    mask=torch.ones(len(y),dtype=torch.bool);mask[val]=False;train=torch.nonzero(mask,as_tuple=True)[0]
    dev=[]
    for lr in [.01,.1,1.]:
        model,_,stats=fit('reward_router',60,lr,x,y,[train],5)
        score=evaluate(model,x[val],y[val]);dev.append({'rate':lr,**score,**stats})
        print('development',lr,score['accuracy'],flush=True)
    selected=max(dev,key=lambda r:r['accuracy']);lr=selected['rate']
    (out/'selection.json').write_text(json.dumps({'candidates':dev,'selected':selected,'continual_enabled':selected['accuracy']>=.90},indent=2))
    results=[]
    for seed in protocol['final_seeds']:
        for mode in ['joint']+(['continual'] if selected['accuracy']>=.90 else []):
            groups=[torch.arange(len(y))] if mode=='joint' else [torch.nonzero((y==a)|(y==b),as_tuple=True)[0] for a,b in SPLITS]
            for arm in protocol['arms']:
                model,states,stats=fit(arm,seed,lr,x,y,groups,5 if mode=='joint' else 2)
                scores=[]
                for state in states:
                    model.load_state_dict(state);scores.append(evaluate(model,tx,ty))
                row={'seed':seed,'mode':mode,'arm':arm,**stats,'stages':scores,'final_accuracy':scores[-1]['accuracy']}
                if mode=='continual':
                    row['old_accuracy']=st.mean(scores[-1]['pairs'][:4]);row['acquisition']=st.mean(scores[i]['pairs'][i] for i in range(5))
                results.append(row);torch.save(model.state_dict(),out/f'{seed}_{mode}_{arm}.pt')
                (out/'results.json').write_text(json.dumps(results,indent=2))
                print(seed,mode,arm,round(row['final_accuracy']*100,2),'router change',stats['router_update_norm'],flush=True)
    lines=['# Reward/eligibility routing pilot','',
           'Hybrid learning: branch selection uses actual-outcome scalar reinforcement; branch computations '
           'and readout use ordinary supervised backpropagation. No counterfactual scoring or input-prototype '
           'memory. Input associations live in additional routing synapses, counted in parameter matching.', '',
           f'Selected router SGD rate{lr} using development validation only. Branch weights150,970; matched '
           'MLP151,060. Three fresh evaluation seeds. Frozen-router and shuffled-reward controls distinguish '
           'routing credit from learning in the backbone.', '',
           '| Mode | Arm | Test accuracy mean ± SD | Train seconds |', '|---|---|---:|---:|']
    for mode in ['joint','continual']:
        for arm in protocol['arms']:
            rs=[r for r in results if r['mode']==mode and r['arm']==arm]
            if not rs:continue
            a=[r['final_accuracy']*100 for r in rs]
            lines.append(f'| {mode} | {arm} | {st.mean(a):.2f} ± {st.stdev(a):.2f}% | {st.mean(r["train_seconds"] for r in rs):.2f} |')
    lines+=['','## Scope','',
            'Each image is an independent trial. The eligibility trace is the local presynaptic input times '
            'sampled participation minus expected participation, retained in factored form until immediate '
            'outcome feedback. There is no persistent trace across unrelated images, and no delayed-feedback '
            'benefit is claimed. Stored scalar reward baseline is action-independent before the batch. '
            'Shuffled reward preserves reward distribution while breaking example-specific feedback.', '',
            'The binary reward is computed by an environment using the supervised training label. The router '
            'sees scalar advantage, not labels or class-specific gradients. Branch responses/classifier still '
            'see full labels and use Adam. This is not a completely biologically plausible learning system. '
            'Hard argmax at inference differs from sampling at training; the mismatch is a possible limitation.', '',
            'Continual evaluation is enabled only if the selected development model exceeds90% validation '
            'accuracy on ordinary learning. If all router controls learn similarly, ordinary classification '
            'does not establish a benefit from the reinforcement rule. If continual accuracy collapses, '
            'eligibility-based credit has not solved stability/plasticity.', '',
            'Routing weights never have backpropagated gradients; frozen routing weights are asserted '
            'unchanged; all evaluations assert complete state invariance. Source, protocol, development '
            'selection, model checkpoints and full per-seed results are saved.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:20]),flush=True)


if __name__=='__main__':main()
