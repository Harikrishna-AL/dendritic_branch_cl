"""Delayed-credit follow-up on frozen branches. Diagnostic, not strict CL."""
import argparse
import json
import statistics as st
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.split_mnist_usefulness import make
from bdh_cl.scripts.frozen_branch_feedback import cache,initialize,evaluate


@torch.no_grad()
def train(q,b,y,model,seed,lr,delay,shuffle):
    u=initialize(seed);baseline=0.;pending=None;peak=0
    def deliver(trace,rng):
        nonlocal baseline
        aug,tag,entropy_credit,reward=trace
        adv=reward-baseline;baseline=.95*baseline+.05*float(reward.mean())
        if shuffle:adv=adv[torch.randperm(len(adv),generator=rng)]
        credit=tag*adv[:,None,None]+entropy_credit
        u.add_((aug.T@credit.reshape(-1,96)).reshape(197,24,4),alpha=lr*(.9**delay)/len(aug))
    for epoch in range(20):
        g=torch.Generator().manual_seed(seed*10000+epoch)
        choices=torch.Generator().manual_seed(seed*10000+epoch+1000000)
        noise=torch.Generator().manual_seed(seed*10000+epoch+2000000)
        order=torch.randperm(len(q),generator=g)
        for ids in order.split(128):
            aug=torch.cat([q[ids],torch.ones(len(ids),1)],1)
            p=(aug@u.reshape(197,96)).reshape(-1,24,4).softmax(-1)
            j=torch.multinomial(p.reshape(-1,4),1,generator=choices).reshape(-1,24)
            z=model.head(b[ids].gather(2,j[:,:,None]).squeeze(-1))
            reward=(z.argmax(1)==y[ids]).float()
            tag=F.one_hot(j,4).float()-p
            logp=p.clamp_min(1e-9).log();entropy=-(p*logp).sum(-1,keepdim=True)
            trace=(aug,tag,.01*(-p*(logp+entropy)),reward)
            peak=max(peak,sum(t.numel()*t.element_size() for t in trace))
            if delay:
                if pending is not None:deliver(pending,noise)
                pending=trace
            else:deliver(trace,noise)
    if pending is not None:deliver(pending,noise)
    return u,peak


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    selection=json.loads(Path('results/frozen_branch_feedback_20261013/selection.json').read_text())
    lr=selection['selected']['binary']['rate']
    protocol={'rate':lr,'epochs':20,'pairs':[[34,75],[35,76],[36,77]],
              'scope':'Frozen-backbone routing diagnostic revisiting historical training data, not strict CL or a fully biological network.',
              'delay':'Either immediate, or reward delivered after one intervening minibatch decision. Trial eligibility retained in a factored tuple and multiplied by .9 for one-step decay. No conversion to biological seconds. At most two decision tuples coexist while delivering old credit.',
              'control':'Identical binary reward signal in all arms; shuffle rewards among trials at delivery to break correct eligibility association. Same initialization, sample order, stochastic-choice generator and selected learning rate; no new tuning.',
              'biology':'Abstract score-function trace, scalar correctness feedback from training environment, local entropy term. Still requires circuit implementation of competition/probabilities; frozen computations originally learned via counterfactual supervised method.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'source.py').write_text(Path(__file__).read_text())
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255;y=torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255;ty=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    prior=json.loads(Path('results/frozen_branch_feedback_20261013/results.json').read_text());rows=[]
    for backbone,seed in protocol['pairs']:
        model=make('local','hard');model.load_state_dict(torch.load(f'results/split_mnist_compute_comparison_20261013/{backbone}_local.pt',weights_only=True));model.requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()}
        q,b=cache(model,x);tq,tb=cache(model,tx)
        for delay,shuffle in [(0,False),(0,True),(1,False),(1,True)]:
            u,size=train(q,b,y,model,seed,lr,delay,shuffle)
            score=evaluate(tq,tb,ty,u,model)
            if delay==0 and not shuffle:
                ref=next(r for r in prior if r['router_seed']==seed and r['mode']=='joint_diagnostic' and r['arm']=='binary')
                assert abs(score['accuracy']-ref['accuracy'])<1e-7
            rows.append({'seed':seed,'backbone_seed':backbone,'delay_batches':delay,'shuffled':shuffle,'trace_tuple_bytes':size,**score})
            torch.save(u,out/f'{seed}_delay{delay}_shuffle{shuffle}.pt')
            print(seed,delay,shuffle,round(score['accuracy']*100,2),flush=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        (out/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Eligibility traces and delayed scalar feedback','',
           'Frozen previously trained branch computations and classifier. Joint router fitting revisits '
           'historical training images. This isolates credit assignment, not end-to-end continual learning.', '',
           '| Feedback | Correct trace/reward pairing | Shuffled pairing |', '|---|---:|---:|']
    for delay in [0,1]:
        cells=[]
        for shuffle in [False,True]:
            a=[r['accuracy']*100 for r in rows if r['delay_batches']==delay and r['shuffled']==shuffle]
            cells.append(f'{st.mean(a):.2f} ± {st.stdev(a):.2f}%')
        lines.append('| '+('Immediate' if not delay else 'One intervening batch')+' | '+' | '.join(cells)+' |')
    lines+=['','All settings inherited from validation-selected binary-reward experiment. No tuning on these '
            'test results. Each actual sampled choice leaves an eligibility tuple; its own reward arrives '
            'later and updates input-to-branch synapses. Shuffling pairs reward with other trials while '
            'keeping its distribution. A pending tuple requires199,680 bytes at batch128; two tuples may '
            'coexist during delivery. This is transient trial credit, not a persistent prototype bank. '
            'Memory for branch weights, router weights, caches and other tensors is additional.', '',
            'Immediate correctly paired results reproduce the previous independent experiment. Backbone '
            'state is asserted unchanged in all arms. Delayed score-function updates use probabilities at '
            'the time of the decision; parameter changes while waiting can introduce stale-gradient effects. '
            'This one-step pilot is not evidence of biological timescale fidelity.', '']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:10]),flush=True)


if __name__=='__main__':main()
