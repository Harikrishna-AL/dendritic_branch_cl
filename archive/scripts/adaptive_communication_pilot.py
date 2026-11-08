"""Frozen-weight inference: do mismatch-triggered extra communication rounds help?"""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from bdh_cl.scripts.collective_context_pilot import Model,make_data

@torch.no_grad()
def run(model,x,rounds,exchange=True):
    states=[]
    for layer in model.layers:
        c=torch.tanh(layer.context(x).reshape(-1,128,4))
        residual=torch.zeros(len(x))
        for _ in range(rounds):
            left,right=(c.roll(1,1),c.roll(-1,1)) if exchange else (c,c)
            update=torch.tanh(layer.self_mix(c)+layer.left_mix(left)+layer.right_mix(right))
            nextc=.75*c+.25*update
            residual=(nextc-c).square().mean((1,2)).sqrt()
            c=nextc
        disagreement=(c-c.roll(1,1)).square().mean((1,2)).sqrt()
        gate=torch.sigmoid((c*layer.receiver[None]).sum(-1))
        x=(F.relu(layer.activity(x))*gate).reshape(-1,32,4).mean(-1)
        states.append({'context':c,'residual':residual,'disagreement':disagreement})
    return x@model.readout,states

@torch.no_grad()
def decode(val,test):
    y=torch.cat([-torch.ones(len(val[0])),torch.ones(len(val[1]))])
    target=torch.cat([torch.zeros(len(test[0])),torch.ones(len(test[1]))]).bool()
    a=torch.cat(val).flatten(1);b=torch.cat(test).flatten(1)
    mean=a.mean(0);std=a.std(0).clamp_min(.01)
    a=torch.cat([(a-mean)/std,torch.ones(len(a),1)],1)
    b=torch.cat([(b-mean)/std,torch.ones(len(b),1)],1)
    coef=torch.linalg.solve(a.T@a+10*torch.eye(a.shape[1]),a.T@y)
    return ((b@coef>0)==target).float().mean().item()

@torch.no_grad()
def mismatch(states):
    # Local disagreement, averaged for a global compute-budget trigger.
    return torch.stack([s['disagreement'] for s in states]).mean(0)

@torch.no_grad()
def adaptive(model,x,threshold,extra=32):
    base,bs=run(model,x,8);selected=mismatch(bs)>threshold
    prediction=base.clone();contexts=[s['context'].clone() for s in bs]
    if selected.any():
        longer,ls=run(model,x[selected],extra)
        prediction[selected]=longer
        for i,s in enumerate(ls):contexts[i][selected]=s['context']
    return prediction,contexts,selected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--source',default='results/collective_context_pilot_20261010')
    args=p.parse_args();torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'seeds':[0,1,2],
        'weights':'reuse learned-exchange checkpoints; no training, adaptation, or optimizer at inference',
        'fixed_rounds':[0,8,16,32,64],
        'adaptive':'8 rounds, then 32 total rounds for samples whose mean neighbor disagreement exceeds threshold; threshold is 95th percentile on first-regime training inputs at first checkpoint, then frozen',
        'control':'32 self-only recurrent rounds with neighbor inputs replaced by self; same weights',
        'diagnostics':'MSE, local disagreement, state update residual, per-regime trigger rate, validation-to-test linear cue decoding',
        'scope':'posthoc iteration of modules trained for eight rounds; additional rounds outside training horizon; trigger is a mismatch proxy, not validated task identity; inputs only at inference',
        'compute':'adaptive implementation recomputes selected samples from inputs; mean rounds measures conceptual 8/32 policy, actual recomputation costs 8+32 on selected samples',
        'selection':'32 total rounds and 95th percentile fixed before test; no choosing best test horizon'},indent=2))
    records=[]
    for seed in (0,1,2):
        saved=torch.load(Path(args.source)/f'{seed}_learned.pt',weights_only=True)
        train=make_data(seed+1000,1024);val=make_data(seed+2000,512);test=make_data(seed+3000,1024)
        model=Model('learned');model.load_state_dict(saved['stages'][0])
        threshold=torch.quantile(mismatch(run(model,train[0][0],8)[1]),.95).item()
        for checkpoint,state in (('first',saved['stages'][0]),('final',saved['stages'][1]),('joint',saved['joint'])):
            model.load_state_dict(state)
            for policy,rounds,exchange in [('fixed0',0,True),('fixed8',8,True),('fixed16',16,True),('fixed32',32,True),('fixed64',64,True),('self32',32,False),('adaptive',32,True)]:
                vals=[];tests=[];tasks=[]
                for dataset,contexts in ((val,vals),(test,tests)):
                    for x,y in dataset:
                        if policy=='adaptive':
                            prediction,c,trigger=adaptive(model,x,threshold)
                            disagreement=None;residual=None
                        else:
                            prediction,s=run(model,x,rounds,exchange);c=[v['context'] for v in s]
                            trigger=torch.zeros(len(x),dtype=torch.bool)
                            disagreement=mismatch(s).mean().item()
                            residual=torch.stack([v['residual'] for v in s]).mean().item()
                        contexts.append(c)
                        if dataset is test:
                            tasks.append({'mse':F.mse_loss(prediction,y).item(),
                                'trigger_rate':trigger.float().mean().item(),
                                'mean_rounds':8+24*trigger.float().mean().item() if policy=='adaptive' else rounds,
                                'disagreement':disagreement,'residual':residual})
                cues=[decode([v[i] for v in vals],[t[i] for t in tests]) for i in (0,1)]
                records.append({'seed':seed,'checkpoint':checkpoint,'policy':policy,'threshold':threshold,'tasks':tasks,'cue_decode':cues})
            print(seed,checkpoint,'completed',flush=True)
    (out/'results.json').write_text(json.dumps(records,indent=2))
    lines=['# Extra communication with frozen weights','','Means across three seeds, final sequential checkpoint. Lower MSE is better.','','| Policy | Old MSE | New MSE | Cue decode L2 | Old/new trigger | Mean rounds | Residual |','|---|---:|---:|---:|---:|---:|---:|']
    for policy in ('fixed0','fixed8','fixed16','fixed32','fixed64','self32','adaptive'):
        rows=[r for r in records if r['checkpoint']=='final' and r['policy']==policy]
        def avg(f):return sum(f(r) for r in rows)/len(rows)
        old,new=[avg(lambda r:r['tasks'][i]['mse']) for i in (0,1)]
        trigger=' / '.join(f'{avg(lambda r:r["tasks"][i]["trigger_rate"])*100:.1f}%' for i in (0,1))
        residual='—' if policy=='adaptive' else f'{avg(lambda r:sum(t["residual"] for t in r["tasks"])/2):.5f}'
        lines.append(f'| {policy} | {old:.3f} | {new:.3f} | {avg(lambda r:r["cue_decode"][1])*100:.1f}% | {trigger} | {avg(lambda r:sum(t["mean_rounds"] for t in r["tasks"])/2):.1f} | {residual} |')
    lines+=['','Conceptual adaptive round counts exclude recomputation overhead; implementation uses 8+32 rounds on triggered samples. No weights changed. First/joint checkpoint diagnostics are in results.json.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
