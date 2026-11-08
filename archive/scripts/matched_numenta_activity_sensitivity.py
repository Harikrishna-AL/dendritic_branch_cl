"""Exploratory activity controls after the fixed-5% matched-size pilot.

Same validation split, data budget, three LR trials per configuration and seeds.
Preserves coefficient counts; changes only native activity configuration.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import torch

from bdh_cl.scripts import matched_numenta_comparison as core


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args = p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source.py').write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2)
    original_construct = core.construct
    active = 'kw20'

    def construct(arm,seed):
        if arm!='numenta':
            return original_construct(arm,seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return core.DendriticMLP(784,10,[64,64],5,784,active!='relu',kw_percent_on=.2,
                                 weight_sparsity=.5,dendrite_weight_sparsity=0.,context_percent_on=.1)
    core.construct = construct
    protocol = {'scope':'Exploratory follow-up added after inspecting original matched-size run and validation acquisition instability. No test scores select hyperparameters; report all configurations. Not a preregistered confirmation.',
                'settings':['kw20','relu'],'tuning_seed':113,'seeds':[101,102,103],
                'learning_rates':[.0005,.001,.003],
                'description':'Same upstream DendriticMLP, [64,64], five segments, 50% feedforward sparsity, identical free/allocated coefficients. kw20 uses 20% activity (13 winners); relu uses upstream kw=False ablation. Native sparse initialization rescales second-layer weights for the configured activity, as upstream implements. Training, selection criterion, clipping, normalization and context access exactly reuse the matched comparison.',
                'baseline_directory':str(args.base)}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    raw = core.load_raw(str(core.ROOT/'data_cache/mnist'))
    x = (torch.tensor(raw['train_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    y = torch.tensor(raw['train_labels'].copy(),dtype=torch.long)
    tx = (torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255-.13062755)/.30810780
    ty = torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    split = np.load(args.base/'split_indices.npz')
    vx,vy = x[split['validation']],y[split['validation']]
    x,y = x[split['train']],y[split['train']]
    contexts = torch.load(args.base/'contexts.pt',weights_only=True)
    indices = [torch.nonzero(y//2==k,as_tuple=True)[0] for k in range(5)]
    trials,records,selected = [],[],{}
    for active in protocol['settings']:
        for rate in protocol['learning_rates']:
            model,routes,trace = core.train('numenta',113,rate,core.ORDERS['forward'],x,y,vx,vy,contexts,indices)
            before = core.state_hash(model)
            pred = core.predictions(model,'numenta',vx,contexts,routes)
            assert core.state_hash(model)==before
            accuracy = float((pred==vy).float().mean())
            trials.append({'setting':active,'rate':rate,'validation_accuracy':accuracy,**trace})
            (args.out/'validation_trials.json').write_text(json.dumps(trials,indent=2))
            print('TUNE ACTIVITY',active,rate,100*accuracy,flush=True)
        selected[active] = max((r for r in trials if r['setting']==active),key=lambda r:(r['validation_accuracy'],-r['rate']))['rate']
    (args.out/'selected_hyperparameters.json').write_text(json.dumps(selected,indent=2))
    # Freeze both selections before any official test prediction in this follow-up.
    for active in protocol['settings']:
        for seed in protocol['seeds']:
            for name,order in core.ORDERS.items():
                model,routes,trace = core.train('numenta',seed,selected[active],order,x,y,vx,vy,contexts,indices)
                scores = core.evaluate(model,'numenta',tx,ty,contexts,routes,order)
                records.append({'setting':active,'seed':seed,'order':name,'learning_rate':selected[active],
                                'scores':scores,**trace,**core.accounting(model)})
                torch.save(model.state_dict(),args.out/f'{active}_{seed}_{name}.pt')
                (args.out/'results.json').write_text(json.dumps(records,indent=2,allow_nan=False))
                print('FINAL ACTIVITY',active,seed,name,100*scores['inferred']['accuracy'],flush=True)
    lines = ['# Matched-size Numenta activity sensitivity','',protocol['scope'],'',protocol['description'],'',
             '| Activity | Learning rate | Inferred accuracy ± seed SD | Oracle accuracy |','|---|---:|---:|---:|']
    for active in protocol['settings']:
        rows = [r for r in records if r['setting']==active]
        vals = [st.mean(100*r['scores']['inferred']['accuracy'] for r in rows if r['seed']==s) for s in protocol['seeds']]
        lines.append(f'| {active} | {selected[active]} | {st.mean(vals):.2f} ± {st.stdev(vals):.2f}% | {st.mean(100*r["scores"]["oracle"]["accuracy"] for r in rows):.2f}% |')
    lines += ['', 'Seed SD follows averaging both task orders within seed. Same known training boundaries and unrestricted ten-class predictions. '
              'No SI, statistical batch clustering or classifier replay. This tests whether low activity alone explains the fixed-5% result; '
              'it does not establish the best possible Numenta configuration. Native initialization changes with activity as upstream specifies.','']
    (args.out/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
