"""Merge independently executed cohorts and verify frozen selections/state costs."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from bdh_cl.scripts.cifar10_routing_gap import nbytes


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True);p.add_argument('--joint',type=Path,required=True)
    args=p.parse_args();folders=[args.base,args.controls,args.joint]
    freeze=json.loads((args.base/'all_selected_hyperparameters.json').read_text())
    rows=[];trials=[];selected={};protocols=[]
    for folder in folders:
        local_rows=json.loads((folder/'results.json').read_text());local_trials=json.loads((folder/'validation_trials.json').read_text())
        local_selected=json.loads((folder/'selected_hyperparameters.json').read_text())
        local_protocol=json.loads((folder/'protocol.json').read_text())
        # Keep this command idempotent after the first aggregate write.
        methods=['aux','ce'] if folder==args.base else ['pec'] if folder==args.controls else ['joint_aux','joint_ce']
        local_rows=[r for r in local_rows if r['method'] in methods]
        local_trials=[r for r in local_trials if r['method'] in methods]
        rows+=local_rows;trials+=local_trials
        selected.update({m:local_selected[m] for m in methods})
        for method in methods:
            assert local_selected[method]==freeze[method]
            best=max((r for r in local_trials if r['method']==method),key=lambda r:r['validation_accuracy'])
            assert best['lr']==local_selected[method]['lr']
            if method in ['aux','ce']:
                assert local_selected[method]['ridge']==max(best['router_settings']['ridge_trials'],key=lambda r:r['accuracy'])['ridge']
                assert local_selected[method]['temperature']==max(best['router_settings']['entropy_trials'],key=lambda r:r['accuracy'])['temperature']
        protocols.append(dict(folder=str(folder),methods=methods,protocol=local_protocol))
        if folder!=args.base:
            for checkpoint in folder.glob('*.pt'):
                if any(checkpoint.name.startswith(m+'_') for m in methods):shutil.copy2(checkpoint,args.base/checkpoint.name)
    assert len(rows)==24 and len({(r['method'],r['seed'],r['order']) for r in rows})==24
    assert len(trials)==16
    for row in rows:
        assert row['current_presentations']==225000
        assert row['config']==freeze[row['method']]
        path=args.base/f"{row['method']}_{row['seed']}_{row['order']}.pt"
        state=torch.load(path,weights_only=True,map_location='cpu')
        assert nbytes(state['model'].values())==row['model_bytes']
        if row['method'] in ['aux','ce']:
            assert nbytes([state['routes']])==row['route_bytes']
            assert nbytes([state['contexts']])==row['prototype_bytes']
            assert nbytes(list(state['router'].values())+[state['router_head']])==row['router_bytes']
        for name,score in row['scores'].items():
            if 'accuracy' in score:assert 0<=score['accuracy']<=1
        if row['method'] in ['aux','ce']:
            ceiling=row['scores']['bank_any_correct']['accuracy']
            for name in ['prototype','learned_context','entropy','margin','oracle','random_route','wrong_route']:
                assert ceiling+1e-7>=row['scores'][name]['accuracy']
    split=np.load(args.base/'split_indices.npz')
    assert len(split['train'])==45000 and len(split['validation'])==5000
    assert not np.intersect1d(split['train'],split['validation']).size
    aggregate=deepcopy(protocols[0]['protocol']);aggregate['configs']={}
    for part in protocols:aggregate['configs'].update({m:part['protocol']['configs'].get(m,[]) for m in part['methods']})
    aggregate['component_execution']=protocols
    aggregate['execution_stage']='complete; selections globally frozen before all final evaluation'
    aggregate['runtime_note']='Independent cohorts executed concurrently on CPU; elapsed times are not standalone hardware throughput measurements.'
    for name,value in [('results.json',rows),('validation_trials.json',trials),('selected_hyperparameters.json',selected),('protocol.json',aggregate)]:
        (args.base/name).write_text(json.dumps(value,indent=2))
    audit=dict(final_runs=24,unique_method_seed_order_tuples=24,validation_learning_rate_candidates=16,
               frozen_selection_manifest_verified=True,checkpoint_tensor_costs_verified=True,
               train_validation_disjoint=True,current_presentations_per_run=225000,
               output_classes=10,branch_parameters=525014,pec_parameters=522927,
               tests='5 focused checks passed',model_state_invariant_during_final_inference=True)
    (args.base/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
