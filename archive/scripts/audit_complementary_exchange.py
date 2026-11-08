"""Frozen inference intervention; association memory and selected masks remain fixed."""
import json
import argparse
import torch
from bdh_cl.data.mnist import load_raw,SPLITS
from bdh_cl.scripts.complementary_exchange import OUT,Model,Memory,evaluate,save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--available',action='store_true',help='Audit completed models while training continues.')
    args=parser.parse_args()
    torch.set_num_threads(2)
    records=json.loads((OUT/'results.json').read_text())
    if not args.available:assert len(records)==240
    selected=json.loads((OUT/'selection.json').read_text())
    ridge=selected['chosen']['ridge']
    direct=selected['direct_chosen']['ridge']
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    saved=OUT/'inference_intervention.json'
    result=json.loads(saved.read_text()) if saved.exists() else []
    done={(r['seed'],r['order'],r['arm']) for r in result}
    for row in records:
        if not row['arm'].endswith('__neighbor'):continue
        seed,order_name,arm=row['seed'],row['order'],row['arm']
        if (seed,order_name,arm) in done:continue
        state=torch.load(OUT/f'{seed}_{order_name}_{arm}.pt',weights_only=True)
        m=Model(seed,arm);m.load_state_dict(state['model'])
        mem=Memory()
        for key,value in state['memory'].items():setattr(mem,key,value)
        order=SPLITS if order_name=='forward' else SPLITS[::-1]
        rr,diag=evaluate(m,mem,state['historical_bank_diagnostic_only'],x,y,order,[ridge],[direct],message_mode='self')
        primary=next(r for r in row['methods'] if r['method']=='ridge')
        changed=next(r for r in rr if r['method']=='ridge')
        result.append({'seed':seed,'order':order_name,'arm':arm,
                       'learned_neighbor_automatic':primary['accuracy'],
                       'same_weights_self_only_automatic':changed['accuracy'],
                       'learned_neighbor_historical':row['diagnostics']['historical']['accuracy'],
                       'same_weights_self_only_historical':diag['historical']['accuracy'],
                       'fixed_weights_and_associations':diag['evaluation_state_unchanged']})
        save('inference_intervention.json',result)
    if args.available and len(result)<120:
        print('Audited available frozen models:',len(result),flush=True)
        return
    assert len(result)==120
    save('inference_intervention_completion.json',{'complete':True,'frozen_models':len(result),
        'meaning':'Replaces neighbor evidence with self evidence at fixed final weights and fixed input-to-mask memory. Automatic masks are identical because retrieval uses same fixed input keys and stored statistics. This measures inference dependence, not superiority to retrained self-only networks.'})


if __name__=='__main__':main()
