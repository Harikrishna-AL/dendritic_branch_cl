"""Label-free feedback prediction audit; labels enter only transition scoring."""
import argparse
import json
import torch
from torch.nn import functional as F
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts.somatic_consensus import OUT,Model,Memory,digest,inference,family,global_key,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    selection=json.loads((OUT/'selection.json').read_text())
    records=json.loads((OUT/'results.json').read_text())
    if not args.available:assert len(records)==160
    path=OUT/'feedback_transition_audit.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    done={(r['seed'],r['order'],r['arm'],r['family']) for r in rows}
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    with torch.no_grad():
        for record in records:
            seed,name,arm=record['seed'],record['order'],record['arm']
            specs=[s for s in selection['readouts'][arm] if s['feedback'] and (seed,name,arm,family(s)) not in done]
            if not specs:continue
            checkpoint=OUT/f'{seed}_{name}_{arm}.pt'
            if args.available and not checkpoint.exists():continue
            state=torch.load(checkpoint,weights_only=True)
            m=Model(seed,arm);m.load_state_dict(state['model']);mem=Memory(m.dimension)
            for k,v in state['memory'].items():setattr(mem,k,v)
            before=digest(m,mem)
            responses=[];local=[];glob=[]
            for b in x.split(512):
                pp=m.patches(b);responses.append(m.response(pp));local.append(m.local_keys(pp));glob.append(global_key(b))
            response=torch.cat(responses);local=torch.cat(local);glob=torch.cat(glob)
            local=torch.einsum('bgd,gkd->bgk',local,F.normalize(mem.local_sums,dim=-1))
            glob=torch.einsum('bd,gkd->bgk',glob,F.normalize(mem.global_sums,dim=-1))
            local[:,mem.count==0]=-1000;glob[:,mem.count==0]=-1000
            for spec in specs:
                score=local if spec['cue']=='local' else glob
                first=[];second=[];tops=[]
                for a in range(0,len(x),512):
                    p0,top=inference(m,response[a:a+512],score[a:a+512],{**spec,'feedback':False})
                    p1,_=inference(m,response[a:a+512],score[a:a+512],spec)
                    first.append(p0);second.append(p1);tops.append(top)
                first=torch.cat(first);second=torch.cat(second);tops=torch.cat(tops)
                a=first==y;b=second==y
                saved=float((~a&b).float().mean());lost=float((a&~b).float().mean())
                net=float(b.float().mean()-a.float().mean())
                assert abs(net-(saved-lost))<1e-6
                original=next(r for r in record['methods'] if all(r[k]==spec[k] for k in spec))
                assert abs(float(b.float().mean())-original['accuracy'])<1e-7
                rows.append({'seed':seed,'order':name,'arm':arm,'family':family(spec),**spec,
                             'before_accuracy':float(a.float().mean()),'after_accuracy':float(b.float().mean()),
                             'wrong_to_correct':saved,'correct_to_wrong':lost,'net_change':net,
                             'prediction_change_rate':float((first!=second).float().mean()),
                             'first_top2_true_class_coverage':float((tops==y[:,None]).any(-1).float().mean()),
                             'final_prediction_outside_initial_top2':float((tops!=second[:,None]).all(-1).float().mean())})
                save('feedback_transition_audit.json',rows)
            assert digest(m,mem)==before
    if args.available and len(rows)<640:
        print('Audited feedback comparisons:',len(rows),flush=True);return
    assert len(rows)==640
    save('feedback_transition_completion.json',{'complete':True,'comparisons':len(rows),'same_backbone_fixed_weights':True,
        'no_test_time_writes':True,'predictions_reproduce_primary_records':True,
        'labels_used_only_after_predictions_for_scoring':True})


if __name__=='__main__':main()
