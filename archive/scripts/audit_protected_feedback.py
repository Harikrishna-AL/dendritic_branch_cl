"""Bound remaining opportunities under the frozen validation-selected margin gate."""
import json
import torch
from bdh_cl.data.mnist import load_raw
from bdh_cl.scripts import complementary_exchange as base
from bdh_cl.scripts.protected_feedback import OUT,SOURCE,forward_candidates,save


def main():
    torch.set_num_threads(2)
    raw=load_raw('data_cache/mnist')
    x=torch.tensor(raw['test_images'].copy(),dtype=torch.float32).flatten(1)/255
    y=torch.tensor(raw['test_labels'].copy(),dtype=torch.long)
    records=json.loads((OUT/'results.json').read_text());rows=[]
    with torch.no_grad():
        for r in records:
            arm,seed,name=r['arm'],r['seed'],r['order']
            state=torch.load(SOURCE/f'{seed}_{name}_{arm}.pt',weights_only=True)
            m=base.Model(seed,arm);m.load_state_dict(state['model']);mem=base.Memory()
            for key,value in state['memory'].items():setattr(mem,key,value)
            before=base.digest(m,mem)
            cc=[forward_candidates(m,mem,b,'protected') for b in x.split(512)]
            c={key:torch.cat([v[key] for v in cc]) for key in ['first','second','margin','classes']}
            tau=r['selected']['tau']
            gate=torch.ones(len(y),dtype=torch.bool) if tau is None else c['margin']<tau
            first_correct=c['first']==y
            top2=(c['classes']==y[:,None]).any(-1)
            prediction=torch.where(gate,c['second'],c['first'])
            accuracy=float((prediction==y).float().mean())
            assert abs(accuracy-r['selected']['accuracy'])<1e-7
            initial=float(first_correct.float().mean())
            repairable=float((gate&~first_correct).float().mean())
            top2_repairable=float((gate&~first_correct&top2).float().mean())
            rows.append({'seed':seed,'order':name,'arm':arm,'tau':tau,
                'baseline_accuracy':initial,'selected_accuracy':accuracy,
                'triggered_initial_errors':repairable,
                'triggered_initial_errors_with_true_class_in_top2':top2_repairable,
                'absolute_any_corrector_ceiling_with_fixed_gate':initial+repairable,
                'top2_reranker_ceiling_with_fixed_gate':initial+top2_repairable,
                'top2_reranker_ceiling_without_gate':float(top2.float().mean()),
                'final_outside_initial_top2':float((c['classes']!=prediction[:,None]).all(-1).float().mean()),
                'state_unchanged':True})
            assert base.digest(m,mem)==before
    assert len(rows)==60
    save('gate_opportunity_audit.json',rows)
    save('gate_audit_completion.json',{'complete':True,'models':len(rows),'predictions_reproduced':True,
        'meaning':'Ceilings are true-label diagnostic opportunities, not implementable methods. Absolute gated ceiling applies to any correction that leaves untriggered examples unchanged; top2 ceiling additionally assumes output restricted to initial top2 classes. Actual final predictions allow all10.'})


if __name__=='__main__':main()
