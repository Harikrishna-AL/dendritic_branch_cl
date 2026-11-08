"""Direct held-out retention before/after newer electricity learning, no audit weight updates."""
import argparse,copy,json,statistics,collections
from pathlib import Path
import numpy as np
import torch
from bdh_cl.scripts.electricity_usefulness_pilot import Useful,cohort,run
from bdh_cl.scripts.electricity_memory_pilot import Forecast,features

VARIANTS=('predict_winner','quality_soft','quality_all','uniform_all','mlp','gru')

def audit(model,d,start,end,error_feedback=True,carry=False):
    model=copy.deepcopy(model)
    if not carry:model.reset()
    weights=[p.detach().clone() for p in model.parameters()];q=collections.deque();rows=[];feedback=getattr(model,'feedback',False) and error_feedback
    with torch.no_grad():
        for t in range(start,end-24):
            while q and q[0][0]<=t:
                when,ticket,base=q.popleft();model.observe(ticket,d['y'][when]-base)
            residual,ticket=model.forecast(features(d['y'],t));pred=d['y'][t]+residual;err=(pred-d['y'][t+24]).square()
            if feedback:q.append((t+24,ticket,d['y'][t].clone()))
            rows.append({'mse':err.mean().item(),'mae':(pred-d['y'][t+24]).abs().mean().item(),'per_client':err.tolist()})
    for a,b in zip(weights,model.parameters()):assert torch.equal(a,b),'audit changed slow weights'
    return {'mse':statistics.mean(r['mse'] for r in rows),'mae':statistics.mean(r['mae'] for r in rows),
            'first24_mse':statistics.mean(r['mse'] for r in rows[:24]),'first_week_mse':statistics.mean(r['mse'] for r in rows[:168]),
            'after_week_mse':statistics.mean(r['mse'] for r in rows[168:]),
            'per_client_mse':np.mean([r['per_client'] for r in rows],axis=0).tolist(),'hours':len(rows)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',default='8,9,10,11,12');args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);d=cohort(16)
    years=d['year'];months=d['month'];a_end=int(np.flatnonzero((years==2012)&(months==10))[0]);a_test_end=int(np.flatnonzero(years==2013)[0])
    b_start=a_test_end+720;b_end=int(np.flatnonzero((years==2013)&(months==10))[0]);b_test_end=int(np.flatnonzero(years==2014)[0]);b_length=b_end-b_start
    (out/'source.py').write_text(Path(__file__).read_text())
    (out/'protocol.json').write_text(json.dumps({**vars(args),'variants':VARIANTS,'clients':d['clients'],
        'A':'train2012 after720h calibration untilOct1; heldoutOct-Dec2012',
        'B':'train2013 afterfirst720h untilOct1; heldoutOct-Dec2013',
        'repeat_A':'afterA, repeatA training prefix for exactlysame numberhoursas B; continue optimizer and runtime state',
        'controls':'evaluate exactsame older/newer heldouts at afterA,afterB andafterrepeatA; all auditweights frozen; livequalityfeedback permitted onlyafter24h',
        'primary':'older heldout MSEafterB minus MSEafterA; positive =forgetting; reportrelativechange andnew-task acquisition; compareequalbudgetrepeatA to separate newdistribution interference fromoptimizer drift',
        'memory_audits':'independent coldreset for matched pre/post slow-retention audit; carryB-state old-return audit oncopy; qualityerrorfeedbackdisabled audit; no slowrelearning in any audit',
        'training':'Adam.001 every8matured tickets,clipping5; onepass; no taskID/calendar/boundary modelinput; nooptimizerorfastreset at A->B/repeatA; heldoutsnotusedforgradientupdates',
        'boundary_bookkeeping':'each excluded-gap trainingsegment ends with pending24h forecasts dropped so heldouttargetsnever train; partial<8matured batch also dropped; samepolicyinB/repeatA. This is scheduling aroundheldoutgaps,not anoraclecontextgate',
        'limits':'same customercohort aspreviousforecastconfirmation but new seeds and newly specifiedretention protocol; no tuning; natural yearsnotguaranteed distinct/conflictingtasks; audits reuse identical windows and retrospective oldperiod,not a live chronological A-B-A world; persistentstatebudgets differ; GRUone-steptruncation'},indent=2))
    records=[]
    for seed in map(int,args.seeds.split(',')):
        for variant in VARIANTS:
            torch.manual_seed(seed);m=Forecast(variant) if variant in ('mlp','gru') else Useful(variant);m.reset();opt=torch.optim.Adam(m.parameters(),lr=.001)
            run(m,opt,d,720,a_end,True,False)
            oldstate=copy.deepcopy(m.state_dict());opstate=copy.deepcopy(opt.state_dict());recent=copy.deepcopy(m.recent)
            row={'seed':seed,'variant':variant,'after_A':{'old':audit(m,d,a_end,a_test_end),'new':audit(m,d,b_end,b_test_end)}}
            torch.save({'model':oldstate,'optimizer':opstate,'recent':recent},out/f'{seed}_{variant}_afterA.pt')
            for label,start,end in [('B',b_start,b_end),('repeatA',720,720+b_length)]:
                m.load_state_dict(oldstate);m.recent=copy.deepcopy(recent);opt.load_state_dict(opstate)
                run(m,opt,d,start,end,True,False)
                row['after_'+label]={'old':audit(m,d,a_end,a_test_end),'new':audit(m,d,b_end,b_test_end)}
                if label=='B':
                    row['old_return_carry']=audit(m,d,a_end,a_test_end,carry=True)
                    if getattr(m,'feedback',False):row['old_no_error_feedback']=audit(m,d,a_end,a_test_end,error_feedback=False)
                torch.save(m.state_dict(),out/f'{seed}_{variant}_after{label}.pt')
            row['forgetting']=row['after_B']['old']['mse']-row['after_A']['old']['mse']
            row['new_gain']=row['after_A']['new']['mse']-row['after_B']['new']['mse']
            row['repeatA_old_change']=row['after_repeatA']['old']['mse']-row['after_A']['old']['mse']
            records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
            print(seed,variant,'forgetting',round(row['forgetting'],4),'newgain',round(row['new_gain'],4),flush=True)
    lines=['# Direct electricity retention audit','','Five new initialization seeds; old/new heldouts never train slow weights. Positive forgetting means older performance worsened after newer learning. Negative means backward transfer.','','| Model | Old afterA | Old afterB | Forgetting | New beforeB | New afterB | New gain | Old change repeatA |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        rs=[r for r in records if r['variant']==v]
        values=[statistics.mean(r[stage][task]['mse'] for r in rs) for stage,task in [('after_A','old'),('after_B','old')]]
        values+=[statistics.mean(r['forgetting'] for r in rs)]
        values+=[statistics.mean(r[stage]['new']['mse'] for r in rs) for stage in ('after_A','after_B')]
        values+=[statistics.mean(r['new_gain'] for r in rs),statistics.mean(r['repeatA_old_change'] for r in rs)]
        lines.append('| '+v+' | '+' | '.join(f'{a:.4f}' for a in values)+' |')
    lines+=['','All audit weights are verified unchanged. Memory can use observed inputs and delayed factual feedback. Cold-reset audits measure retained forecasting ability with fresh runtime context; the carry-state audit additionally measures return after B context. This is not proof of autonomous task discovery or a natural continuous A-B-A sequence.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
